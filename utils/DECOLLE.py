import math
from dataclasses import dataclass
from typing import Callable, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm


def get_module(model, path):
    module = model
    for part in path.split('.'):
        if not part:
            continue

        if isinstance(module, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            module = module[int(part)]
        elif isinstance(module, nn.ModuleDict):
            module = module[part]
        else:
            module = getattr(module, part)
    return module


def align_for_delay(target, prediction, delay=0):
    if delay == 0:
        return target, prediction

    if target.ndim < 3 or prediction.ndim < 3:
        raise ValueError('delay requires temporal tensors with time on the last dimension')

    if target.size(-1) != prediction.size(-1):
        raise ValueError('target and prediction must have the same sequence length')

    if abs(delay) >= target.size(-1):
        raise ValueError('abs(delay) must be smaller than the sequence length')

    if delay > 0:
        return target[..., :-delay], prediction[..., delay:]
    future = -delay
    return target[..., future:], prediction[..., :-future]


def default_spike_extractor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value

    raise TypeError('Could not extract spikes from module output')


def detach_value(value):
    if torch.is_tensor(value):
        return value.detach()

    if isinstance(value, tuple):
        return tuple(detach_value(v) for v in value)

    if isinstance(value, list):
        return [detach_value(v) for v in value]

    if isinstance(value, dict):
        return {k: detach_value(v) for k, v in value.items()}

    return value


def spike_fbeta(pred_spikes, target_spikes, beta=2.0, eps=1e-8):
    dims = tuple(range(1, pred_spikes.ndim))

    tp = (pred_spikes * target_spikes).sum(dim=dims)
    fp = (pred_spikes * (1 - target_spikes)).sum(dim=dims)
    fn = ((1 - pred_spikes) * target_spikes).sum(dim=dims)

    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp + eps

    return numerator / denominator


def spike_fbeta_loss(pred_spikes, target_spikes, beta=2.0, eps=1e-8):
    return 1 - spike_fbeta(pred_spikes, target_spikes, beta=beta, eps=eps).mean()


def _dilate_spikes(spikes, tolerance):
    if tolerance <= 0:
        return spikes

    batch_size, channels, timesteps = spikes.shape
    flat = spikes.reshape(batch_size * channels, 1, timesteps)
    dilated = F.max_pool1d(flat, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance)

    return dilated.reshape(batch_size, channels, timesteps)


def nearby_spike_scores(pred_spikes, target_spikes, tolerance=1, eps=1e-8):
    target_near = _dilate_spikes(target_spikes, tolerance)
    pred_near = _dilate_spikes(pred_spikes, tolerance)

    dims = tuple(range(1, pred_spikes.ndim))

    matched_predictions = (pred_spikes * target_near).sum(dim=dims)
    prediction_count = pred_spikes.sum(dim=dims)

    matched_targets = (target_spikes * pred_near).sum(dim=dims)
    target_count = target_spikes.sum(dim=dims)

    precision = matched_predictions / (prediction_count + eps)
    recall = matched_targets / (target_count + eps)

    return precision, recall


def nearby_spike_f1(pred_spikes, target_spikes, tolerance=1, eps=1e-8):
    precision, recall = nearby_spike_scores(pred_spikes, target_spikes, tolerance=tolerance, eps=eps)

    return 2 * precision * recall / (precision + recall + eps)


def nearby_spike_fbeta(pred_spikes, target_spikes, tolerance=1, beta=2.0, eps=1e-8):
    precision, recall = nearby_spike_scores(pred_spikes, target_spikes, tolerance=tolerance, eps=eps)

    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = beta_sq * precision + recall + eps

    return numerator / denominator


def nearby_spike_fbeta_loss(pred_spikes, target_spikes, tolerance=1, beta=2.0, eps=1e-8):
    return 1 - nearby_spike_fbeta(
        pred_spikes,
        target_spikes,
        tolerance=tolerance,
        beta=beta,
        eps=eps
    ).mean()


def spike_count_loss(pred_spikes, target_spikes):
    timesteps = pred_spikes.size(-1)

    pred_count = pred_spikes.sum(dim=-1) / timesteps
    target_count = target_spikes.sum(dim=-1) / timesteps

    return F.mse_loss(pred_count, target_count)


def make_spike_output_loss(
    exact_weight=1.0,
    fbeta_weight=1.0,
    nearby_fbeta_weight=1.0,
    count_weight=3.0,
    nearby_tolerance=1,
    beta=2.0
):
    def output_loss(pred_spikes, target_spikes):
        exact = F.mse_loss(pred_spikes, target_spikes)
        exact_fbeta = spike_fbeta_loss(pred_spikes, target_spikes, beta=beta)
        nearby_fbeta = nearby_spike_fbeta_loss(
            pred_spikes,
            target_spikes,
            tolerance=nearby_tolerance,
            beta=beta
        )
        count = spike_count_loss(pred_spikes, target_spikes)

        return (
            exact_weight * exact
            + fbeta_weight * exact_fbeta
            + nearby_fbeta_weight * nearby_fbeta
            + count_weight * count
        )

    output_loss.__name__ = 'spike_output_loss'
    output_loss.fbeta_beta = beta
    output_loss.nearby_tolerance = nearby_tolerance

    return output_loss


def make_decolle_losses(sites, hidden_loss, output_loss):
    losses = []

    for site in sites:
        if site.direct_output:
            losses.append(output_loss)
        else:
            losses.append(hidden_loss)

    return losses


def reconstruction_metrics(outputs, targets):
    outputs = outputs.reshape(outputs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    tps = ((outputs == 1) & (targets == 1)).sum()
    fps = ((outputs == 1) & (targets == 0)).sum()
    fns = ((outputs == 0) & (targets == 1)).sum()

    return tps, fps, fns


def nearby_reconstruction_metrics(outputs, targets, tolerance=1):
    outputs = (outputs > 0.5).float()
    targets = (targets > 0.5).float()

    target_near = _dilate_spikes(targets, tolerance)
    output_near = _dilate_spikes(outputs, tolerance)

    matched_predictions = (outputs * target_near).sum()
    prediction_count = outputs.sum()

    matched_targets = (targets * output_near).sum()
    target_count = targets.sum()

    return matched_predictions, prediction_count, matched_targets, target_count


def plot_reconstruction(inputs, outputs, epoch, criterion, tau=10.0, delay=0):
    aligned_target, aligned_output = align_for_delay(inputs, outputs, delay)
    target = aligned_target[0].detach().cpu()
    output = aligned_output[0].detach().cpu()
    criterion_name = getattr(criterion, '__name__', criterion.__class__.__name__).lower()

    if 'rossum' in criterion_name:
        alpha = torch.exp(torch.tensor(-1.0 / tau))
        pred_trace = torch.zeros_like(output)
        target_trace = torch.zeros_like(target)
        pred_trace[:, 0] = output[:, 0]
        target_trace[:, 0] = target[:, 0]
        for t in range(1, output.shape[-1]):
            pred_trace[:, t] = alpha * pred_trace[:, t - 1] + output[:, t]
            target_trace[:, t] = alpha * target_trace[:, t - 1] + target[:, t]
        timestep_loss = ((pred_trace - target_trace) ** 2).mean(dim=0)
    else:
        timestep_loss = ((output - target) ** 2).mean(dim=0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].imshow(target, aspect='auto', interpolation='nearest', origin='lower')
    if delay > 0:
        task_title = f'Reconstruct {delay} timestep(s) in the past'
    elif delay < 0:
        task_title = f'Predict {-delay} timestep(s) into the future'
    else:
        task_title = 'Reconstruction'

    axes[0].set_title(f'Target - Epoch {epoch + 1} | {task_title}')
    axes[0].set_ylabel('Channel')

    axes[1].imshow(output, aspect='auto', interpolation='nearest', origin='lower')
    axes[1].set_title(f'Reconstruction - Epoch {epoch + 1}')
    axes[1].set_ylabel('Channel')

    axes[2].plot(timestep_loss)
    axes[2].set_title('Exact MSE Per Timestep')
    axes[2].set_ylabel('Loss')
    axes[2].set_xlabel('Timestep')
    axes[2].grid()

    plt.tight_layout()
    plt.show()


@dataclass
class DECOLLESite:
    name: str
    spike_module: str
    trainable_modules: Sequence[str]
    cut_module: str | None = None
    state_argnums: Sequence[int] = ()
    spike_extractor: Callable | None = None
    direct_output: bool = False


class DECOLLETrainer(nn.Module):
    """
    Generic DECOLLE-style local trainer.

    Each site defines:
      - where spikes are observed,
      - which parameters that local loss may update,
      - where the graph is cut from previous layers,
      - which state arguments should be detached to prevent temporal BPTT.

    Losses may be one callable for every site or a list of length L.
    """

    def __init__(self, model, sites, losses, optimizer, readout_features=None, train_readouts=False, detach_between_sites=True, detach_states=True, loss_weights=None):
        super().__init__()
        self.model = model
        self.sites = self._resolve_sites(sites)
        self.losses = self._resolve_losses(losses)
        self.optimizer = optimizer
        self.readout_features = readout_features
        self.train_readouts = train_readouts
        self.detach_between_sites = detach_between_sites
        self.detach_states = detach_states
        self.readouts = nn.ModuleDict()

        if loss_weights is None:
            self.loss_weights = [1.0] * len(self.sites)
        else:
            if len(loss_weights) != len(self.sites):
                raise ValueError('loss_weights must have length L')
            self.loss_weights = list(loss_weights)

        self._validate_sites()

    def _resolve_sites(self, sites):
        if sites is None and hasattr(self.model, 'get_decolle_sites'):
            sites = self.model.get_decolle_sites()
        elif sites is None and hasattr(self.model, 'decolle_sites'):
            sites = self.model.decolle_sites

        if sites is None:
            raise ValueError('Pass sites or implement model.get_decolle_sites()')

        resolved = []
        for site in sites:
            if isinstance(site, DECOLLESite):
                resolved.append(site)
            else:
                resolved.append(DECOLLESite(**site))

        if not resolved:
            raise ValueError('At least one DECOLLE site is required')

        return resolved

    def _resolve_losses(self, losses):
        if callable(losses):
            return [losses] * len(self.sites)

        losses = list(losses)
        if len(losses) != len(self.sites):
            raise ValueError(f'Expected one loss or {len(self.sites)} losses, got {len(losses)}')

        if not all(callable(loss_fn) for loss_fn in losses):
            raise TypeError('Every loss must be callable')

        return losses

    def _validate_sites(self):
        names = set()
        used_params = {}
        for site in self.sites:
            if site.name in names:
                raise ValueError(f'Duplicate site name: {site.name}')

            names.add(site.name)
            get_module(self.model, site.spike_module)

            if not site.trainable_modules:
                raise ValueError(f'{site.name} has no trainable_modules')

            for path in site.trainable_modules:
                module = get_module(self.model, path)
                for param in module.parameters():
                    if id(param) in used_params:
                        raise ValueError(f'Parameter shared by sites {used_params[id(param)]} and {site.name}')
                    used_params[id(param)] = site.name

            if site.cut_module is not None:
                get_module(self.model, site.cut_module)

    def _site_parameters(self, site):
        params = []
        seen = set()
        for path in site.trainable_modules:
            for param in get_module(self.model, path).parameters():
                if param.requires_grad and id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))
        return params

    def _register_hooks(self, records):
        handles = []
        for site in self.sites:
            spike_module = get_module(self.model, site.spike_module)
            extractor = site.spike_extractor or default_spike_extractor

            def record_hook(module, args, output, site=site, extractor=extractor):
                records[site.name].append(extractor(output))
            handles.append(spike_module.register_forward_hook(record_hook))

            if self.detach_states and site.state_argnums:
                state_argnums = tuple(site.state_argnums)

                def state_hook(module, args, state_argnums=state_argnums):
                    args = list(args)
                    for index in state_argnums:
                        if -len(args) <= index < len(args):
                            args[index] = detach_value(args[index])
                    return tuple(args)
                handles.append(spike_module.register_forward_pre_hook(state_hook))

        if self.detach_between_sites:
            used = set()
            for site in self.sites:
                path = site.cut_module or site.trainable_modules[0]

                if path in used:
                    continue
                used.add(path)
                module = get_module(self.model, path)

                def cut_hook(module, args):
                    return tuple((detach_value(arg) for arg in args))
                handles.append(module.register_forward_pre_hook(cut_hook))
        return handles

    def _stack_spikes(self, records, expected_timesteps, site_name):
        if not records:
            raise RuntimeError(f'Site {site_name} did not execute')

        if expected_timesteps is not None and len(records) == expected_timesteps:
            spikes = []
            for value in records:
                if value.ndim == 1:
                    value = value.unsqueeze(0)

                if value.ndim != 2:
                    raise ValueError(f'{site_name} returned {tuple(value.shape)} per timestep; expected [B, H]')
                spikes.append(value)
            return torch.stack(spikes, dim=-1)

        if len(records) == 1:
            value = records[0]
            if value.ndim == 2:
                return value.unsqueeze(-1)

            if value.ndim == 3:
                return value

        raise ValueError(f'Could not convert {site_name} activations to [B, H, T]. Use a custom spike_extractor if needed.')

    def _get_readout(self, site, hidden_size, target, spikes):
        if site.name not in self.readouts:
            if self.readout_features is not None:
                output_size = self.readout_features
            elif target.ndim >= 2:
                output_size = target.size(1)
            else:
                raise ValueError('Pass readout_features for non-[B,C,...] targets')

            readout = nn.Linear(hidden_size, output_size, bias=False).to(device=spikes.device, dtype=spikes.dtype)

            with torch.no_grad():
                nn.init.normal_(readout.weight, mean=0.0, std=1.0 / math.sqrt(max(1, hidden_size)))

            if not self.train_readouts:
                for param in readout.parameters():
                    param.requires_grad_(False)

            self.readouts[site.name] = readout

            if self.train_readouts:
                self.optimizer.add_param_group({'params': list(readout.parameters())})

        return self.readouts[site.name]

    def _forward(self, inputs, target):
        records = {site.name: [] for site in self.sites}
        handles = self._register_hooks(records)
        try:
            model_output = self.model(inputs)
        finally:
            for handle in handles:
                handle.remove()

        if inputs.ndim >= 3:
            expected_timesteps = inputs.size(-1)
        else:
            expected_timesteps = None

        local_spikes = []
        local_predictions = []

        for site in self.sites:
            spikes = self._stack_spikes(records[site.name], expected_timesteps, site.name)

            if site.direct_output:
                if target.ndim >= 2 and spikes.size(1) != target.size(1):
                    raise ValueError(
                        f"Direct-output site '{site.name}' has {spikes.size(1)} channels "
                        f"but the target has {target.size(1)} channels"
                    )

                prediction = spikes
            else:
                readout = self._get_readout(site, spikes.size(1), target, spikes)
                prediction = readout(spikes.transpose(1, 2)).transpose(1, 2)

            local_spikes.append(spikes)
            local_predictions.append(prediction)

        return model_output, local_predictions, local_spikes

    def train_batch(self, inputs, target=None, delay=0, clip_grad_norm=None):
        self.model.train()

        if target is None:
            target = inputs

        self.optimizer.zero_grad(set_to_none=True)
        model_output, predictions, spikes = self._forward(inputs, target)

        local_losses = []
        for loss_fn, weight, prediction in zip(self.losses, self.loss_weights, predictions):
            aligned_target, aligned_prediction = align_for_delay(target, prediction, delay)
            loss = loss_fn(aligned_prediction, aligned_target)
            if loss.ndim != 0:
                loss = loss.mean()

            local_losses.append(loss * weight)

        for index, (site, loss) in enumerate(zip(self.sites, local_losses)):
            params = self._site_parameters(site)

            if self.train_readouts and not site.direct_output:
                params += [p for p in self.readouts[site.name].parameters() if p.requires_grad]

            grads = torch.autograd.grad(loss, params, retain_graph=index < len(local_losses) - 1, allow_unused=True)
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.grad = grad.detach()

        if clip_grad_norm is not None:
            params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params_with_grad, clip_grad_norm)

        self.optimizer.step()

        detached_losses = [loss.detach() for loss in local_losses]
        if torch.is_tensor(model_output):
            model_output = model_output.detach()

        return {'loss': torch.stack(detached_losses).mean().item(), 'local_losses': [loss.item() for loss in detached_losses], 'local_predictions': [value.detach() for value in predictions], 'local_spikes': [value.detach() for value in spikes], 'model_output': model_output}

    @torch.no_grad()
    def evaluate_metrics(
        self,
        te_dl,
        device,
        delay=0,
        epoch=None,
        plot=True,
        plot_criterion=None,
        tau=10.0,
        nearby_tolerance=1,
        fbeta_beta=2.0
    ):
        self.model.eval()

        tp = 0
        fp = 0
        fn = 0

        nearby_matched_predictions = 0
        nearby_prediction_count = 0
        nearby_matched_targets = 0
        nearby_target_count = 0

        output_spikes = 0
        target_spikes = 0

        for batch_idx, batch in enumerate(te_dl):
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            outputs = self.model(inputs)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            aligned_target, aligned_output = align_for_delay(inputs, outputs, delay)

            batch_tp, batch_fp, batch_fn = reconstruction_metrics(aligned_output, aligned_target)
            tp += batch_tp.item()
            fp += batch_fp.item()
            fn += batch_fn.item()

            batch_near_pred, batch_pred_count, batch_near_target, batch_target_count = nearby_reconstruction_metrics(
                aligned_output,
                aligned_target,
                tolerance=nearby_tolerance
            )

            nearby_matched_predictions += batch_near_pred.item()
            nearby_prediction_count += batch_pred_count.item()
            nearby_matched_targets += batch_near_target.item()
            nearby_target_count += batch_target_count.item()

            output_spikes += aligned_output.sum().item()
            target_spikes += aligned_target.sum().item()

            if batch_idx == 0 and plot:
                if plot_criterion is None:
                    criterion = self.losses[-1]
                else:
                    criterion = plot_criterion

                if epoch is None:
                    plot_epoch = 0
                else:
                    plot_epoch = epoch

                plot_reconstruction(inputs, outputs, plot_epoch, criterion, tau=tau, delay=delay)

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        beta_sq = fbeta_beta ** 2
        fbeta_denominator = beta_sq * precision + recall

        if fbeta_denominator > 0:
            fbeta = (1 + beta_sq) * precision * recall / fbeta_denominator
        else:
            fbeta = 0.0

        if nearby_prediction_count > 0:
            nearby_precision = nearby_matched_predictions / nearby_prediction_count
        else:
            nearby_precision = 0.0

        if nearby_target_count > 0:
            nearby_recall = nearby_matched_targets / nearby_target_count
        else:
            nearby_recall = 0.0

        if nearby_precision + nearby_recall > 0:
            nearby_f1 = 2 * nearby_precision * nearby_recall / (nearby_precision + nearby_recall)
        else:
            nearby_f1 = 0.0

        nearby_fbeta_denominator = beta_sq * nearby_precision + nearby_recall

        if nearby_fbeta_denominator > 0:
            nearby_fbeta = (
                (1 + beta_sq)
                * nearby_precision
                * nearby_recall
                / nearby_fbeta_denominator
            )
        else:
            nearby_fbeta = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fbeta': fbeta,
            'nearby_precision': nearby_precision,
            'nearby_recall': nearby_recall,
            'nearby_f1': nearby_f1,
            'nearby_fbeta': nearby_fbeta,
            'output_spikes': output_spikes,
            'target_spikes': target_spikes
        }

    @torch.no_grad()
    def evaluate_f1(
        self,
        te_dl,
        device,
        delay=0,
        epoch=None,
        plot=True,
        plot_criterion=None,
        tau=10.0,
        nearby_tolerance=1,
        fbeta_beta=2.0
    ):
        metrics = self.evaluate_metrics(
            te_dl,
            device=device,
            delay=delay,
            epoch=epoch,
            plot=plot,
            plot_criterion=plot_criterion,
            tau=tau,
            nearby_tolerance=nearby_tolerance,
            fbeta_beta=fbeta_beta
        )

        return metrics['f1']

    def fit(
        self,
        tr_dl,
        te_dl,
        device,
        num_epochs=100,
        delay=0,
        plot=True,
        plot_criterion=None,
        tau=10.0,
        clip_grad_norm=None,
        nearby_tolerance=1,
        fbeta_beta=2.0
    ):
        device = torch.device(device)
        self.model.to(device)

        best_f1 = -1.0
        best_epoch = -1
        best_state = None

        results = {
            'epoch': [],
            'train_loss': [],
            'train_local_losses': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': [],
            'test_fbeta': [],
            'test_nearby_precision': [],
            'test_nearby_recall': [],
            'test_nearby_f1': [],
            'test_nearby_fbeta': [],
            'output_spikes': [],
            'target_spikes': [],
            'best_f1': None,
            'best_epoch': None
        }

        for epoch in range(num_epochs):
            self.model.train()

            running_loss = 0.0
            running_local_losses = [0.0] * len(self.sites)
            batch_count = 0

            train_bar = tqdm.tqdm(tr_dl, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)

            for batch in train_bar:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)

                batch_result = self.train_batch(inputs, target=inputs, delay=delay, clip_grad_norm=clip_grad_norm)

                running_loss += batch_result['loss']
                batch_count += 1

                for i, local_loss in enumerate(batch_result['local_losses']):
                    running_local_losses[i] += local_loss

                train_loss = running_loss / batch_count
                output_loss = running_local_losses[-1] / batch_count

                train_bar.set_postfix(loss=f'{train_loss:.4f}', output=f'{output_loss:.4f}')

            train_loss = running_loss / max(batch_count, 1)
            epoch_local_losses = [loss / max(batch_count, 1) for loss in running_local_losses]

            metrics = self.evaluate_metrics(
                te_dl,
                device=device,
                delay=delay,
                epoch=epoch,
                plot=plot,
                plot_criterion=plot_criterion,
                tau=tau,
                nearby_tolerance=nearby_tolerance,
                fbeta_beta=fbeta_beta
            )

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

            results['epoch'].append(epoch + 1)
            results['train_loss'].append(train_loss)
            results['train_local_losses'].append(epoch_local_losses)
            results['test_precision'].append(metrics['precision'])
            results['test_recall'].append(metrics['recall'])
            results['test_f1'].append(metrics['f1'])
            results['test_fbeta'].append(metrics['fbeta'])
            results['test_nearby_precision'].append(metrics['nearby_precision'])
            results['test_nearby_recall'].append(metrics['nearby_recall'])
            results['test_nearby_f1'].append(metrics['nearby_f1'])
            results['test_nearby_fbeta'].append(metrics['nearby_fbeta'])
            results['output_spikes'].append(metrics['output_spikes'])
            results['target_spikes'].append(metrics['target_spikes'])

            local_string = ' | '.join(f'L{i}: {loss:.4f}' for i, loss in enumerate(epoch_local_losses))

            tqdm.tqdm.write(
                f'Epoch [{epoch + 1}/{num_epochs}] | '
                f'{local_string} | '
                f'P: {metrics["precision"]:.4f} | '
                f'R: {metrics["recall"]:.4f} | '
                f'F1: {metrics["f1"]:.4f}\n'
                f'F{fbeta_beta:g}: {metrics["fbeta"]:.4f} | '
                f'Nearby F1(±{nearby_tolerance}): {metrics["nearby_f1"]:.4f} | '
                f'Nearby F{fbeta_beta:g}(±{nearby_tolerance}): {metrics["nearby_fbeta"]:.4f} | '
                f'Spikes: {int(metrics["output_spikes"])}/{int(metrics["target_spikes"])} | '
                f'Best F1: {best_f1:.4f}'
            )

        self.best_f1 = best_f1
        self.best_epoch = best_epoch
        results['best_f1'] = best_f1

        if best_epoch >= 0:
            results['best_epoch'] = best_epoch + 1
        else:
            results['best_epoch'] = None

        print(f"Best F1: {best_f1:.4f} (epoch {results['best_epoch']})")

        return best_state, results

    @torch.no_grad()
    def eval_batch(self, inputs, target=None, delay=0):
        self.model.eval()

        if target is None:
            target = inputs

        model_output, predictions, spikes = self._forward(inputs, target)

        local_losses = []
        for loss_fn, weight, prediction in zip(self.losses, self.loss_weights, predictions):
            aligned_target, aligned_prediction = align_for_delay(target, prediction, delay)
            loss = loss_fn(aligned_prediction, aligned_target)
            if loss.ndim != 0:
                loss = loss.mean()

            local_losses.append(loss * weight)

        return {'loss': torch.stack(local_losses).mean().item(), 'local_losses': [loss.item() for loss in local_losses], 'local_predictions': predictions, 'local_spikes': spikes, 'model_output': model_output}


def make_multilayer_ae_sites(model):
    """
    Builds DECOLLE sites for:
        model.encoder_layers
        model.decoder_layers
        model.lif_layers

    The corresponding Linear weights and trainable LIF beta/threshold values
    are updated by the same local loss.

    Hidden sites use fixed random readouts. The final decoder site is marked as
    direct_output=True so its local loss is applied directly to the real network
    output spikes rather than to a random projection of them.
    """
    sites = []
    lif_idx = 0

    for i in range(len(model.encoder_layers)):
        sites.append(
            DECOLLESite(
                name=f'encoder_{i}',
                spike_module=f'lif_layers.{lif_idx}',
                trainable_modules=(f'encoder_layers.{i}', f'lif_layers.{lif_idx}'),
                cut_module=f'encoder_layers.{i}',
                state_argnums=(1,)
            )
        )

        lif_idx += 1

    for i in range(len(model.decoder_layers)):
        final_layer = i == len(model.decoder_layers) - 1

        sites.append(
            DECOLLESite(
                name=f'decoder_{i}',
                spike_module=f'lif_layers.{lif_idx}',
                trainable_modules=(f'decoder_layers.{i}', f'lif_layers.{lif_idx}'),
                cut_module=f'decoder_layers.{i}',
                state_argnums=(1,),
                direct_output=final_layer
            )
        )

        lif_idx += 1

    return sites