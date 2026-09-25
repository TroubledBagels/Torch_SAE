from dataclasses import dataclass

import matplotlib.pyplot as plt
import snntorch as snn
import torch
import torch.nn as nn
import tqdm


def _loss_scalar(loss):
    if loss.ndim == 0:
        return loss

    return loss.mean()


def _criterion_name(criterion):
    return getattr(criterion, "__name__", criterion.__class__.__name__).lower()


def reconstruction_metrics(outputs, targets):
    """Exact-timestep spike reconstruction counts."""
    outputs = outputs.reshape(outputs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    tp = ((outputs == 1) & (targets == 1)).sum().item()
    fp = ((outputs == 1) & (targets == 0)).sum().item()
    fn = ((outputs == 0) & (targets == 1)).sum().item()

    return tp, fp, fn


def reconstruction_metrics_tolerant(outputs, targets, tolerance=2):
    """
    Spike reconstruction counts with temporal tolerance.

    A predicted spike is a true positive when it can be matched one-to-one with
    a target spike on the same sample/channel within +/- ``tolerance`` timesteps.
    Each predicted spike and target spike can be used at most once.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    if outputs.shape != targets.shape:
        raise ValueError(
            f"outputs and targets must have the same shape, got "
            f"{tuple(outputs.shape)} and {tuple(targets.shape)}"
        )

    if outputs.ndim < 2:
        raise ValueError("outputs and targets must include a timestep dimension")

    # Keep the final dimension as time and collapse every leading dimension
    # (batch, channel, etc.) into independent spike trains.
    pred = (outputs == 1).reshape(-1, outputs.size(-1))
    truth = (targets == 1).reshape(-1, targets.size(-1))

    tp = 0
    fp = 0
    fn = 0

    for pred_train, target_train in zip(pred, truth):
        pred_times = torch.nonzero(pred_train, as_tuple=False).flatten().tolist()
        target_times = torch.nonzero(target_train, as_tuple=False).flatten().tolist()

        pred_index = 0
        target_index = 0
        matched = 0

        # Sorted two-pointer matching gives a one-to-one temporal match without
        # allowing several predictions to claim the same target event.
        while pred_index < len(pred_times) and target_index < len(target_times):
            pred_t = pred_times[pred_index]
            target_t = target_times[target_index]

            if abs(pred_t - target_t) <= tolerance:
                matched += 1
                pred_index += 1
                target_index += 1
            elif pred_t < target_t - tolerance:
                pred_index += 1
            else:
                target_index += 1

        tp += matched
        fp += len(pred_times) - matched
        fn += len(target_times) - matched

    return tp, fp, fn


def f1_from_counts(tp, fp, fn):
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

    return precision, recall, f1


def _diagnostic_loss_per_timestep(outputs, targets, criterion, tau=10.0):
    """
    Per-timestep diagnostic used only for plotting.

    For Van Rossum-like losses this plots filtered-trace MSE at each timestep.
    For arbitrary losses it falls back to exact MSE per timestep. The actual
    training loss is always the user-supplied criterion.
    """
    criterion_name = _criterion_name(criterion)

    if "rossum" in criterion_name:
        alpha = torch.exp(torch.tensor(-1.0 / tau, device=outputs.device, dtype=outputs.dtype))
        pred_trace = torch.zeros_like(outputs)
        target_trace = torch.zeros_like(targets)
        pred_trace[..., 0] = outputs[..., 0]
        target_trace[..., 0] = targets[..., 0]

        for t in range(1, outputs.shape[-1]):
            pred_trace[..., t] = alpha * pred_trace[..., t - 1] + outputs[..., t]
            target_trace[..., t] = alpha * target_trace[..., t - 1] + targets[..., t]

        return ((pred_trace - target_trace) ** 2).mean(dim=(0, 1))

    return ((outputs - targets) ** 2).mean(dim=(0, 1))


def plot_stage_reconstruction(targets, outputs, layer_spikes, epoch, stage_name, criterion, tau=10.0):
    target = targets[0].detach().cpu()
    output = outputs[0].detach().cpu()
    timestep_loss = _diagnostic_loss_per_timestep(outputs, targets, criterion, tau=tau).detach().cpu()

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].imshow(target, aspect="auto", interpolation="nearest", origin="lower")
    axes[0].set_title(f"Target - {stage_name} - Epoch {epoch + 1}")
    axes[0].set_ylabel("Channel")

    axes[1].imshow(output, aspect="auto", interpolation="nearest", origin="lower")
    axes[1].set_title(f"Reconstruction - {stage_name} - Epoch {epoch + 1}")
    axes[1].set_ylabel("Channel")

    axes[2].plot(timestep_loss)
    axes[2].set_title("Diagnostic Loss Per Timestep")
    axes[2].set_ylabel("Loss")
    axes[2].grid()

    loss_axis = axes[3].twinx()

    for layer_name, spikes in layer_spikes.items():
        spike_count = spikes[0].detach().cpu().sum(dim=0)
        axes[3].plot(spike_count, label=layer_name)

    loss_axis.plot(timestep_loss, linestyle="--", alpha=0.45, label="Loss")
    axes[3].set_title("Spikes Per Timestep for Trainable Layers")
    axes[3].set_ylabel("Spike count")
    axes[3].set_xlabel("Timestep")
    loss_axis.set_ylabel("Loss")
    axes[3].grid()

    handles_1, labels_1 = axes[3].get_legend_handles_labels()
    handles_2, labels_2 = loss_axis.get_legend_handles_labels()
    axes[3].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.show()


class TemporarySpikingReadout(nn.Module):
    def __init__(self, input_size, output_size, beta=0.9, threshold=1.0, surrogate_slope=25):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.spike_grad = snn.surrogate.fast_sigmoid(slope=surrogate_slope)
        self.lif = snn.Leaky(
            beta=torch.tensor(float(beta), dtype=torch.float32),
            threshold=torch.tensor(float(threshold), dtype=torch.float32),
            spike_grad=self.spike_grad
        )

    def init_state(self):
        return self.lif.init_leaky()

    def forward_step(self, x, state):
        return self.lif(self.linear(x), state)


@dataclass
class ProgressiveStage:
    name: str
    structure: list
    active_end: int
    train_indices: list
    target_mode: str
    use_temp_readout: bool
    temp_input_size: int | None = None
    temp_output_size: int | None = None


class ProgressiveSpikingAutoencoderTrainer:
    """
    Greedy layer-wise trainer for a symmetric feed-forward spiking autoencoder.

    Default model interface:
        model.encoder_layers : ModuleList of Linear-like synaptic layers
        model.decoder_layers : ModuleList of Linear-like synaptic layers
        model.lif_layers     : ModuleList of matching spiking neurons

    Example schedule for I -> M -> N -> C -> N -> M -> I:
        I -> M -> I
        I -> M -> N -> M
        I -> M -> N -> C -> N
        I -> M -> N -> C -> N -> M -> N
        I -> M -> N -> C -> N -> M -> I

    Earlier retained layers are frozen before the next stage. Temporary
    spiking reconstruction heads are discarded after their stage. A final
    all-layer fine-tuning phase is optional.

    loss_fn may be any callable accepting loss_fn(prediction, target). If it
    returns a non-scalar tensor, its mean is used.

    Optional early stopping is controlled by ``patience`` and ``patience_metric``.
    ``patience=None`` disables early stopping. When enabled, each progressive
    stage and the optional fine-tuning phase stop independently after the chosen
    number of consecutive epochs without improvement and restore their best
    checkpoint according to the selected metric.

    If ``reuse_initial_decoder_for_final_layer=True``, the retained temporary
    decoder learned by the first I -> M -> I stage is saved. Immediately before
    the final M -> I decoder layer is trained, its compatible linear and spiking
    neuron state is initialised from that saved decoder. The final layer is then
    trained normally; this option changes its initialisation, not whether it is
    trainable.
    """

    def __init__(
        self,
        model,
        loss_fn,
        device=None,
        lr=1e-3,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        encoder_layers=None,
        decoder_layers=None,
        neuron_layers=None,
        temp_beta=0.9,
        temp_threshold=1.0,
        surrogate_slope=25,
        detach_state_each_timestep=False,
        f1_tolerance=2,
        patience=None,
        patience_metric="loss",
        patience_min_delta=0.0,
        reuse_initial_decoder_for_final_layer=False
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.temp_beta = temp_beta
        self.temp_threshold = temp_threshold
        self.surrogate_slope = surrogate_slope
        self.detach_state_each_timestep = detach_state_each_timestep
        self.f1_tolerance = int(f1_tolerance)
        self.patience = None if patience is None else int(patience)
        self.patience_metric = self._normalise_patience_metric(patience_metric)
        self.patience_min_delta = float(patience_min_delta)
        self.reuse_initial_decoder_for_final_layer = bool(reuse_initial_decoder_for_final_layer)

        # Filled from the retained temporary readout of the first I -> M -> I
        # stage. It deliberately lives outside model.state_dict() because the
        # temporary readout is not part of the user's model.
        self._initial_decoder_state = None

        if self.f1_tolerance < 0:
            raise ValueError("f1_tolerance must be >= 0")

        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be None or an integer >= 1")

        if self.patience_min_delta < 0:
            raise ValueError("patience_min_delta must be >= 0")

        if device is None:
            device = next(model.parameters()).device

        self.device = torch.device(device)
        self.model.to(self.device)

        if encoder_layers is None:
            if not hasattr(model, "encoder_layers"):
                raise ValueError("Pass encoder_layers or give the model an encoder_layers attribute")

            encoder_layers = list(model.encoder_layers)

        if decoder_layers is None:
            if not hasattr(model, "decoder_layers"):
                raise ValueError("Pass decoder_layers or give the model a decoder_layers attribute")

            decoder_layers = list(model.decoder_layers)

        self.encoder_layers = list(encoder_layers)
        self.decoder_layers = list(decoder_layers)
        self.layers = self.encoder_layers + self.decoder_layers
        self.encoder_depth = len(self.encoder_layers)
        self.decoder_depth = len(self.decoder_layers)

        if neuron_layers is None:
            if not hasattr(model, "lif_layers"):
                raise ValueError("Pass neuron_layers or give the model a lif_layers attribute")

            neuron_layers = list(model.lif_layers)

        self.neurons = list(neuron_layers)

        self._validate_architecture()
        self.stages = self._build_stages()


    @staticmethod
    def _normalise_patience_metric(metric):
        """
        Normalise the validation metric used for early stopping.

        Supported metrics:
            loss / test_loss
            train_loss
            precision / test_precision
            recall / test_recall
            f1 / test_f1
            precision_tolerant / test_precision_tolerant
            recall_tolerant / test_recall_tolerant
            f1_tolerant / test_f1_tolerant

        Loss metrics are minimised. Precision/recall/F1 metrics are maximised.
        """
        aliases = {
            "loss": "loss",
            "test_loss": "loss",
            "train_loss": "train_loss",
            "precision": "precision",
            "test_precision": "precision",
            "recall": "recall",
            "test_recall": "recall",
            "f1": "f1",
            "test_f1": "f1",
            "precision_tolerant": "precision_tolerant",
            "test_precision_tolerant": "precision_tolerant",
            "recall_tolerant": "recall_tolerant",
            "test_recall_tolerant": "recall_tolerant",
            "f1_tolerant": "f1_tolerant",
            "test_f1_tolerant": "f1_tolerant",
        }

        key = str(metric).lower().strip()

        if key not in aliases:
            choices = ", ".join(sorted(aliases))
            raise ValueError(
                f"Unsupported patience_metric '{metric}'. Choose one of: {choices}"
            )

        return aliases[key]

    def _patience_mode(self):
        if self.patience_metric in ("loss", "train_loss"):
            return "min"

        return "max"

    def _patience_value(self, train_loss, test_metrics):
        if self.patience_metric == "train_loss":
            return float(train_loss)

        return float(test_metrics[self.patience_metric])

    def _patience_improved(self, value, best_value):
        if best_value is None:
            return True

        delta = self.patience_min_delta

        if self._patience_mode() == "min":
            return value < best_value - delta

        return value > best_value + delta

    def _patience_status_text(self, best_value, epochs_without_improvement):
        if self.patience is None:
            return "patience disabled"

        return (
            f"patience {epochs_without_improvement}/{self.patience} | "
            f"best {self.patience_metric}={best_value:.6f}"
        )

    def _validate_architecture(self):
        if self.encoder_depth == 0 or self.decoder_depth == 0:
            raise ValueError("The autoencoder must contain at least one encoder and one decoder layer")

        if self.encoder_depth != self.decoder_depth:
            raise ValueError(
                "This progressive schedule expects a symmetric autoencoder with the same "
                "number of encoder and decoder layers"
            )

        if len(self.neurons) != len(self.layers):
            raise ValueError(
                f"Expected one spiking neuron module per synaptic layer, "
                f"got {len(self.neurons)} neurons for {len(self.layers)} layers"
            )

        for i in range(len(self.layers) - 1):
            left = self.layers[i]
            right = self.layers[i + 1]

            if not hasattr(left, "out_features") or not hasattr(right, "in_features"):
                raise TypeError("The default trainer expects Linear-like layers with in_features/out_features")

            if left.out_features != right.in_features:
                raise ValueError(
                    f"Layer {i} outputs {left.out_features}, but layer {i + 1} expects {right.in_features}"
                )

        if self.layers[-1].out_features != self.layers[0].in_features:
            raise ValueError(
                f"Final output size {self.layers[-1].out_features} does not match "
                f"input size {self.layers[0].in_features}"
            )

    def _network_widths(self):
        widths = [self.layers[0].in_features]

        for layer in self.layers:
            widths.append(layer.out_features)

        return widths

    def _build_stages(self):
        widths = self._network_widths()
        e = self.encoder_depth
        d = self.decoder_depth
        stages = []

        for encoder_index in range(e - 1):
            previous_size = self.layers[encoder_index].in_features
            new_size = self.layers[encoder_index].out_features
            structure = widths[:encoder_index + 2] + [previous_size]

            stages.append(
                ProgressiveStage(
                    name=" -> ".join(str(v) for v in structure),
                    structure=structure,
                    active_end=encoder_index,
                    train_indices=[encoder_index],
                    target_mode="local_input",
                    use_temp_readout=True,
                    temp_input_size=new_size,
                    temp_output_size=previous_size
                )
            )

        center_encoder = e - 1
        center_decoder = e

        if d == 1:
            center_target_mode = "global_input"
        else:
            center_target_mode = "local_input"

        center_structure = widths[:center_decoder + 2]

        stages.append(
            ProgressiveStage(
                name=" -> ".join(str(v) for v in center_structure),
                structure=center_structure,
                active_end=center_decoder,
                train_indices=[center_encoder, center_decoder],
                target_mode=center_target_mode,
                use_temp_readout=False
            )
        )

        for decoder_index in range(1, d - 1):
            actual_index = e + decoder_index
            previous_size = self.layers[actual_index].in_features
            new_size = self.layers[actual_index].out_features
            structure = widths[:actual_index + 2] + [previous_size]

            stages.append(
                ProgressiveStage(
                    name=" -> ".join(str(v) for v in structure),
                    structure=structure,
                    active_end=actual_index,
                    train_indices=[actual_index],
                    target_mode="local_input",
                    use_temp_readout=True,
                    temp_input_size=new_size,
                    temp_output_size=previous_size
                )
            )

        if d > 1:
            final_index = len(self.layers) - 1
            final_structure = widths

            stages.append(
                ProgressiveStage(
                    name=" -> ".join(str(v) for v in final_structure),
                    structure=final_structure,
                    active_end=final_index,
                    train_indices=[final_index],
                    target_mode="global_input",
                    use_temp_readout=False
                )
            )

        return stages

    def describe_schedule(self):
        for i, stage in enumerate(self.stages):
            train_text = ", ".join(str(index) for index in stage.train_indices)
            print(f"Stage {i + 1}: {stage.name} | train actual layer(s): {train_text}")

    def _freeze_all_model_parameters(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _unfreeze_all_model_parameters(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def _prepare_stage_parameters(self, stage, temp_readout):
        self._freeze_all_model_parameters()
        params = []

        for index in stage.train_indices:
            for parameter in self.layers[index].parameters():
                parameter.requires_grad = True
                params.append(parameter)

            for parameter in self.neurons[index].parameters():
                parameter.requires_grad = True
                params.append(parameter)

        if temp_readout is not None:
            for parameter in temp_readout.parameters():
                parameter.requires_grad = True
                params.append(parameter)

        return params

    def _make_optimizer(self, params):
        kwargs = dict(self.optimizer_kwargs)
        kwargs.setdefault("lr", self.lr)

        return self.optimizer_class(params, **kwargs)

    def _make_temp_readout(self, stage):
        if not stage.use_temp_readout:
            return None

        readout = TemporarySpikingReadout(
            input_size=stage.temp_input_size,
            output_size=stage.temp_output_size,
            beta=self.temp_beta,
            threshold=self.temp_threshold,
            surrogate_slope=self.surrogate_slope
        )

        return readout.to(self.device)

    @staticmethod
    def _clone_state_dict(module):
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    @staticmethod
    def _load_compatible_state(target_module, source_state):
        """Load only state entries whose names and tensor shapes match."""
        target_state = target_module.state_dict()
        compatible = {}

        for key, value in source_state.items():
            if key in target_state and target_state[key].shape == value.shape:
                compatible[key] = value.to(
                    device=target_state[key].device,
                    dtype=target_state[key].dtype
                )

        if compatible:
            target_module.load_state_dict(compatible, strict=False)

        return sorted(compatible)

    def _is_initial_i_m_i_stage(self, stage, stage_number):
        return (
            stage_number == 1
            and stage.use_temp_readout
            and stage.train_indices == [0]
            and stage.active_end == 0
        )

    def _is_final_decoder_stage(self, stage):
        final_index = len(self.layers) - 1
        return (
            not stage.use_temp_readout
            and stage.active_end == final_index
            and stage.train_indices == [final_index]
        )

    def _save_initial_decoder(self, temp_readout, retained_epoch):
        if temp_readout is None:
            raise RuntimeError(
                "Cannot save the initial I -> M -> I decoder because that stage "
                "does not have a temporary readout."
            )

        self._initial_decoder_state = {
            "linear": self._clone_state_dict(temp_readout.linear),
            "lif": self._clone_state_dict(temp_readout.lif),
            "retained_epoch": retained_epoch,
            "input_size": temp_readout.linear.in_features,
            "output_size": temp_readout.linear.out_features,
        }

        tqdm.tqdm.write(
            "Saved retained decoder from initial I -> M -> I stage "
            f"(epoch {retained_epoch}) for final-layer initialisation."
        )

    def _initialise_final_decoder_from_initial(self):
        if not self.reuse_initial_decoder_for_final_layer:
            return False

        if self._initial_decoder_state is None:
            raise RuntimeError(
                "reuse_initial_decoder_for_final_layer=True, but no decoder from "
                "the initial I -> M -> I stage has been saved. This option "
                "requires a progressive schedule whose first stage uses a "
                "temporary M -> I readout."
            )

        final_layer = self.layers[-1]
        final_neuron = self.neurons[-1]
        source_in = self._initial_decoder_state["input_size"]
        source_out = self._initial_decoder_state["output_size"]

        if (
            getattr(final_layer, "in_features", None) != source_in
            or getattr(final_layer, "out_features", None) != source_out
        ):
            raise ValueError(
                "The initial I -> M -> I decoder cannot initialise the final "
                f"layer: saved decoder is {source_in}->{source_out}, but the "
                f"final layer is {getattr(final_layer, 'in_features', '?')}->"
                f"{getattr(final_layer, 'out_features', '?')}."
            )

        loaded_linear = self._load_compatible_state(
            final_layer,
            self._initial_decoder_state["linear"]
        )

        if "weight" not in loaded_linear:
            raise ValueError(
                "Could not copy the saved initial decoder weight into the final "
                "decoder layer. Their state_dict layouts are incompatible."
            )

        loaded_neuron = self._load_compatible_state(
            final_neuron,
            self._initial_decoder_state["lif"]
        )

        neuron_text = (
            f"; copied neuron state: {', '.join(loaded_neuron)}"
            if loaded_neuron
            else "; no compatible neuron state to copy"
        )
        tqdm.tqdm.write(
            "Initialised final decoder from retained initial I -> M -> I decoder "
            f"(source epoch {self._initial_decoder_state['retained_epoch']}; "
            f"copied linear state: {', '.join(loaded_linear)}{neuron_text})."
        )

        return True

    def _init_neuron_state(self, neuron):
        if hasattr(neuron, "init_leaky"):
            return neuron.init_leaky()

        raise TypeError(
            f"Neuron {type(neuron).__name__} does not expose init_leaky(). "
            "Pass snnTorch Leaky-compatible neuron modules or extend _init_neuron_state()."
        )

    def _step_neuron(self, neuron, current, state):
        result = neuron(current, state)

        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise TypeError(f"Neuron {type(neuron).__name__} must return at least (spikes, state)")

        spikes = result[0]
        new_state = result[1]

        if self.detach_state_each_timestep:
            new_state = new_state.detach()

        return spikes, new_state

    def _stage_forward(self, inputs, stage, temp_readout=None, record_spikes=False):
        timesteps = inputs.size(2)
        train_start = min(stage.train_indices)
        actual_states = {
            index: self._init_neuron_state(self.neurons[index])
            for index in range(stage.active_end + 1)
        }

        if temp_readout is not None:
            temp_state = temp_readout.init_state()
        else:
            temp_state = None

        outputs = []
        targets = []
        recorded = {}

        if record_spikes:
            for index in stage.train_indices:
                recorded[f"Layer {index} ({self.layers[index].in_features}->{self.layers[index].out_features})"] = []

            if temp_readout is not None:
                recorded["Temporary readout"] = []

        for t in range(timesteps):
            current = inputs[:, :, t]

            if stage.target_mode == "global_input":
                target_t = current
            else:
                target_t = None

            for index in range(stage.active_end + 1):
                if index == train_start and stage.target_mode == "local_input":
                    target_t = current.detach()

                if index < train_start:
                    with torch.no_grad():
                        synaptic = self.layers[index](current)
                        current, actual_states[index] = self._step_neuron(
                            self.neurons[index],
                            synaptic,
                            actual_states[index]
                        )

                    current = current.detach()
                else:
                    synaptic = self.layers[index](current)
                    current, actual_states[index] = self._step_neuron(
                        self.neurons[index],
                        synaptic,
                        actual_states[index]
                    )

                if record_spikes and index in stage.train_indices:
                    key = f"Layer {index} ({self.layers[index].in_features}->{self.layers[index].out_features})"
                    recorded[key].append(current.detach())

            if temp_readout is not None:
                current, temp_state = temp_readout.forward_step(current, temp_state)

                if self.detach_state_each_timestep:
                    temp_state = temp_state.detach()

                if record_spikes:
                    recorded["Temporary readout"].append(current.detach())

            if target_t is None:
                raise RuntimeError("Stage target was not created")

            outputs.append(current)
            targets.append(target_t)

        outputs = torch.stack(outputs, dim=2)
        targets = torch.stack(targets, dim=2)

        if record_spikes:
            recorded = {
                name: torch.stack(values, dim=2)
                for name, values in recorded.items()
            }

        return outputs, targets, recorded

    def _full_forward(self, inputs, record_spikes=False):
        timesteps = inputs.size(2)
        states = [self._init_neuron_state(neuron) for neuron in self.neurons]
        outputs = []
        recorded = {}

        if record_spikes:
            for index, layer in enumerate(self.layers):
                recorded[f"Layer {index} ({layer.in_features}->{layer.out_features})"] = []

        for t in range(timesteps):
            current = inputs[:, :, t]

            for index, (layer, neuron) in enumerate(zip(self.layers, self.neurons)):
                current, states[index] = self._step_neuron(neuron, layer(current), states[index])

                if record_spikes:
                    key = f"Layer {index} ({layer.in_features}->{layer.out_features})"
                    recorded[key].append(current.detach())

            outputs.append(current)

        outputs = torch.stack(outputs, dim=2)

        if record_spikes:
            recorded = {
                name: torch.stack(values, dim=2)
                for name, values in recorded.items()
            }

        return outputs, recorded

    def _evaluate_stage(self, loader, stage, temp_readout, epoch, plot, tau):
        self.model.eval()

        if temp_readout is not None:
            temp_readout.eval()

        running_loss = 0.0
        sample_count = 0
        tp = 0
        fp = 0
        fn = 0
        tolerant_tp = 0
        tolerant_fp = 0
        tolerant_fn = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)
                should_record = plot and batch_index == 0

                outputs, targets, layer_spikes = self._stage_forward(
                    inputs,
                    stage,
                    temp_readout=temp_readout,
                    record_spikes=should_record
                )

                loss = _loss_scalar(self.loss_fn(outputs, targets))
                running_loss += loss.item() * inputs.size(0)
                sample_count += inputs.size(0)

                batch_tp, batch_fp, batch_fn = reconstruction_metrics(outputs, targets)
                tp += batch_tp
                fp += batch_fp
                fn += batch_fn

                batch_tp, batch_fp, batch_fn = reconstruction_metrics_tolerant(
                    outputs,
                    targets,
                    tolerance=self.f1_tolerance
                )
                tolerant_tp += batch_tp
                tolerant_fp += batch_fp
                tolerant_fn += batch_fn

                if should_record:
                    plot_stage_reconstruction(
                        targets,
                        outputs,
                        layer_spikes,
                        epoch,
                        stage.name,
                        self.loss_fn,
                        tau=tau
                    )

        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        precision_tolerant, recall_tolerant, f1_tolerant = f1_from_counts(
            tolerant_tp,
            tolerant_fp,
            tolerant_fn
        )

        return {
            "loss": running_loss / max(sample_count, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_tolerant": precision_tolerant,
            "recall_tolerant": recall_tolerant,
            "f1_tolerant": f1_tolerant
        }

    def _print_stage_recap(self, stage_results):
        epochs = len(stage_results["train_loss"])
        first_train_loss = stage_results["train_loss"][0]
        final_train_loss = stage_results["train_loss"][-1]
        first_test_loss = stage_results["test_loss"][0]
        final_test_loss = stage_results["test_loss"][-1]
        first_f1 = stage_results["test_f1"][0]
        final_f1 = stage_results["test_f1"][-1]
        f1_change = final_f1 - first_f1
        first_f1_tolerant = stage_results["test_f1_tolerant"][0]
        final_f1_tolerant = stage_results["test_f1_tolerant"][-1]
        f1_tolerant_change = final_f1_tolerant - first_f1_tolerant

        print("")
        print("=" * 88)
        print(f"STAGE {stage_results['stage']} COMPLETE: {stage_results['name']}")
        print(f"Epochs trained: {epochs}")
        print(f"Trainable actual layer(s): {stage_results['train_layers']}")
        if stage_results.get("initialised_from_initial_decoder", False):
            print("Final decoder initialisation: retained decoder from initial I -> M -> I stage")
        print(f"Train loss: {first_train_loss:.6f} -> {final_train_loss:.6f}")
        print(f"Test loss:  {first_test_loss:.6f} -> {final_test_loss:.6f}")
        print(f"Test F1:    {first_f1:.4f} -> {final_f1:.4f} ({f1_change:+.4f})")
        print(
            f"Test F1 (±{self.f1_tolerance}t): "
            f"{first_f1_tolerant:.4f} -> {final_f1_tolerant:.4f} "
            f"({f1_tolerant_change:+.4f})"
        )
        print(
            f"Best F1: {stage_results['best_f1']:.4f} "
            f"at epoch {stage_results['best_f1_epoch']}"
        )
        print(
            f"Best F1 (±{self.f1_tolerance}t): "
            f"{stage_results['best_f1_tolerant']:.4f} "
            f"at epoch {stage_results['best_f1_tolerant_epoch']}"
        )
        print(
            f"Best test loss: {stage_results['best_test_loss']:.6f} "
            f"at epoch {stage_results['best_test_loss_epoch']}"
        )
        print(
            f"Retained checkpoint: {stage_results['retained_metric']}="
            f"{stage_results['retained_metric_value']:.6f} "
            f"at epoch {stage_results['retained_metric_epoch']}"
        )
        print(
            f"Retained checkpoint metrics: "
            f"P={stage_results['retained_precision']:.4f} | "
            f"R={stage_results['retained_recall']:.4f} | "
            f"F1={stage_results['retained_f1']:.4f} | "
            f"F1(±{self.f1_tolerance}t)={stage_results['retained_f1_tolerant']:.4f}"
        )

        if stage_results["patience"] is not None:
            stop_text = "triggered" if stage_results["stopped_early"] else "not triggered"
            print(
                f"Early stopping: {stop_text} | "
                f"patience={stage_results['patience']} | "
                f"metric={stage_results['patience_metric']} | "
                f"min_delta={stage_results['patience_min_delta']:g}"
            )

        print("=" * 88)
        print("")
    def _print_fine_tune_recap(self, results):
        epochs = len(results["train_loss"])
        first_train_loss = results["train_loss"][0]
        final_train_loss = results["train_loss"][-1]
        first_test_loss = results["test_loss"][0]
        final_test_loss = results["test_loss"][-1]
        first_f1 = results["test_f1"][0]
        final_f1 = results["test_f1"][-1]
        f1_change = final_f1 - first_f1
        first_f1_tolerant = results["test_f1_tolerant"][0]
        final_f1_tolerant = results["test_f1_tolerant"][-1]
        f1_tolerant_change = final_f1_tolerant - first_f1_tolerant

        print("")
        print("=" * 88)
        print("FINAL FINE TUNING COMPLETE")
        print(f"Epochs trained: {epochs}")
        print(f"Train loss: {first_train_loss:.6f} -> {final_train_loss:.6f}")
        print(f"Test loss:  {first_test_loss:.6f} -> {final_test_loss:.6f}")
        print(f"Test F1:    {first_f1:.4f} -> {final_f1:.4f} ({f1_change:+.4f})")
        print(
            f"Test F1 (±{self.f1_tolerance}t): "
            f"{first_f1_tolerant:.4f} -> {final_f1_tolerant:.4f} "
            f"({f1_tolerant_change:+.4f})"
        )
        print(f"Best F1: {results['best_f1']:.4f} at epoch {results['best_f1_epoch']}")
        print(
            f"Best F1 (±{self.f1_tolerance}t): "
            f"{results['best_f1_tolerant']:.4f} "
            f"at epoch {results['best_f1_tolerant_epoch']}"
        )
        print(
            f"Best test loss: {results['best_test_loss']:.6f} "
            f"at epoch {results['best_test_loss_epoch']}"
        )
        print(
            f"Retained checkpoint: {results['retained_metric']}="
            f"{results['retained_metric_value']:.6f} "
            f"at epoch {results['retained_metric_epoch']}"
        )
        print(
            f"Retained checkpoint metrics: "
            f"P={results['retained_precision']:.4f} | "
            f"R={results['retained_recall']:.4f} | "
            f"F1={results['retained_f1']:.4f} | "
            f"F1(±{self.f1_tolerance}t)={results['retained_f1_tolerant']:.4f}"
        )

        if results["patience"] is not None:
            stop_text = "triggered" if results["stopped_early"] else "not triggered"
            print(
                f"Early stopping: {stop_text} | "
                f"patience={results['patience']} | "
                f"metric={results['patience_metric']} | "
                f"min_delta={results['patience_min_delta']:g}"
            )

        print("=" * 88)
        print("")
    def _print_training_recap(self, results):
        print("")
        print("#" * 88)
        print("SYMMETRIC TRAINING RECAP")

        for stage_results in results["stages"]:
            print(
                f"Stage {stage_results['stage']}: {stage_results['name']} | "
                f"Best F1={stage_results['best_f1']:.4f} "
                f"(epoch {stage_results['best_f1_epoch']}) | "
                f"Retained F1={stage_results['retained_f1']:.4f} | "
                f"Best F1(±{self.f1_tolerance}t)={stage_results['best_f1_tolerant']:.4f} "
                f"(epoch {stage_results['best_f1_tolerant_epoch']}) | "
                f"Retained F1(±{self.f1_tolerance}t)="
                f"{stage_results['retained_f1_tolerant']:.4f} | "
                f"Best loss={stage_results['best_test_loss']:.6f}"
            )

        if results["fine_tune"] is not None:
            fine_tune = results["fine_tune"]
            print(
                f"Fine tune: Best F1={fine_tune['best_f1']:.4f} "
                f"(epoch {fine_tune['best_f1_epoch']}) | "
                f"Retained F1={fine_tune['retained_f1']:.4f} | "
                f"Best F1(±{self.f1_tolerance}t)={fine_tune['best_f1_tolerant']:.4f} "
                f"(epoch {fine_tune['best_f1_tolerant_epoch']}) | "
                f"Retained F1(±{self.f1_tolerance}t)="
                f"{fine_tune['retained_f1_tolerant']:.4f} | "
                f"Best loss={fine_tune['best_test_loss']:.6f}"
            )

        print("#" * 88)
        print("")

    def _train_stage(self, train_loader, test_loader, stage, num_epochs, stage_number, plot, plot_every, tau):
        temp_readout = self._make_temp_readout(stage)

        initialised_from_initial_decoder = False
        if self._is_final_decoder_stage(stage):
            initialised_from_initial_decoder = self._initialise_final_decoder_from_initial()

        params = self._prepare_stage_parameters(stage, temp_readout)
        optimizer = self._make_optimizer(params)

        best_test_loss = float("inf")
        best_test_loss_epoch = None
        best_f1 = 0.0
        best_f1_epoch = None
        best_f1_tolerant = 0.0
        best_f1_tolerant_epoch = None

        # When patience is enabled, the retained checkpoint follows the selected
        # patience metric. With patience disabled, retain the historical behaviour
        # of restoring the minimum validation-loss checkpoint.
        retained_metric = self.patience_metric if self.patience is not None else "loss"
        retained_delta = self.patience_min_delta if self.patience is not None else 0.0
        retained_metric_value = None
        retained_metric_epoch = None
        retained_precision = 0.0
        retained_recall = 0.0
        retained_f1 = 0.0
        retained_precision_tolerant = 0.0
        retained_recall_tolerant = 0.0
        retained_f1_tolerant = 0.0
        best_model_state = None
        best_temp_readout_state = None

        epochs_without_improvement = 0
        stopped_early = False

        stage_results = {
            "stage": stage_number,
            "structure": stage.structure,
            "name": stage.name,
            "train_layers": list(stage.train_indices),
            "train_loss": [],
            "test_loss": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "test_precision_tolerant": [],
            "test_recall_tolerant": [],
            "test_f1_tolerant": []
        }

        print(f"\nStage {stage_number}/{len(self.stages)}: {stage.name}")
        print(f"Training actual layer(s): {stage.train_indices}")

        if self.patience is not None:
            print(
                f"Early stopping: patience={self.patience} | "
                f"metric={self.patience_metric} | "
                f"min_delta={self.patience_min_delta:g}"
            )

        for epoch in range(num_epochs):
            self.model.train()

            if temp_readout is not None:
                temp_readout.train()

            running_loss = 0.0
            batch_count = 0

            pbar = tqdm.tqdm(
                train_loader,
                desc=f"Stage {stage_number} Epoch {epoch + 1}/{num_epochs}",
                dynamic_ncols=True
            )

            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                outputs, targets, _ = self._stage_forward(
                    inputs,
                    stage,
                    temp_readout=temp_readout,
                    record_spikes=False
                )

                loss = _loss_scalar(self.loss_fn(outputs, targets))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                pbar.set_postfix(loss=f"{running_loss / batch_count:.6f}")

            train_loss = running_loss / max(batch_count, 1)
            should_plot = plot and ((epoch + 1) % plot_every == 0 or epoch == 0)

            test_metrics = self._evaluate_stage(
                test_loader,
                stage,
                temp_readout,
                epoch,
                should_plot,
                tau
            )

            stage_results["train_loss"].append(train_loss)
            stage_results["test_loss"].append(test_metrics["loss"])
            stage_results["test_precision"].append(test_metrics["precision"])
            stage_results["test_recall"].append(test_metrics["recall"])
            stage_results["test_f1"].append(test_metrics["f1"])
            stage_results["test_precision_tolerant"].append(test_metrics["precision_tolerant"])
            stage_results["test_recall_tolerant"].append(test_metrics["recall_tolerant"])
            stage_results["test_f1_tolerant"].append(test_metrics["f1_tolerant"])

            if test_metrics["loss"] < best_test_loss:
                best_test_loss = test_metrics["loss"]
                best_test_loss_epoch = epoch + 1

            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                best_f1_epoch = epoch + 1

            if test_metrics["f1_tolerant"] > best_f1_tolerant:
                best_f1_tolerant = test_metrics["f1_tolerant"]
                best_f1_tolerant_epoch = epoch + 1

            if retained_metric == "train_loss":
                current_retained_value = float(train_loss)
                retained_mode = "min"
            else:
                current_retained_value = float(test_metrics[retained_metric])
                retained_mode = "min" if retained_metric == "loss" else "max"

            if retained_metric_value is None:
                retained_improved = True
            elif retained_mode == "min":
                retained_improved = (
                    current_retained_value < retained_metric_value - retained_delta
                )
            else:
                retained_improved = (
                    current_retained_value > retained_metric_value + retained_delta
                )

            if retained_improved:
                retained_metric_value = current_retained_value
                retained_metric_epoch = epoch + 1
                retained_precision = test_metrics["precision"]
                retained_recall = test_metrics["recall"]
                retained_f1 = test_metrics["f1"]
                retained_precision_tolerant = test_metrics["precision_tolerant"]
                retained_recall_tolerant = test_metrics["recall_tolerant"]
                retained_f1_tolerant = test_metrics["f1_tolerant"]
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                if temp_readout is not None:
                    best_temp_readout_state = self._clone_state_dict(temp_readout)
                epochs_without_improvement = 0
            elif self.patience is not None:
                epochs_without_improvement += 1

            progress_text = (
                f"Stage {stage_number} Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_metrics['loss']:.6f} | "
                f"Test F1: {test_metrics['f1']:.4f} | "
                f"Test F1 (±{self.f1_tolerance}t): {test_metrics['f1_tolerant']:.4f} | "
                f"Best F1: {best_f1:.4f} | "
                f"Best F1 (±{self.f1_tolerance}t): {best_f1_tolerant:.4f}"
            )

            if self.patience is not None:
                progress_text += (
                    f" | {self.patience_metric}: {current_retained_value:.6f} | "
                    f"patience {epochs_without_improvement}/{self.patience}"
                )

            tqdm.tqdm.write(progress_text)

            if self.patience is not None and epochs_without_improvement >= self.patience:
                stopped_early = True
                tqdm.tqdm.write(
                    f"Early stopping stage {stage_number} at epoch {epoch + 1}: "
                    f"{self.patience_metric} did not improve for "
                    f"{self.patience} consecutive epoch(s). "
                    f"Restoring epoch {retained_metric_epoch}."
                )
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if temp_readout is not None and best_temp_readout_state is not None:
            temp_readout.load_state_dict(best_temp_readout_state)

        if (
            self.reuse_initial_decoder_for_final_layer
            and self._is_initial_i_m_i_stage(stage, stage_number)
        ):
            self._save_initial_decoder(temp_readout, retained_metric_epoch)

        stage_results["best_test_loss"] = best_test_loss
        stage_results["best_test_loss_epoch"] = best_test_loss_epoch
        stage_results["retained_precision"] = retained_precision
        stage_results["retained_recall"] = retained_recall
        stage_results["retained_f1"] = retained_f1
        stage_results["retained_precision_tolerant"] = retained_precision_tolerant
        stage_results["retained_recall_tolerant"] = retained_recall_tolerant
        stage_results["retained_f1_tolerant"] = retained_f1_tolerant
        stage_results["best_f1"] = best_f1
        stage_results["best_f1_epoch"] = best_f1_epoch
        stage_results["best_f1_tolerant"] = best_f1_tolerant
        stage_results["best_f1_tolerant_epoch"] = best_f1_tolerant_epoch
        stage_results["patience"] = self.patience
        stage_results["patience_metric"] = self.patience_metric
        stage_results["patience_min_delta"] = self.patience_min_delta
        stage_results["retained_metric"] = retained_metric
        stage_results["retained_metric_value"] = retained_metric_value
        stage_results["retained_metric_epoch"] = retained_metric_epoch
        stage_results["epochs_without_improvement"] = epochs_without_improvement
        stage_results["stopped_early"] = stopped_early
        stage_results["epochs_trained"] = len(stage_results["train_loss"])
        stage_results["reuse_initial_decoder_for_final_layer"] = self.reuse_initial_decoder_for_final_layer
        stage_results["initialised_from_initial_decoder"] = initialised_from_initial_decoder

        self._print_stage_recap(stage_results)

        return stage_results
    def _evaluate_fine_tune(self, loader, epoch, plot, tau):
        self.model.eval()

        running_loss = 0.0
        sample_count = 0
        tp = 0
        fp = 0
        fn = 0
        tolerant_tp = 0
        tolerant_fp = 0
        tolerant_fn = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)
                should_record = plot and batch_index == 0

                outputs, layer_spikes = self._full_forward(inputs, record_spikes=should_record)
                loss = _loss_scalar(self.loss_fn(outputs, inputs))

                running_loss += loss.item() * inputs.size(0)
                sample_count += inputs.size(0)

                batch_tp, batch_fp, batch_fn = reconstruction_metrics(outputs, inputs)
                tp += batch_tp
                fp += batch_fp
                fn += batch_fn

                batch_tp, batch_fp, batch_fn = reconstruction_metrics_tolerant(
                    outputs,
                    inputs,
                    tolerance=self.f1_tolerance
                )
                tolerant_tp += batch_tp
                tolerant_fp += batch_fp
                tolerant_fn += batch_fn

                if should_record:
                    plot_stage_reconstruction(
                        inputs,
                        outputs,
                        layer_spikes,
                        epoch,
                        "Full fine tuning",
                        self.loss_fn,
                        tau=tau
                    )

        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        precision_tolerant, recall_tolerant, f1_tolerant = f1_from_counts(
            tolerant_tp,
            tolerant_fp,
            tolerant_fn
        )

        return {
            "loss": running_loss / max(sample_count, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_tolerant": precision_tolerant,
            "recall_tolerant": recall_tolerant,
            "f1_tolerant": f1_tolerant
        }

    def _fine_tune(self, train_loader, test_loader, num_epochs, plot, plot_every, tau):
        self._unfreeze_all_model_parameters()
        optimizer = self._make_optimizer(self.model.parameters())

        best_test_loss = float("inf")
        best_test_loss_epoch = None
        best_f1 = 0.0
        best_f1_epoch = None
        best_f1_tolerant = 0.0
        best_f1_tolerant_epoch = None

        retained_metric = self.patience_metric if self.patience is not None else "loss"
        retained_delta = self.patience_min_delta if self.patience is not None else 0.0
        retained_metric_value = None
        retained_metric_epoch = None
        retained_precision = 0.0
        retained_recall = 0.0
        retained_f1 = 0.0
        retained_precision_tolerant = 0.0
        retained_recall_tolerant = 0.0
        retained_f1_tolerant = 0.0
        best_model_state = None

        epochs_without_improvement = 0
        stopped_early = False

        results = {
            "train_loss": [],
            "test_loss": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "test_precision_tolerant": [],
            "test_recall_tolerant": [],
            "test_f1_tolerant": []
        }

        print("\nFinal fine tuning: all layers unfrozen")

        if self.patience is not None:
            print(
                f"Early stopping: patience={self.patience} | "
                f"metric={self.patience_metric} | "
                f"min_delta={self.patience_min_delta:g}"
            )

        for epoch in range(num_epochs):
            self.model.train()

            running_loss = 0.0
            batch_count = 0

            pbar = tqdm.tqdm(
                train_loader,
                desc=f"Fine tune {epoch + 1}/{num_epochs}",
                dynamic_ncols=True
            )

            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                outputs, _ = self._full_forward(inputs, record_spikes=False)
                loss = _loss_scalar(self.loss_fn(outputs, inputs))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                pbar.set_postfix(loss=f"{running_loss / batch_count:.6f}")

            train_loss = running_loss / max(batch_count, 1)
            should_plot = plot and ((epoch + 1) % plot_every == 0 or epoch == 0)

            test_metrics = self._evaluate_fine_tune(
                test_loader,
                epoch,
                should_plot,
                tau
            )

            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_metrics["loss"])
            results["test_precision"].append(test_metrics["precision"])
            results["test_recall"].append(test_metrics["recall"])
            results["test_f1"].append(test_metrics["f1"])
            results["test_precision_tolerant"].append(test_metrics["precision_tolerant"])
            results["test_recall_tolerant"].append(test_metrics["recall_tolerant"])
            results["test_f1_tolerant"].append(test_metrics["f1_tolerant"])

            if test_metrics["loss"] < best_test_loss:
                best_test_loss = test_metrics["loss"]
                best_test_loss_epoch = epoch + 1

            if test_metrics["f1"] > best_f1:
                best_f1 = test_metrics["f1"]
                best_f1_epoch = epoch + 1

            if test_metrics["f1_tolerant"] > best_f1_tolerant:
                best_f1_tolerant = test_metrics["f1_tolerant"]
                best_f1_tolerant_epoch = epoch + 1

            if retained_metric == "train_loss":
                current_retained_value = float(train_loss)
                retained_mode = "min"
            else:
                current_retained_value = float(test_metrics[retained_metric])
                retained_mode = "min" if retained_metric == "loss" else "max"

            if retained_metric_value is None:
                retained_improved = True
            elif retained_mode == "min":
                retained_improved = (
                    current_retained_value < retained_metric_value - retained_delta
                )
            else:
                retained_improved = (
                    current_retained_value > retained_metric_value + retained_delta
                )

            if retained_improved:
                retained_metric_value = current_retained_value
                retained_metric_epoch = epoch + 1
                retained_precision = test_metrics["precision"]
                retained_recall = test_metrics["recall"]
                retained_f1 = test_metrics["f1"]
                retained_precision_tolerant = test_metrics["precision_tolerant"]
                retained_recall_tolerant = test_metrics["recall_tolerant"]
                retained_f1_tolerant = test_metrics["f1_tolerant"]
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            elif self.patience is not None:
                epochs_without_improvement += 1

            progress_text = (
                f"Fine Tune Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_metrics['loss']:.6f} | "
                f"Test F1: {test_metrics['f1']:.4f} | "
                f"Test F1 (±{self.f1_tolerance}t): {test_metrics['f1_tolerant']:.4f} | "
                f"Best F1: {best_f1:.4f} | "
                f"Best F1 (±{self.f1_tolerance}t): {best_f1_tolerant:.4f}"
            )

            if self.patience is not None:
                progress_text += (
                    f" | {self.patience_metric}: {current_retained_value:.6f} | "
                    f"patience {epochs_without_improvement}/{self.patience}"
                )

            tqdm.tqdm.write(progress_text)

            if self.patience is not None and epochs_without_improvement >= self.patience:
                stopped_early = True
                tqdm.tqdm.write(
                    f"Early stopping fine tuning at epoch {epoch + 1}: "
                    f"{self.patience_metric} did not improve for "
                    f"{self.patience} consecutive epoch(s). "
                    f"Restoring epoch {retained_metric_epoch}."
                )
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        results["best_test_loss"] = best_test_loss
        results["best_test_loss_epoch"] = best_test_loss_epoch
        results["retained_precision"] = retained_precision
        results["retained_recall"] = retained_recall
        results["retained_f1"] = retained_f1
        results["retained_precision_tolerant"] = retained_precision_tolerant
        results["retained_recall_tolerant"] = retained_recall_tolerant
        results["retained_f1_tolerant"] = retained_f1_tolerant
        results["best_f1"] = best_f1
        results["best_f1_epoch"] = best_f1_epoch
        results["best_f1_tolerant"] = best_f1_tolerant
        results["best_f1_tolerant_epoch"] = best_f1_tolerant_epoch
        results["patience"] = self.patience
        results["patience_metric"] = self.patience_metric
        results["patience_min_delta"] = self.patience_min_delta
        results["retained_metric"] = retained_metric
        results["retained_metric_value"] = retained_metric_value
        results["retained_metric_epoch"] = retained_metric_epoch
        results["epochs_without_improvement"] = epochs_without_improvement
        results["stopped_early"] = stopped_early
        results["epochs_trained"] = len(results["train_loss"])

        self._print_fine_tune_recap(results)

        return results
    def test(self, test_loader, plot=False, tau=10.0, print_results=True):
        """
        Run a test-only evaluation of the current model weights.

        This does not create an optimizer, call backward(), or update any model
        parameters. Both the exact-timestep F1 and the temporally tolerant F1
        are returned.

        Example:
            metrics = trainer.test(test_loader, plot=True)
        """
        if test_loader is None:
            raise ValueError("test_loader must be provided")

        was_training = self.model.training

        metrics = self._evaluate_fine_tune(
            test_loader,
            epoch=0,
            plot=plot,
            tau=tau
        )

        # Restore the mode the caller had selected before evaluation.
        self.model.train(was_training)

        results = {
            "loss": metrics["loss"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "precision_tolerant": metrics["precision_tolerant"],
            "recall_tolerant": metrics["recall_tolerant"],
            "f1_tolerant": metrics["f1_tolerant"],
            "f1_tolerance": self.f1_tolerance
        }

        if print_results:
            print("")
            print("=" * 88)
            print("TEST ONLY")
            print(f"Test loss:             {results['loss']:.6f}")
            print(
                f"Exact:                 "
                f"P={results['precision']:.4f} | "
                f"R={results['recall']:.4f} | "
                f"F1={results['f1']:.4f}"
            )
            print(
                f"Within ±{self.f1_tolerance} timesteps: "
                f"P={results['precision_tolerant']:.4f} | "
                f"R={results['recall_tolerant']:.4f} | "
                f"F1={results['f1_tolerant']:.4f}"
            )
            print("=" * 88)
            print("")

        return results

    def fit(
        self,
        train_loader=None,
        test_loader=None,
        epochs_per_stage=10,
        fine_tune_epochs=0,
        plot=True,
        plot_every=1,
        tau=10.0,
        test_only=False
    ):
        if test_loader is None:
            raise ValueError("test_loader must be provided")

        if test_only:
            test_results = self.test(
                test_loader,
                plot=plot,
                tau=tau,
                print_results=True
            )

            final_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }

            return final_state, {
                "schedule": [stage.structure for stage in self.stages],
                "f1_tolerance": self.f1_tolerance,
                "patience": self.patience,
                "patience_metric": self.patience_metric,
                "patience_min_delta": self.patience_min_delta,
                "reuse_initial_decoder_for_final_layer": self.reuse_initial_decoder_for_final_layer,
                "stages": [],
                "fine_tune": None,
                "test_only": test_results
            }

        if train_loader is None:
            raise ValueError(
                "train_loader must be provided unless test_only=True. "
                "For evaluation only, call trainer.test(test_loader) or "
                "trainer.fit(train_loader=None, test_loader=test_loader, test_only=True)."
            )

        # A new fit() run should source the final-layer initialisation from the
        # decoder trained during this run's own first I -> M -> I stage.
        self._initial_decoder_state = None

        if isinstance(epochs_per_stage, int):
            stage_epochs = [epochs_per_stage] * len(self.stages)
        else:
            stage_epochs = list(epochs_per_stage)

            if len(stage_epochs) != len(self.stages):
                raise ValueError(
                    f"epochs_per_stage must be an int or a sequence of length {len(self.stages)}"
                )

        results = {
            "schedule": [stage.structure for stage in self.stages],
            "f1_tolerance": self.f1_tolerance,
            "patience": self.patience,
            "patience_metric": self.patience_metric,
            "patience_min_delta": self.patience_min_delta,
            "reuse_initial_decoder_for_final_layer": self.reuse_initial_decoder_for_final_layer,
            "stages": [],
            "fine_tune": None
        }

        self.describe_schedule()

        for stage_number, (stage, num_epochs) in enumerate(zip(self.stages, stage_epochs), start=1):
            stage_results = self._train_stage(
                train_loader,
                test_loader,
                stage,
                num_epochs,
                stage_number,
                plot,
                plot_every,
                tau
            )
            results["stages"].append(stage_results)

        if fine_tune_epochs > 0:
            results["fine_tune"] = self._fine_tune(
                train_loader,
                test_loader,
                fine_tune_epochs,
                plot,
                plot_every,
                tau
            )

        self._print_training_recap(results)

        self._unfreeze_all_model_parameters()
        final_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        return final_state, results

