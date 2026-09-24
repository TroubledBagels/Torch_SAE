import torch
import torch.nn.functional as F
import snntorch as snn
import utils.Network as N
import tqdm
import matplotlib.pyplot as plt

def align_for_delay(inputs, outputs, delay=0):
    """
    Align network output with the desired target in time.

    delay > 0:
        reconstruct the past
        output[t] is compared with input[t - delay]

    delay < 0:
        predict the future
        output[t] is compared with input[t + abs(delay)]

    delay == 0:
        standard reconstruction

    Invalid boundary timesteps are cropped rather than zero padded.
    """
    if inputs.shape[-1] != outputs.shape[-1]:
        raise ValueError(f"Input length {inputs.shape[-1]} does not match output length {outputs.shape[-1]}")

    timesteps = inputs.shape[-1]

    if abs(delay) >= timesteps:
        raise ValueError(f"abs(delay) must be smaller than the sequence length ({timesteps}), got {delay}")

    if delay > 0:
        aligned_targets = inputs[..., :-delay]
        aligned_outputs = outputs[..., delay:]
    elif delay < 0:
        future = -delay
        aligned_targets = inputs[..., future:]
        aligned_outputs = outputs[..., :-future]
    else:
        aligned_targets = inputs
        aligned_outputs = outputs

    return aligned_outputs, aligned_targets


def plot_reconstruction(inputs, outputs, epoch, criterion, tau=10.0, delay=0):
    outputs, inputs = align_for_delay(inputs, outputs, delay)

    target = inputs[0].detach().cpu()
    output = outputs[0].detach().cpu()

    criterion_name = getattr(criterion, "__name__", criterion.__class__.__name__).lower()

    if "rossum" in criterion_name:
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
    axes[2].set_title('Loss Per Timestep')
    axes[2].set_ylabel('Loss')
    axes[2].set_xlabel('Timestep')
    axes[2].grid()

    plt.tight_layout()
    plt.show()

def van_rossum_loss(pred_spikes, target_spikes, tau=10.0):
    alpha = torch.exp(torch.tensor(-1.0 / tau, device=pred_spikes.device))

    pred_trace = torch.zeros_like(pred_spikes)
    target_trace = torch.zeros_like(target_spikes)

    pred_trace[..., 0] = pred_spikes[..., 0]
    target_trace[..., 0] = target_spikes[..., 0]

    for t in range(1, pred_spikes.shape[-1]):
        pred_trace[..., t] = alpha * pred_trace[..., t - 1] + pred_spikes[..., t]
        target_trace[..., t] = alpha * target_trace[..., t - 1] + target_spikes[..., t]


    return F.mse_loss(pred_trace, target_trace)

def van_rossum_loss_count(pred_spikes, target_spikes, tau=10.0, count_weight=1.0):
    alpha = torch.exp(torch.tensor(-1.0 / tau, device=pred_spikes.device))

    pred_trace = torch.zeros_like(pred_spikes)
    target_trace = torch.zeros_like(target_spikes)

    pred_trace[..., 0] = pred_spikes[..., 0]
    target_trace[..., 0] = target_spikes[..., 0]

    for t in range(1, pred_spikes.shape[-1]):
        pred_trace[..., t] = alpha * pred_trace[..., t - 1] + pred_spikes[..., t]
        target_trace[..., t] = alpha * target_trace[..., t - 1] + target_spikes[..., t]

    vr_loss = F.mse_loss(pred_trace, target_trace)

    pred_count = pred_spikes.sum(dim=-1)
    target_count = target_spikes.sum(dim=-1)

    timesteps = pred_spikes.shape[-1]
    count_loss = F.mse_loss(pred_count / timesteps, target_count / timesteps)

    return vr_loss + count_weight * count_loss

def reconstruction_metrics(outputs, targets):
    outputs = outputs.reshape(outputs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    tps = ((outputs == 1) & (targets == 1)).sum()
    fps = ((outputs == 1) & (targets == 0)).sum()
    fns = ((outputs == 0) & (targets == 1)).sum()

    return tps, fps, fns

def _normal_train_multilayer(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs, delay=0):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm.tqdm(tr_dl)
        for i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs, latents = model(inputs, ret_lat=True)
            aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)
            loss = criterion(aligned_outputs, aligned_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/(i+1))

        epoch_loss = running_loss / len(tr_dl.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        tp = 0
        fp = 0
        fn = 0

        qbar = tqdm.tqdm(te_dl)

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(qbar):
                inputs = inputs.to(device)
                outputs, latents = model(inputs, ret_lat=True)
                aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)

                loss = criterion(aligned_outputs, aligned_targets)
                test_loss += loss.item() * inputs.size(0)

                t_tp, t_fp, t_fn = reconstruction_metrics(aligned_outputs, aligned_targets)
                tp += t_tp.sum().item()
                fp += t_fp.sum().item()
                fn += t_fn.sum().item()

                if batch_idx == 0:
                    print("Target spikes:", aligned_targets[0].sum().item())
                    print("Output spikes:", aligned_outputs[0].sum().item())
                    plot_reconstruction(inputs, outputs, epoch, criterion, delay=delay)

        test_loss /= len(te_dl.dataset)
        print(f'Test Loss: {test_loss:.4f}')
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

def normal_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=10, delay=0):
    # if isinstance(model, N.MultilayerAE) or isinstance(model, N.RecurrentSpikingAutoencoder):
    #     return _normal_train_multilayer(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs, delay=delay)

    print("Entered normal_train")

    model.to(device)
    model.train()

    cur_best_f1 = 0.0
    cur_best = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(tr_dl)
        for i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)

            # inputs = inputs.reshape(inputs.size(0), inputs.size(2), inputs.size(1))

            optimizer.zero_grad()
            outputs = model(inputs)
            aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)
            loss = criterion(aligned_outputs, aligned_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/(i+1))
            # exit()

        epoch_loss = running_loss / len(tr_dl.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        tp = 0
        fp = 0
        fn = 0

        qbar = tqdm.tqdm(te_dl)

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(qbar):
                inputs = inputs.to(device)
                # inputs = inputs.reshape(inputs.size(0), inputs.size(2), inputs.size(1))
                outputs = model(inputs)
                aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)

                loss = criterion(aligned_outputs, aligned_targets)
                test_loss += loss.item() * inputs.size(0)

                t_tp, t_fp, t_fn = reconstruction_metrics(aligned_outputs, aligned_targets)
                tp += t_tp.sum().item()
                fp += t_fp.sum().item()
                fn += t_fn.sum().item()

                if batch_idx == 0:
                    print("Target spikes:", aligned_targets[0].sum().item())
                    print("Output spikes:", aligned_outputs[0].sum().item())
                    plot_reconstruction(inputs, outputs, epoch, criterion, delay=delay)

        test_loss /= len(te_dl.dataset)
        print(f'Test Loss: {test_loss:.4f}')
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

        if f1 > cur_best_f1:
            cur_best_f1 = f1
            cur_best = model.state_dict()

    print(f"Trained model with best f1 score: {cur_best_f1:.4f}")

    return cur_best



def _freeze_train_batch(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs, backwards, delay=0):
    model.to(device)
    model.train()

    is_multilayer = isinstance(model, N.MultilayerAE) or isinstance(model, N.RecurrentSpikingAutoencoder)

    if not is_multilayer:
        return normal_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=num_epochs, delay=delay)

    assert isinstance(model, N.MultilayerAE)

    total_layers = model.get_total_layers()

    if backwards:
        cur_training = total_layers - 1
        model.freeze_but(cur_training)
    else:
        cur_training = 0
        model.freeze_but(cur_training)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in tr_dl:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)
            loss = criterion(aligned_outputs, aligned_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if backwards:
                cur_training -= 1
                if cur_training < 0:
                    cur_training = total_layers - 1
                model.freeze_but(cur_training)
            else:
                cur_training += 1
                if cur_training >= total_layers:
                    cur_training = 0
                model.freeze_but(cur_training)

        epoch_loss = running_loss / len(tr_dl.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(te_dl):
                inputs = inputs.to(device)
                outputs = model(inputs)
                aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)

                loss = criterion(aligned_outputs, aligned_targets)
                test_loss += loss.item() * inputs.size(0)

                t_tp, t_fp, t_fn = reconstruction_metrics(aligned_outputs, aligned_targets)
                tp += t_tp.sum().item()
                fp += t_fp.sum().item()
                fn += t_fn.sum().item()

                if batch_idx == 0:
                    print("Target spikes:", aligned_targets[0].sum().item())
                    print("Output spikes:", aligned_outputs[0].sum().item())
                    plot_reconstruction(inputs, outputs, epoch, criterion, delay=delay)

        test_loss /= len(te_dl.dataset)
        print(f'Test Loss: {test_loss:.4f}')

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

        model.train()

    return None

def freeze_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=10, backwards=True, batchwise=False, delay=0):
    if batchwise:
        return _freeze_train_batch(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs, backwards, delay=delay)

    model.to(device)
    model.train()

    is_multilayer = isinstance(model, N.MultilayerAE) or isinstance(model, N.RecurrentSpikingAutoencoder)

    if not is_multilayer:
        return normal_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=num_epochs, delay=delay)

    total_layers = model.get_total_layers()

    if backwards:
        cur_training = total_layers - 1
        model.freeze_but(cur_training)
    else:
        cur_training = 0
        model.freeze_but(cur_training)

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in tr_dl:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)
            loss = criterion(aligned_outputs, aligned_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(tr_dl.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(te_dl):
                inputs = inputs.to(device)
                outputs = model(inputs)
                aligned_outputs, aligned_targets = align_for_delay(inputs, outputs, delay)

                loss = criterion(aligned_outputs, aligned_targets)
                test_loss += loss.item() * inputs.size(0)

                t_tp, t_fp, t_fn = reconstruction_metrics(aligned_outputs, aligned_targets)
                tp += t_tp.sum().item()
                fp += t_fp.sum().item()
                fn += t_fn.sum().item()

                if batch_idx == 0:
                    print("Target spikes:", aligned_targets[0].sum().item())
                    print("Output spikes:", aligned_outputs[0].sum().item())
                    plot_reconstruction(inputs, outputs, epoch, criterion, delay=delay)

        test_loss /= len(te_dl.dataset)
        print(f'Test Loss: {test_loss:.4f}')

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch

        model.train()
        if backwards:
            cur_training -= 1
            if cur_training < 0:
                cur_training = total_layers - 1
            model.freeze_but(cur_training)
        else:
            cur_training += 1
            if cur_training >= total_layers:
                cur_training = 0
            model.freeze_but(cur_training)

        print(f"Learning layer index: {cur_training}")

    print(f"Best f1 score: {best_f1:.4f}")
    print(f"Best epoch index: {best_epoch}")
    return None