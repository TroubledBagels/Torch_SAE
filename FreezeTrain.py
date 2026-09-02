import torch
import snntorch as snn
import utils.Network as N

def reconstruction_metrics(outputs, targets):
    # In form (B, C, T) where B is batch size, C is number of channels,
    # and T is number of time steps and data is a spike train
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tps = ((outputs == 1) & (targets == 1)).sum(dim=1).float()
    fps = ((outputs == 1) & (targets == 0)).sum(dim=1).float()
    fns = ((outputs == 0) & (targets == 1)).sum(dim=1).float()

    return tps, fps, fns

def normal_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in tr_dl:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
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
        for inputs, _ in te_dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item() * inputs.size(0)
            t_tp, t_fp, t_fn = reconstruction_metrics(outputs, inputs)
            tp += t_tp.sum().item()
            fp += t_fp.sum().item()
            fn += t_fn.sum().item()

    test_loss /= len(te_dl.dataset)
    print(f'Test Loss: {test_loss:.4f}')



def _freeze_train_batch(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs, backwards):
    pass

def freeze_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=10, backwards=True, batchwise=False):
    model.to(device)
    model.train()

    is_multilayer = isinstance(model, N.MultilayerAE)

    if not is_multilayer:
        return normal_train(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs=num_epochs)

    assert isinstance(model, N.MultilayerAE)

    total_layers = model.get_total_layers()

    if backwards:
        cur_training = total_layers - 1
        model.freeze_but(cur_training)
    else:
        cur_training = 0
        model.freeze_but(cur_training)

    if batchwise:
        _freeze_train_batch(model, tr_dl, te_dl, optimizer, criterion, device, num_epochs, backwards)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in tr_dl:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
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
            for inputs, _ in te_dl:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item() * inputs.size(0)
                t_tp, t_fp, t_fn = reconstruction_metrics(outputs, inputs)
                tp += t_tp.sum().item()
                fp += t_fp.sum().item()
                fn += t_fn.sum().item()

        test_loss /= len(te_dl.dataset)
        print(f'Test Loss: {test_loss:.4f}')

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

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

    return None