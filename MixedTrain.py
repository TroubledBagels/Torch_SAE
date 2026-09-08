import torch
import tqdm
import utils.CMAES as C
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def train_mixed_network(model, tr_dl, te_dl, criterion, device="cuda", num_epochs=20, decoder_lr=1e-3, sigma=0.05, population_size=None, cma_max_batches=None, plot=False, cma_generations_per_epoch=3):
    model.to(device)

    decoder_params = list(model.decoder_layers.parameters()) + list(model.output_layer.parameters())

    if hasattr(model, "projection") and model.projection.train_mode == "backprop":
        decoder_params += model.projection.get_backprop_params()
        print("Training with backprop mode")
    elif hasattr(model, "projection") and model.projection.train_mode != "backprop":
        print("Training with CMAES mode")

    decoder_optimizer = torch.optim.Adam(decoder_params, lr=decoder_lr)

    initial_encoder = C.ga_to_vector(model)
    cma = C.CMAES(initial_encoder, sigma=sigma, population_size=population_size)

    for epoch in range(num_epochs):

        # ============================================================
        # 1. Train encoder using one CMA-ES generation
        # ============================================================

        for cma_gen in range(cma_generations_per_epoch):
            candidates, y = cma.ask()
            fitness = torch.empty(cma.pop_size, device=device)

            pbar = tqdm.tqdm(range(cma.pop_size),
                             desc=f"Epoch {epoch + 1} CMA-ES {cma_gen + 1}/{cma_generations_per_epoch}")

            for i in pbar:
                C.vector_to_ga(model, candidates[i])

                fitness[i], _, _, _ = C.evaluate_model(
                    model=model,
                    dataloader=tr_dl,
                    criterion=criterion,
                    device=device,
                    max_batches=cma_max_batches
                )

                pbar.set_postfix(
                    best=fitness[:i + 1].min().item(),
                    mean=fitness[:i + 1].mean().item()
                )

            cma.tell(candidates, y, fitness)
            C.vector_to_ga(model, cma.mean)

        cma_loss, _, _, _ = C.evaluate_model(
            model=model,
            dataloader=tr_dl,
            criterion=criterion,
            device=device,
            max_batches=cma_max_batches
        )

        # ============================================================
        # 2. Train decoder using backprop
        # ============================================================

        model.train()
        decoder_running_loss = 0.0
        total_samples = 0

        pbar = tqdm.tqdm(tr_dl, desc=f"Epoch {epoch + 1} Decoder")

        for inputs, _ in pbar:
            inputs = inputs.to(device)

            decoder_optimizer.zero_grad()
            model.zero_grad(set_to_none=True)

            outputs = model(inputs)
            processed_outputs = C.process_outputs(outputs, criterion)
            loss = criterion(processed_outputs, inputs)

            loss.backward()
            decoder_optimizer.step()

            decoder_running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            pbar.set_postfix(loss=decoder_running_loss / total_samples)

        decoder_loss = decoder_running_loss / total_samples

        # ============================================================
        # 3. Test
        # ============================================================

        test_loss, prec, rec, f1 = C.evaluate_model(
            model=model,
            dataloader=te_dl,
            criterion=criterion,
            device=device,
            plot=plot
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | CMA Loss: {cma_loss:.6f} | Decoder Loss: {decoder_loss:.6f} | Sigma: {cma.sigma:.6f}\n"
            f"Test Loss: {test_loss:.6f} | Precision: {prec:.6f} | Recall: {rec:.6f} | F1: {f1:.6f}"
        )

    C.vector_to_ga(model, cma.mean)

    return model