import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import tqdm


# ============================================================
# Dataset
# ============================================================

class SinWaveDS(data.Dataset):
    def __init__(self, channels, timesteps, frequency=20, num_samples=1000):
        self.channels = channels
        self.timesteps = timesteps
        self.frequency = frequency
        self.num_samples = num_samples
        self.data = self.generate_data()

    def generate_data(self):
        data = torch.zeros((self.num_samples, self.channels, self.timesteps))

        for i in range(self.num_samples):
            phase_shift = torch.rand(1) * 2 * torch.pi

            for t in range(self.timesteps):
                angle = 2 * torch.pi * t / self.frequency + phase_shift
                s = torch.sin(angle)
                spike_c = ((s + 1) / 2 * (self.channels - 1)).long()
                data[i, spike_c, t] = 1.0

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0


# ============================================================
# Temporal non-spiking autoencoder
# ============================================================

class TemporalAutoencoder(nn.Module):
    def __init__(self, channels, latent_dim=16, decoder_dim=32):
        super().__init__()

        self.channels = channels
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim

        # Encoder processes one timestep at a time.
        self.encoder = nn.GRUCell(channels, latent_dim)

        # Decoder also has temporal state.
        self.decoder = nn.GRUCell(latent_dim, decoder_dim)

        # Reconstruct the channels for the current timestep.
        self.output = nn.Linear(decoder_dim, channels)

    def forward(self, x):
        batch_size, channels, timesteps = x.shape

        h_enc = torch.zeros(batch_size, self.latent_dim, device=x.device)
        h_dec = torch.zeros(batch_size, self.decoder_dim, device=x.device)

        outputs = []

        for t in range(timesteps):
            x_t = x[:, :, t]

            # Encode current timestep + previous temporal state.
            h_enc = self.encoder(x_t, h_enc)

            # h_enc is the temporal latent representation.
            z_t = h_enc

            # Decode temporally.
            h_dec = self.decoder(z_t, h_dec)

            # Raw reconstruction logits.
            out_t = self.output(h_dec)

            outputs.append(out_t)

        return torch.stack(outputs, dim=2)


# ============================================================
# Metrics
# ============================================================

def reconstruction_metrics(outputs, targets):
    predictions = torch.sigmoid(outputs) >= 0.5
    targets = targets >= 0.5

    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()

    return tp, fp, fn


# ============================================================
# Plot
# ============================================================

def plot_reconstruction(inputs, outputs, epoch):
    target = inputs[0].detach().cpu()
    reconstruction = (torch.sigmoid(outputs[0]) >= 0.5).float().detach().cpu()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].imshow(target, aspect="auto", interpolation="nearest", origin="lower")
    axes[0].set_title(f"Target - Epoch {epoch + 1}")
    axes[0].set_ylabel("Channel")

    axes[1].imshow(reconstruction, aspect="auto", interpolation="nearest", origin="lower")
    axes[1].set_title(f"Reconstruction - Epoch {epoch + 1}")
    axes[1].set_ylabel("Channel")
    axes[1].set_xlabel("Timestep")

    plt.tight_layout()
    plt.show()


# ============================================================
# Training
# ============================================================

def train_autoencoder(model, tr_dl, te_dl, optimizer, device="cuda", num_epochs=20, plot=True):
    model.to(device)

    pos_weight = torch.tensor([model.channels - 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm.tqdm(tr_dl)

        for inputs, _ in pbar:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            pbar.set_postfix(loss=loss.item())

        train_loss /= len(tr_dl.dataset)

        model.eval()

        test_loss = 0.0
        tp = 0
        fp = 0
        fn = 0

        plot_inputs = None
        plot_outputs = None

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(te_dl):
                inputs = inputs.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, inputs)
                test_loss += loss.item() * inputs.size(0)

                t_tp, t_fp, t_fn = reconstruction_metrics(outputs, inputs)

                tp += t_tp
                fp += t_fp
                fn += t_fn

                if batch_idx == 0:
                    plot_inputs = inputs
                    plot_outputs = outputs

        test_loss /= len(te_dl.dataset)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_loss:.6f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        if plot:
            plot_reconstruction(plot_inputs, plot_outputs, epoch)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    CHANNELS = 20
    TIMESTEPS = 1000
    FREQUENCY = 50
    NUM_SAMPLES = 1000

    LATENT_DIM = 16
    DECODER_DIM = 32

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    dataset = SinWaveDS(
        channels=CHANNELS,
        timesteps=TIMESTEPS,
        frequency=FREQUENCY,
        num_samples=NUM_SAMPLES
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    tr_dl = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    te_dl = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = TemporalAutoencoder(
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        decoder_dim=DECODER_DIM
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(model)

    train_autoencoder(
        model=model,
        tr_dl=tr_dl,
        te_dl=te_dl,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        plot=True
    )