import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CMAES:
    def __init__(self, mean, sigma=0.05, population_size=None):
        self.mean = mean.clone()
        self.n = mean.numel()
        self.sigma = sigma

        self.pop_size = population_size or 4 + int(3 * math.log(self.n))
        self.mu = self.pop_size // 2

        weights = torch.log(torch.tensor(self.mu + 0.5, device=mean.device)) - torch.log(torch.arange(1, self.mu + 1, device=mean.device).float())
        self.weights = weights / weights.sum()
        self.mu_eff = 1 / torch.sum(self.weights ** 2).item()

        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs

        self.cov = torch.ones(self.n, device=mean.device)
        self.pc = torch.zeros(self.n, device=mean.device)
        self.ps = torch.zeros(self.n, device=mean.device)

        self.chi_n = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        self.generation = 0

    def ask(self):
        z = torch.randn(self.pop_size, self.n, device=self.mean.device)
        y = z * torch.sqrt(self.cov)
        candidates = self.mean.unsqueeze(0) + self.sigma * y
        return candidates, y

    def tell(self, candidates, y, fitness):
        order = torch.argsort(fitness)
        y_best = y[order[:self.mu]]
        y_mean = torch.sum(self.weights[:, None] * y_best, dim=0)

        self.mean = self.mean + self.sigma * y_mean

        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * y_mean / torch.sqrt(self.cov)

        ps_norm = torch.linalg.vector_norm(self.ps).item()
        denominator = math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
        hsig = float(ps_norm / denominator < (1.4 + 2 / (self.n + 1)) * self.chi_n)

        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_mean

        rank_mu = torch.sum(self.weights[:, None] * (y_best ** 2), dim=0)
        correction = self.c1 * (1 - hsig) * self.cc * (2 - self.cc)

        self.cov = (1 - self.c1 - self.cmu + correction) * self.cov + self.c1 * self.pc ** 2 + self.cmu * rank_mu
        self.cov = torch.clamp(self.cov, min=1e-12)

        self.sigma *= math.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1))
        self.generation += 1


def get_loss_type(criterion):
    if isinstance(criterion, nn.MSELoss):
        return "mse"

    name = getattr(criterion, "__name__", criterion.__class__.__name__).lower()

    if "mse" in name:
        return "mse"

    if "rossum" in name or "van_rossum" in name:
        return "vr"

    raise ValueError(f"Unknown loss type: {name}")


def process_outputs(outputs, criterion):
    loss_type = get_loss_type(criterion)

    if loss_type == "mse":
        return outputs

    if loss_type == "vr":
        return torch.sigmoid(outputs)

def get_encoder_params(model):
    return [p for p in model.encoder.parameters() if p.requires_grad]


def get_ga_params(model):
    params = list(model.encoder.parameters())

    if hasattr(model, "projection") and model.projection.train_mode == "ga":
        params += model.projection.get_ga_params()

    return params


def ga_to_vector(model):
    params = get_ga_params(model)
    return torch.cat([p.detach().flatten() for p in params])


def vector_to_ga(model, vector):
    params = get_ga_params(model)

    offset = 0

    with torch.no_grad():
        for p in params:
            numel = p.numel()
            p.copy_(vector[offset:offset + numel].view_as(p))
            offset += numel

    assert offset == vector.numel()


def reconstruction_metrics(outputs, targets, criterion, threshold=0.5):
    outputs = process_outputs(outputs, criterion)

    predictions = outputs >= threshold
    targets = targets >= 0.5

    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()

    return tp, fp, fn

def plot_reconstruction(inputs, outputs, criterion, threshold=0.5):
    loss_type = get_loss_type(criterion)

    target = inputs[0].detach().cpu()
    raw_output = outputs[0].detach().cpu()

    if loss_type == "mse":
        processed_output = raw_output
        title = "Raw Output (MSE)"
    else:
        processed_output = torch.sigmoid(raw_output)
        title = "Sigmoid Output (Van Rossum)"

    reconstruction = (processed_output >= threshold).float()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].imshow(target, aspect="auto", interpolation="nearest", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Target")
    axes[0].set_ylabel("Channel")

    im = axes[1].imshow(processed_output, aspect="auto", interpolation="nearest", origin="lower")
    axes[1].set_title(title)
    axes[1].set_ylabel("Channel")
    fig.colorbar(im, ax=axes[1])

    axes[2].imshow(reconstruction, aspect="auto", interpolation="nearest", origin="lower", vmin=0, vmax=1)
    axes[2].set_title(f"Thresholded Reconstruction ({threshold})")
    axes[2].set_ylabel("Channel")
    axes[2].set_xlabel("Timestep")

    plt.tight_layout()
    plt.show()


def evaluate_model(model, dataloader, criterion, device, max_batches=None, threshold=0.5, plot=False):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    tp = 0
    fp = 0
    fn = 0

    plot_inputs = None
    plot_outputs = None

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = inputs.to(device)
            outputs = model(inputs)

            processed_outputs = process_outputs(outputs, criterion)
            loss = criterion(processed_outputs, inputs)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            t_tp, t_fp, t_fn = reconstruction_metrics(outputs, inputs, criterion, threshold)

            tp += t_tp
            fp += t_fp
            fn += t_fn

            if plot_inputs is None:
                plot_inputs = inputs
                plot_outputs = outputs

    loss = total_loss / total_samples

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    if plot and plot_inputs is not None:
        plot_reconstruction(plot_inputs, plot_outputs, criterion, threshold)

    return loss, precision, recall, f1