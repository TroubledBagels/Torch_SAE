import torch
import torch.nn as nn
from dataclasses import dataclass
import snntorch as snn
import snntorch.surrogate

@dataclass
class RecurrentParams:
    hidden_size: int
    beta: float = 0.9
    threshold: float = 1.0
    spectral_radius: float = 0.9
    train_recurrent: bool = True
    self_connections: bool = False


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = 1 / (1 + 25 * x.abs()) ** 2
        return grad_output * grad


spike_fn = SurrogateSpike.apply

class LIFProjection(nn.Module):
    def __init__(self, input_size, latent_size, beta=0.9, threshold=1.0, train_mode="backprop"):
        super().__init__()

        assert train_mode in ["backprop", "ga"]

        self.input_size = input_size
        self.latent_size = latent_size
        self.train_mode = train_mode

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.projection = nn.Linear(input_size, latent_size)
        self.lif = snntorch.Leaky(beta=beta, threshold=threshold, spike_grad=self.sur_grad)

        if train_mode == "ga":
            for param in self.projection.parameters():
                param.requires_grad = False

    def forward(self, x, ret_separate=False):
        mem = self.lif.init_leaky()

        latent = []
        mem_rec = []
        spk_rec = []

        for t in range(x.size(2)):
            cur = self.projection(x[:, :, t])
            spk, mem = self.lif(cur, mem)

            mem_rec.append(mem)
            spk_rec.append(spk)

            combined = torch.stack((mem, spk), dim=2).flatten(1)
            latent.append(combined)

        latent = torch.stack(latent, dim=2)

        if ret_separate:
            mem_rec = torch.stack(mem_rec, dim=2)
            spk_rec = torch.stack(spk_rec, dim=2)
            return latent, mem_rec, spk_rec

        return latent

    def get_ga_params(self):
        if self.train_mode == "ga":
            return list(self.projection.parameters())

        return []

    def get_backprop_params(self):
        if self.train_mode == "backprop":
            return list(self.projection.parameters())

        return []

class AllToAllLIF(nn.Module):
    def __init__(self, input_size, recurrent_params):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = int(recurrent_params.hidden_size)
        self.beta = recurrent_params.beta
        self.threshold = recurrent_params.threshold

        self.input_layer = nn.Linear(input_size, self.hidden_size)
        self.recurrent = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.recurrent.weight)

        mask = torch.ones(self.hidden_size, self.hidden_size)

        if not recurrent_params.self_connections:
            mask.fill_diagonal_(0)

        self.register_buffer("recurrent_mask", mask)

        self._scale_spectral_radius(recurrent_params.spectral_radius)

        self.recurrent.weight.requires_grad = recurrent_params.train_recurrent

    def _scale_spectral_radius(self, target_radius):
        with torch.no_grad():
            weight = self.recurrent.weight * self.recurrent_mask
            radius = torch.linalg.eigvals(weight).abs().max().real

            if radius > 0:
                self.recurrent.weight.mul_(target_radius / radius)

    def init_state(self, batch_size, device):
        spk = torch.zeros(batch_size, self.hidden_size, device=device)
        mem = torch.zeros(batch_size, self.hidden_size, device=device)
        return spk, mem

    def forward_step(self, x, spk, mem):
        input_current = self.input_layer(x)
        recurrent_current = nn.functional.linear(spk, self.recurrent.weight * self.recurrent_mask)

        mem = self.beta * mem + input_current + recurrent_current
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk.detach() * self.threshold

        return spk, mem

    def forward(self, x, ret_mem=False):
        batch_size = x.size(0)
        spk, mem = self.init_state(batch_size, x.device)

        spk_rec = []
        mem_rec = []

        for t in range(x.size(2)):
            spk, mem = self.forward_step(x[:, :, t], spk, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        spk_rec = torch.stack(spk_rec, dim=2)
        mem_rec = torch.stack(mem_rec, dim=2)

        if ret_mem:
            return spk_rec, mem_rec

        return spk_rec

class MixedNetwork(nn.Module):
    def __init__(self, input_size, recurrent_params, decoder_sizes: list, latent_size=10, projection_train_mode='ga'):
        super().__init__()

        self.input_size = input_size
        self.recurrent_params = recurrent_params

        self.encoder = AllToAllLIF(input_size, recurrent_params)
        self.projection = LIFProjection(
            input_size=recurrent_params.hidden_size,
            latent_size=latent_size,
            beta=0.9,
            threshold=1.0,
            train_mode=projection_train_mode
        )

        cur_size = latent_size * 2
        self.decoder_layers = nn.ModuleList()

        for d_l in decoder_sizes:
            self.decoder_layers.append(nn.Linear(cur_size, d_l))
            cur_size = d_l

        self.output_layer = nn.Linear(cur_size, input_size)

    def forward(self, x, ret_lat=False):
        encoded = self.encoder(x)
        latent = self.projection(encoded)

        outputs = []

        for t in range(latent.size(2)):
            out = latent[:, :, t]

            for layer in self.decoder_layers:
                out = torch.relu(layer(out))

            out = self.output_layer(out)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=2)

        if ret_lat:
            return outputs, latent

        return outputs