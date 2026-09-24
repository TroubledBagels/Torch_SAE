import torch
import snntorch
import snntorch.surrogate
import torch.nn as nn
import torch.nn.functional as F

def surrogate_step(x, threshold=0.5, slope=25):
    hard = (x >= threshold).float()
    soft = torch.sigmoid(slope * (x - threshold))
    return (hard - soft).detach() + soft

class SingleLayerAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleLayerAutoencoder, self).__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.lif1 = snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.lif2 = snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad)

        self.name = "SingleLayerAutoencoder"

    def forward(self, x, ret_lat=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_lat = []
        spk_out = []

        for t in range(x.size(2)):
            spk1, mem1 = self.lif1(self.encoder(x[:, :, t]), mem1)
            spk2, mem2 = self.lif2(self.decoder(spk1), mem2)

            spk_lat.append(spk1)
            spk_out.append(spk2)

        if ret_lat:
            return torch.stack(spk_out, dim=2), torch.stack(spk_lat, dim=2)

        return torch.stack(spk_out, dim=2)

    def get_total_layers(self):
        return 2

class SingleLayerAutoencoderTrainable(nn.Module):
    def __init__(self, input_size, hidden_size, learn_beta=True, learn_threshold=True):
        super(SingleLayerAutoencoderTrainable, self).__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.lif1 = snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad, learn_beta=learn_beta, learn_threshold=learn_threshold)
        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.lif2 = snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad, learn_beta=learn_beta, learn_threshold=learn_threshold)

        self.name = "SingleLayerAutoencoderTrainable"

    def forward(self, x, ret_lat=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_lat = []
        spk_out = []

        for t in range(x.size(2)):
            spk1, mem1 = self.lif1(self.encoder(x[:, :, t]), mem1)
            spk2, mem2 = self.lif2(self.decoder(spk1), mem2)

            spk_lat.append(spk1)
            spk_out.append(spk2)

        if ret_lat:
            return torch.stack(spk_out, dim=2), torch.stack(spk_lat, dim=2)

        return torch.stack(spk_out, dim=2)

    def get_total_layers(self):
        return 2


class MultilayerAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, beta=0.9, threshold=1):
        super(MultilayerAE, self).__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # Encoder
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold))
            prev_size = hidden_size

        # Decoder
        for hidden_size in reversed(hidden_sizes[:-1]):
            self.decoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold))
            prev_size = hidden_size

        # Final decoder layer
        self.decoder_layers.append(nn.Linear(prev_size, input_size))
        self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold))

        self.name = "MultilayerAE"

    def forward(self, x, ret_lat=False):
        mems = [lif.init_leaky() for lif in self.lif_layers]

        spk_lat = []
        spk_out = []

        for t in range(x.size(2)):
            out = x[:, :, t]
            for i in range(len(self.encoder_layers)):
                out, mems[i] = self.lif_layers[i](self.encoder_layers[i](out), mems[i])
                if i == len(self.encoder_layers) - 1:
                    spk_lat.append(out)

            for i in range(len(self.decoder_layers)):
                out, mems[len(self.encoder_layers) + i] = self.lif_layers[len(self.encoder_layers) + i](self.decoder_layers[i](out), mems[len(self.encoder_layers) + i])

            spk_out.append(out)

        if ret_lat:
            return torch.stack(spk_out, dim=2), torch.stack(spk_lat, dim=2)

        return torch.stack(spk_out, dim=2)

    def get_total_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers)

    def freeze_but(self, layer_idx):
        temp_layer_list = self.encoder_layers + self.decoder_layers
        for i, layer in enumerate(temp_layer_list):
            if i != layer_idx:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def learn_all(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def freeze_all(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False


class MultilayerAETrainable(nn.Module):
    def __init__(self, input_size, hidden_sizes, beta=0.9, threshold=1.0):
        super(MultilayerAETrainable, self).__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # Encoder
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold, learn_beta=True, learn_threshold=True))
            prev_size = hidden_size

        # Decoder
        for hidden_size in reversed(hidden_sizes[:-1]):
            self.decoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold, learn_beta=True, learn_threshold=True))
            prev_size = hidden_size

        # Final decoder layer
        self.decoder_layers.append(nn.Linear(prev_size, input_size))
        self.lif_layers.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold, learn_beta=True, learn_threshold=True))

        self.name = "MultilayerAETrainable"

    def forward(self, x, ret_lat=False):
        mems = [lif.init_leaky() for lif in self.lif_layers]

        spk_lat = []
        spk_out = []

        for t in range(x.size(2)):
            out = x[:, :, t]
            for i in range(len(self.encoder_layers)):
                out, mems[i] = self.lif_layers[i](self.encoder_layers[i](out), mems[i])
                if i == len(self.encoder_layers) - 1:
                    spk_lat.append(out)

            for i in range(len(self.decoder_layers)):
                out, mems[len(self.encoder_layers) + i] = self.lif_layers[len(self.encoder_layers) + i](self.decoder_layers[i](out), mems[len(self.encoder_layers) + i])

            spk_out.append(out)

        if ret_lat:
            return torch.stack(spk_out, dim=2), torch.stack(spk_lat, dim=2)

        return torch.stack(spk_out, dim=2)

    def get_total_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers)

    def freeze_but(self, layer_idx):
        temp_layer_list = self.encoder_layers + self.decoder_layers
        for i, layer in enumerate(temp_layer_list):
            if i != layer_idx:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def learn_all(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def freeze_all(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def get_gradient_viewer_structure(self):
        layers = []
        stage_names = ["Input"]

        lif_idx = 0

        for i in range(len(self.encoder_layers)):
            layers.append({
                "module": f"encoder_layers.{i}",
                "neuron": f"lif_layers.{lif_idx}",
                "name": f"Encoder {i}"
            })

            stage_names.append(f"Encoder {i}")
            lif_idx += 1

        latent_stage = len(self.encoder_layers)
        stage_names[latent_stage] = "Latent"

        for i in range(len(self.decoder_layers)):
            layers.append({
                "module": f"decoder_layers.{i}",
                "neuron": f"lif_layers.{lif_idx}",
                "name": f"Decoder {i}"
            })

            if i == len(self.decoder_layers) - 1:
                stage_names.append("Output")
            else:
                stage_names.append(f"Decoder {i}")

            lif_idx += 1

        return {
            "layers": layers,
            "stage_names": stage_names,
            "latent_stage": latent_stage,
            "spike_threshold": 0.5
        }


class RecurrentSingleLayerAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, beta=0.9):
        super().__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder = nn.Linear(input_size, hidden_size)
        self.rlif1 = snntorch.RLeaky(beta=beta, linear_features=hidden_size, spike_grad=self.sur_grad)

        self.decoder = nn.Linear(hidden_size, input_size)
        self.rlif2 = snntorch.RLeaky(beta=beta, linear_features=input_size, spike_grad=self.sur_grad)

        self.name = "RecurrentSingleLayerAutoencoder"

    def forward(self, x, ret_lat=False):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()

        spk_lat = []
        spk_out = []

        for t in range(x.size(2)):
            cur1 = self.encoder(x[:, :, t])
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)

            cur2 = self.decoder(spk1)
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)

            spk_lat.append(spk1)
            spk_out.append(spk2)
            # spk_out.append(F.sigmoid(cur2))

        spk_out = torch.stack(spk_out, dim=2)
        spk_lat = torch.stack(spk_lat, dim=2)

        if ret_lat:
            return spk_out, spk_lat

        return spk_out

    def get_total_layers(self):
        return 2

class RecurrentSpikingAutoencoder(nn.Module):
    def __init__(self, input_size, recurrent_size, encoder_sizes, decoder_sizes, beta=0.9, multiply_weights=False, threshold=1.0):
        super().__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder_layers = nn.ModuleList()
        self.encoder_lifs = nn.ModuleList()

        prev_size = input_size

        for size in encoder_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, size))
            if multiply_weights:
                self.encoder_layers[-1].weight.data = self.encoder_layers[-1].weight.data * 10
            self.encoder_lifs.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold))
            prev_size = size

        self.recurrent_linear = nn.Linear(prev_size, recurrent_size)
        self.recurrent_lif = snntorch.RLeaky(beta=beta, linear_features=recurrent_size, spike_grad=self.sur_grad, threshold=threshold)

        self.decoder_layers = nn.ModuleList()
        self.decoder_lifs = nn.ModuleList()

        prev_size = recurrent_size

        for size in decoder_sizes:
            self.decoder_layers.append(nn.Linear(prev_size, size))
            if multiply_weights:
                self.decoder_layers[-1].weight.data = self.decoder_layers[-1].weight.data * 10
            self.decoder_lifs.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold))
            prev_size = size

        self.output_layer = nn.Linear(prev_size, input_size)
        self.output_lif = snntorch.Leaky(beta=beta, spike_grad=self.sur_grad, threshold=threshold)

        self.name = "RecurrentSpikingAutoencoder"

    def forward(self, x, ret_lat=False):
        encoder_mems = [lif.init_leaky() for lif in self.encoder_lifs]
        decoder_mems = [lif.init_leaky() for lif in self.decoder_lifs]

        rec_spk, rec_mem = self.recurrent_lif.init_rleaky()
        out_mem = self.output_lif.init_leaky()

        spk_lat = []
        spk_out = []

        for t in range(x.size(2)):
            out = x[:, :, t]

            for i in range(len(self.encoder_layers)):
                out, encoder_mems[i] = self.encoder_lifs[i](self.encoder_layers[i](out), encoder_mems[i])

            cur = self.recurrent_linear(out)
            rec_spk, rec_mem = self.recurrent_lif(cur, rec_spk, rec_mem)

            spk_lat.append(rec_spk)
            out = rec_spk

            for i in range(len(self.decoder_layers)):
                out, decoder_mems[i] = self.decoder_lifs[i](self.decoder_layers[i](out), decoder_mems[i])

            out, out_mem = self.output_lif(self.output_layer(out), out_mem)
            spk_out.append(out)

        spk_out = torch.stack(spk_out, dim=2)
        spk_lat = torch.stack(spk_lat, dim=2)

        if ret_lat:
            return spk_out, spk_lat

        return spk_out

    def get_total_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers) + 1

    def freeze_but(self, layer_idx):
        temp_layer_list = self.encoder_layers + [self.recurrent_linear] + self.decoder_layers
        for i, layer in enumerate(temp_layer_list):
            if i != layer_idx:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def learn_all(self):
        for layer in self.encoder_layers + [self.recurrent_linear] + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def freeze_all(self):
        for layer in self.encoder_layers + [self.recurrent_linear] + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

class RecurrentTanhAutoencoder(nn.Module):
    def __init__(self, input_size, recurrent_size, encoder_sizes, decoder_sizes):
        super().__init__()

        self.recurrent_size = recurrent_size

        self.encoder_layers = nn.ModuleList()

        prev_size = input_size
        for size in encoder_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, size))
            prev_size = size

        self.recurrent = nn.RNNCell(prev_size, recurrent_size, nonlinearity="tanh")

        self.decoder_layers = nn.ModuleList()

        prev_size = recurrent_size
        for size in decoder_sizes:
            self.decoder_layers.append(nn.Linear(prev_size, size))
            prev_size = size

        self.output_layer = nn.Linear(prev_size, input_size)

        self.name = "RecurrentTanhAutoencoder"

    def forward(self, x, ret_lat=False):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.recurrent_size, device=x.device)

        latents = []
        outputs = []

        for t in range(x.size(2)):
            out = x[:, :, t]

            for layer in self.encoder_layers:
                out = torch.tanh(layer(out))

            h = self.recurrent(out, h)
            latents.append(h)

            out = h

            for layer in self.decoder_layers:
                out = torch.tanh(layer(out))

            out = self.output_layer(out)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=2)
        latents = torch.stack(latents, dim=2)

        if ret_lat:
            return outputs, latents

        return outputs

    def get_total_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers) + 1

def step_threshold(x, threshold=0.5):
    return (x >= threshold).float()

class StepRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, threshold=0.5, slope=25):
        super().__init__()

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.recurrent_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.threshold = threshold
        self.slope = slope

    def step(self, x):
        hard = (x >= self.threshold).float()
        soft = torch.sigmoid(self.slope * (x - self.threshold))
        return (hard - soft).detach() + soft

    def forward(self, x, h):
        return self.step(self.input_linear(x) + self.recurrent_linear(h))

class RecurrentStepAutoencoder(nn.Module):
    def __init__(self, input_size, recurrent_size, encoder_sizes, decoder_sizes, multiply_weights=False):
        super().__init__()

        self.recurrent_size = recurrent_size

        self.encoder_layers = nn.ModuleList()

        prev_size = input_size
        for size in encoder_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, size))
            if multiply_weights:
                self.encoder_layers[-1].weight.data = self.encoder_layers[-1].weight.data * 10
            prev_size = size

        self.recurrent = StepRNNCell(prev_size, recurrent_size, threshold=0.5)

        self.decoder_layers = nn.ModuleList()

        prev_size = recurrent_size
        for size in decoder_sizes:
            self.decoder_layers.append(nn.Linear(prev_size, size))
            if multiply_weights:
                self.decoder_layers[-1].weight.data = self.decoder_layers[-1].weight.data * 10
            prev_size = size

        self.output_layer = nn.Linear(prev_size, input_size)

        self.name = "RecurrentStepAutoencoder"

    def forward(self, x, ret_lat=False):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.recurrent_size, device=x.device)

        latents = []
        outputs = []

        for t in range(x.size(2)):
            # print("====================================================================")
            out = x[:, :, t]
            # print(out)

            for layer in self.encoder_layers:
                out = surrogate_step(layer(out))
                # print(out)

            h = self.recurrent(out, h)
            latents.append(h)

            out = surrogate_step(h)
            # print(out)

            for layer in self.decoder_layers:
                out = surrogate_step(layer(out))
                # print(out)

            out = self.output_layer(out)

            out = surrogate_step(out)
            # print(out)
            # out = step_threshold(out)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=2)
        latents = torch.stack(latents, dim=2)

        if ret_lat:
            return outputs, latents

        return outputs

    def get_total_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers) + 1

class UNetSpikingAutoencoder(nn.Module):
    """
    Feedforward spiking U-Net autoencoder for inputs shaped [B, C, T].

    hidden_sizes contains the encoder widths, including the latent width.

    Example:
        input_size=20
        hidden_sizes=[32, 24, 15]

        Encoder:
            20 -> 32 -> 24 -> 15

        Decoder:
            15 -> 24 -> 32 -> 20

    Skip connections join matching encoder and decoder resolutions.

    concat_mode="spatial":
        Decoder and skip spikes are concatenated across the feature/channel
        dimension at each timestep.

        Example:
            [B, 24, T] + [B, 24, T] -> [B, 48, T]

    concat_mode="temporal":
        Skip spikes are presented first, followed by decoder spikes, through
        the next spiking decoder layer. The two sequences are concatenated
        along time internally:

            [B, 24, T] + [B, 24, T] -> [B, 24, 2T]

        The LIF state is carried across the complete 2T sequence and only the
        decoder-half outputs are retained, returning the external sequence
        length to T. This lets the skip connection influence the decoder
        through temporal membrane state rather than extra spatial features.
    """

    def __init__(self, input_size, hidden_sizes, beta=0.9, threshold=1.0, learn_beta=True, learn_threshold=True, concat_mode="spatial"):
        super(UNetSpikingAutoencoder, self).__init__()

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer size")

        if concat_mode not in ("spatial", "temporal"):
            raise ValueError("concat_mode must be 'spatial' or 'temporal'")

        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.concat_mode = concat_mode
        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=float(beta), spike_grad=self.sur_grad, threshold=float(threshold), learn_beta=learn_beta, learn_threshold=learn_threshold))
            prev_size = hidden_size

        decoder_sizes = list(reversed(hidden_sizes[:-1])) + [input_size]
        prev_size = hidden_sizes[-1]

        for i, decoder_size in enumerate(decoder_sizes):
            decoder_input_size = prev_size

            if concat_mode == "spatial" and i > 0:
                decoder_input_size = prev_size * 2

            self.decoder_layers.append(nn.Linear(decoder_input_size, decoder_size))
            self.lif_layers.append(snntorch.Leaky(beta=float(beta), spike_grad=self.sur_grad, threshold=float(threshold), learn_beta=learn_beta, learn_threshold=learn_threshold))
            prev_size = decoder_size

        self.name = f"UNetSpikingAutoencoder_{concat_mode}"

    def _run_spiking_layer(self, x, layer, lif):
        mem = lif.init_leaky()
        spikes = []

        for t in range(x.size(2)):
            spk, mem = lif(layer(x[:, :, t]), mem)
            spikes.append(spk)

        return torch.stack(spikes, dim=2)

    def _run_temporal_concat_layer(self, decoder_spikes, skip_spikes, layer, lif):
        if decoder_spikes.size(1) != skip_spikes.size(1):
            raise ValueError(
                f"Temporal concatenation requires matching feature sizes, got "
                f"{decoder_spikes.size(1)} and {skip_spikes.size(1)}"
            )

        if decoder_spikes.size(2) != skip_spikes.size(2):
            raise ValueError(
                f"Temporal concatenation requires matching sequence lengths, got "
                f"{decoder_spikes.size(2)} and {skip_spikes.size(2)}"
            )

        timesteps = decoder_spikes.size(2)
        temporal_input = torch.cat((skip_spikes, decoder_spikes), dim=2)
        temporal_output = self._run_spiking_layer(temporal_input, layer, lif)

        return temporal_output[:, :, timesteps:]

    def forward(self, x, ret_lat=False):
        encoder_spikes = []
        out = x

        for i in range(len(self.encoder_layers)):
            out = self._run_spiking_layer(out, self.encoder_layers[i], self.lif_layers[i])
            encoder_spikes.append(out)

        spk_lat = encoder_spikes[-1]
        out = spk_lat

        for i in range(len(self.decoder_layers)):
            lif_idx = len(self.encoder_layers) + i

            if i == 0:
                out = self._run_spiking_layer(out, self.decoder_layers[i], self.lif_layers[lif_idx])

                continue

            skip_idx = len(encoder_spikes) - i - 1
            skip = encoder_spikes[skip_idx]

            if self.concat_mode == "spatial":
                out = torch.cat((out, skip), dim=1)
                out = self._run_spiking_layer(out, self.decoder_layers[i], self.lif_layers[lif_idx])

            if self.concat_mode == "temporal":
                out = self._run_temporal_concat_layer(out, skip, self.decoder_layers[i], self.lif_layers[lif_idx])

        if ret_lat:
            return out, spk_lat

        return out

    def get_total_layers(self):
        return len(self.encoder_layers) + len(self.decoder_layers)

    def freeze_but(self, layer_idx):
        layers = list(self.encoder_layers) + list(self.decoder_layers)

        for i, layer in enumerate(layers):
            requires_grad = i == layer_idx

            for param in layer.parameters():
                param.requires_grad = requires_grad

            for param in self.lif_layers[i].parameters():
                param.requires_grad = requires_grad

    def learn_all(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = True

        for lif in self.lif_layers:
            for param in lif.parameters():
                param.requires_grad = True

    def freeze_all(self):
        for layer in self.encoder_layers + self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

        for lif in self.lif_layers:
            for param in lif.parameters():
                param.requires_grad = False