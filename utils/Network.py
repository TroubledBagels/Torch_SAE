import torch
import snntorch
import snntorch.surrogate
import torch.nn as nn

class SingleLayerAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleLayerAutoencoder, self).__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder = nn.Linear(input_size, hidden_size)
        self.lif1 = snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.lif2 = snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad)

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

class MultilayerAE(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MultilayerAE, self).__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # Encoder
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad))
            prev_size = hidden_size

        # Decoder
        for hidden_size in reversed(hidden_sizes[:-1]):
            self.decoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.lif_layers.append(snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad))
            prev_size = hidden_size

        # Final decoder layer
        self.decoder_layers.append(nn.Linear(prev_size, input_size))
        self.lif_layers.append(snntorch.Leaky(beta=0.9, spike_grad=self.sur_grad))

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

class RecurrentSingleLayerAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, beta=0.9):
        super().__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder = nn.Linear(input_size, hidden_size)
        self.rlif1 = snntorch.RLeaky(beta=beta, linear_features=hidden_size, spike_grad=self.sur_grad)

        self.decoder = nn.Linear(hidden_size, input_size)
        self.rlif2 = snntorch.RLeaky(beta=beta, linear_features=input_size, spike_grad=self.sur_grad)

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

        spk_out = torch.stack(spk_out, dim=2)
        spk_lat = torch.stack(spk_lat, dim=2)

        if ret_lat:
            return spk_out, spk_lat

        return spk_out

class RecurrentSpikingAutoencoder(nn.Module):
    def __init__(self, input_size, recurrent_size, encoder_sizes, decoder_sizes, beta=0.9):
        super().__init__()

        self.sur_grad = snntorch.surrogate.fast_sigmoid(slope=25)

        self.encoder_layers = nn.ModuleList()
        self.encoder_lifs = nn.ModuleList()

        prev_size = input_size

        for size in encoder_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, size))
            self.encoder_lifs.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad))
            prev_size = size

        self.recurrent_linear = nn.Linear(prev_size, recurrent_size)
        self.recurrent_lif = snntorch.RLeaky(beta=beta, linear_features=recurrent_size, spike_grad=self.sur_grad)

        self.decoder_layers = nn.ModuleList()
        self.decoder_lifs = nn.ModuleList()

        prev_size = recurrent_size

        for size in decoder_sizes:
            self.decoder_layers.append(nn.Linear(prev_size, size))
            self.decoder_lifs.append(snntorch.Leaky(beta=beta, spike_grad=self.sur_grad))
            prev_size = size

        self.output_layer = nn.Linear(prev_size, input_size)
        self.output_lif = snntorch.Leaky(beta=beta, spike_grad=self.sur_grad)

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