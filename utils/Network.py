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

        for t in range(x.size(1)):
            mem1, spk1 = self.lif1(self.encoder(x[:, :, t]), mem1)
            mem2, spk2 = self.lif2(self.decoder(spk1), mem2)

            spk_lat.append(spk1)
            spk_out.append(spk2)

        if ret_lat:
            return torch.stack(spk_out, dim=1), torch.stack(spk_lat, dim=1)

        return torch.stack(spk_out, dim=1)

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

        for t in range(x.size(1)):
            out = x[:, :, t]
            for i in range(len(self.encoder_layers)):
                mems[i], out = self.lif_layers[i](self.encoder_layers[i](out), mems[i])
                if i == len(self.encoder_layers) - 1:
                    spk_lat.append(out)

            for i in range(len(self.decoder_layers)):
                mems[len(self.encoder_layers) + i], out = self.lif_layers[len(self.encoder_layers) + i](self.decoder_layers[i](out), mems[len(self.encoder_layers) + i])

            spk_out.append(out)

        if ret_lat:
            return torch.stack(spk_out, dim=1), torch.stack(spk_lat, dim=1)

        return torch.stack(spk_out, dim=1)

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