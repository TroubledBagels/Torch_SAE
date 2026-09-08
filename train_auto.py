import torch
import torch.optim as optim
import utils.Network as N
import utils.GenData as GD
import FreezeTrain as FT
import torch.nn as nn
import numpy as np

tr_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=500)
te_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=100)

tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
te_dl = torch.utils.data.DataLoader(te_ds, batch_size=32, shuffle=False)

# net = N.SingleLayerAutoencoder(20, 15)
# net = N.MultilayerAE(20, [32, 15], beta=0, threshold=0.5)
# net = N.RecurrentSpikingAutoencoder(
#     input_size=20,
#     recurrent_size=16,
#     encoder_sizes=[32],
#     decoder_sizes=[32],
#     beta=0,
#     multiply_weights=True,
#     threshold=0.5
# )
# net = N.RecurrentStepAutoencoder(
#     20,
#     16,
#     [32],
#     [32],
#     multiply_weights=True
# )
net = N.RecurrentSingleLayerAutoencoder(20, 15, beta=0)

net.encoder.weight.data = net.encoder.weight.data * 10
net.decoder.weight.data = net.decoder.weight.data * 10

optimizer = optim.Adam(net.parameters(), lr=1e-3)

loss_fn = FT.van_rossum_loss
# loss_fn = lambda x, y: nn.MSELoss()(x, y) * 6

print(f"Number of layers: {net.get_total_layers()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MULTILAYER = False
LOAD = False

if MULTILAYER and LOAD:
    folder = "multilayer_20_32_15_32_20"
    layer_1_weights = np.load(f'pretrained_weights/{folder}/layer_1_weights.npy')
    layer_2_weights = np.load(f'pretrained_weights/{folder}/layer_2_weights.npy')
    layer_3_weights = np.load(f'pretrained_weights/{folder}/layer_3_weights.npy')
    layer_4_weights = np.load(f'pretrained_weights/{folder}/layer_4_weights.npy')

    print(f"Loaded encoder weights: {layer_1_weights.shape}")
    print(f"Expected encoder weights: {tuple(net.encoder_layers[0].weight.shape)}")

    print(f"Loaded decoder weights: {layer_2_weights.shape}")
    # print(f"Expected decoder weights: {tuple(net.decoder_layers[0].weight.shape)}")

    with torch.no_grad():
        net.encoder_layers[0].weight.copy_(torch.tensor(layer_1_weights.T, dtype=net.encoder_layers[0].weight.dtype, device=device))
        net.encoder_layers[1].weight.copy_(torch.tensor(layer_2_weights.T, dtype=net.encoder_layers[1].weight.dtype, device=device))
        net.decoder_layers[0].weight.copy_(torch.tensor(layer_3_weights.T, dtype=net.decoder_layers[0].weight.dtype, device=device))
        net.decoder_layers[1].weight.copy_(torch.tensor(layer_4_weights.T, dtype=net.decoder_layers[1].weight.dtype, device=device))
elif LOAD:
    folder = "single_autoencoder_20_15"
    layer_1_weights = np.load(f'pretrained_weights/{folder}/layer_1_weights.npy')
    layer_2_weights = np.load(f'pretrained_weights/{folder}/layer_2_weights.npy')

    print(f"Loaded encoder weights: {layer_1_weights.shape}")
    print(f"Loaded decoder weights: {layer_2_weights.shape}")

    with torch.no_grad():
        net.encoder.weight.copy_(torch.tensor(layer_1_weights.T, dtype=net.encoder.weight.dtype, device=device))
        net.decoder.weight.copy_(torch.tensor(layer_2_weights.T, dtype=net.decoder.weight.dtype, device=device))


FT.normal_train(net, tr_dl, te_dl, optimizer, loss_fn, device=device, num_epochs=100)
# FT.freeze_train(net, tr_dl, te_dl, optimizer, loss_fn, device=device, num_epochs=100, backwards=False, batchwise=True)