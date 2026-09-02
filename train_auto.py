import torch
import torch.optim as optim
import utils.Network as N
import utils.GenData as GD
import FreezeTrain as FT

tr_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=500)
te_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=100)

tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
te_dl = torch.utils.data.DataLoader(te_ds, batch_size=32, shuffle=False)

# net = N.SingleLayerAutoencoder(20, 15)
# net = N.MultilayerAutoencoder(20, [15, 10, 15])
net = N.RecurrentSpikingAutoencoder(
    input_size=20,
    recurrent_size=16,
    encoder_sizes=[64, 32],
    decoder_sizes=[32, 64]
)
optimizer = optim.Adam(net.parameters(), lr=5e-3)

loss_fn = FT.van_rossum_loss

print(f"Number of layers: {net.get_total_layers()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FT.normal_train(net, tr_dl, te_dl, optimizer, loss_fn, device='cuda', num_epochs=20)
FT.freeze_train(net, tr_dl, te_dl, optimizer, loss_fn, device=device, num_epochs=40, backwards=False, batchwise=False)