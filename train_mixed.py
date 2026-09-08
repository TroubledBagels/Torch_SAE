import utils.MixedNetwork as M
import MixedTrain as MT
import torch
import utils.GenData as GD
import torch.nn as nn

tr_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=500)
te_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=100)

tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
te_dl = torch.utils.data.DataLoader(te_ds, batch_size=32, shuffle=False)

recurrent_params = M.RecurrentParams(
    hidden_size=100,
    beta=0.9,
    threshold=1.0,
    spectral_radius=0.9,
    train_recurrent=True,
    self_connections=False
)

model = M.MixedNetwork(
    input_size=20,
    recurrent_params=recurrent_params,
    decoder_sizes=[64, 32],
    projection_train_mode='backprop'
)

model = MT.train_mixed_network(
    model=model,
    tr_dl=tr_dl,
    te_dl=te_dl,
    # criterion=MT.van_rossum_loss,
    criterion=nn.MSELoss(),
    device="cpu",
    num_epochs=20,
    decoder_lr=1e-3,
    sigma=0.05,
    population_size=20,
    cma_max_batches=5,
    plot=True,
    cma_generations_per_epoch=5
)