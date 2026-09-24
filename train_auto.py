import torch
import torch.optim as optim
import utils.Network as N
import FreezeTrain as FT
import torch.nn as nn
import numpy as np
import EDDataset as edd
import torchaudio
import torchvision.transforms as transforms
import pathlib
from utils.GradientViewer import export_gradient_viewer
from utils import DECOLLE as DC
from utils import SymmetricTrainer as ST

import warnings
warnings.filterwarnings("ignore")


def generate_gradient_view(model, test_loader, output_dir, loss_mode, device):
    sample, _ = next(iter(test_loader))

    if sample.shape[0] > 1 and len(sample.shape) > 2:
        sample = sample[0]
    elif sample.shape[0] == 1 and len(sample.shape) == 2:
        sample = sample[0]

    if loss_mode == "mse":
        paths = export_gradient_viewer(
            model=model,
            sample=sample,
            output_dir=output_dir,
            loss_mode="mse",
            loss_scale=6.0,
            device=device
        )

    elif loss_mode == "vr":
        paths = export_gradient_viewer(
            model=model,
            sample=sample,
            output_dir=output_dir,
            loss_mode="vr",
            tau=10.0,
            device=device
        )

    else:
        raise ValueError(f"Unknown gradient viewer loss mode: {loss_mode}")

    return paths


DELAY = 0
PREFIX = "" if DELAY == 0 else f"DELAY_{DELAY}_"

home = pathlib.Path("~").expanduser()
local_dir = home / "data" / "URBAN-SED"

a_transform = transforms.Compose([
    torchaudio.transforms.Resample(44100, 16000),
    edd.UnsqueezeTransform(dim=0),
    edd.ToSpikeTransform(num_channels=10),
    edd.SqueezeTransform(dim=0),
    edd.SqueezeTransform(dim=0),
    edd.ReshapeTransform(1, 0)
])

# tr_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=500)
# te_ds = GD.SinWaveDS(channels=20, timesteps=1000, wavelength=100, num_samples=100)
#
tr_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TRAIN, transform=a_transform)
te_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TEST, transform=a_transform)

if isinstance(tr_ds, edd.URBANDataset):
    PREFIX += "urban_"

tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
te_dl = torch.utils.data.DataLoader(te_ds, batch_size=32, shuffle=False)

# net = N.SingleLayerAutoencoder(20, 15)
# net = N.SingleLayerAutoencoderTrainable(20, 15)
net = N.MultilayerAETrainable(20, [16, 12])
# net = N.UNetSpikingAutoencoder(
#     input_size=20,
#     hidden_sizes=[32, 15],
#     beta=0.9,
#     threshold=1.0,
#     learn_beta=True,
#     learn_threshold=True,
#     concat_mode="spatial"
# )
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
# net = N.RecurrentSingleLayerAutoencoder(20, 15, beta=0)

# net.encoder.weight.data = net.encoder.weight.data * 10xkn
# net.decoder.weight.data = net.decoder.weight.data * 10

# net.load_state_dict(torch.load(f"output_models/{PREFIX + net.name}.pth"))
# net.load_state_dict(torch.load("output_models/symmetric_urban__20_16_12_16_20MultilayerAETrainable.pth"))

TRAIN = True
MULTILAYER = True
LOAD = True
DECOLLE = False
SYMMETRIC = True

if SYMMETRIC:
    PREFIX = "symmetric_" + PREFIX
elif DECOLLE:
    PREFIX = "decolle_" + PREFIX

if DECOLLE and SYMMETRIC:
    raise ValueError("DECOLLE and SYMMETRIC are not compatible")

# if SYMMETRIC: PREFIX += "sym_"
# if DECOLLE: PREFIX += "decolle_"

optimizer = optim.Adam(net.parameters(), lr=1e-3)

loss_fn = lambda x, y: FT.van_rossum_loss_count(
    x,
    y,
    tau=10.0,
    count_weight=0.0
)
# loss_fn = lambda x, y: nn.MSELoss()(x, y) * 6

print(f"Number of layers: {net.get_total_layers()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if MULTILAYER and LOAD:
    dimensions = "20_16_12_16_20"
    folder = f"{PREFIX}multilayer_{dimensions}"
    PREFIX += f"{dimensions}_"
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
    folder = f"{PREFIX}single_autoencoder_20_15"
    layer_1_weights = np.load(f'pretrained_weights/{folder}/layer_1_weights.npy')
    layer_2_weights = np.load(f'pretrained_weights/{folder}/layer_2_weights.npy')

    print(f"Loaded encoder weights: {layer_1_weights.shape}")
    print(f"Loaded decoder weights: {layer_2_weights.shape}")

    with torch.no_grad():
        net.encoder.weight.copy_(torch.tensor(layer_1_weights.T, dtype=net.encoder.weight.dtype, device=device))
        net.decoder.weight.copy_(torch.tensor(layer_2_weights.T, dtype=net.decoder.weight.dtype, device=device))

training_interrupted = False

if TRAIN:
    if SYMMETRIC:
        trainer_kwargs = {}

        if not hasattr(net, "encoder_layers"):
            trainer_kwargs["encoder_layers"] = [net.encoder]
            trainer_kwargs["decoder_layers"] = [net.decoder]

            if hasattr(net, "lif1") and hasattr(net, "lif2"):
                trainer_kwargs["neuron_layers"] = [net.lif1, net.lif2]
            elif hasattr(net, "rlif1") and hasattr(net, "rlif2"):
                trainer_kwargs["neuron_layers"] = [net.rlif1, net.rlif2]
            else:
                raise AttributeError("Single-layer network needs lif1/lif2 or rlif1/rlif2 neuron attributes for SymmetricTrainer.")

        symmetric_trainer = ST.ProgressiveSpikingAutoencoderTrainer(
            model=net,
            loss_fn=loss_fn,
            device=device,
            lr=1e-3,
            patience=8,
            patience_min_delta=0.0001,
            **trainer_kwargs
        )

        inputs, _ = next(iter(tr_dl))
        inputs = inputs.to(device)

        net.eval()

        try:
            best_model_sd, results = symmetric_trainer.fit(
                train_loader=tr_dl,
                test_loader=te_dl,
                epochs_per_stage=300,
                fine_tune_epochs=100,
                plot=True,
                plot_every=1,
                tau=10.0
            )

            net.load_state_dict(best_model_sd)

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}.pth")

        except KeyboardInterrupt:
            training_interrupted = True

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}_interrupted.pth")

            print("\nTraining interrupted.")
            print("Using current network state for gradient viewer.")

    elif DECOLLE:
        NEARBY_TOLERANCE = 1
        FBETA_BETA = 2.0

        sites = DC.make_multilayer_ae_sites(net)

        output_loss = DC.make_spike_output_loss(
            exact_weight=1.0,
            fbeta_weight=1.0,
            nearby_fbeta_weight=1.0,
            count_weight=3.0,
            nearby_tolerance=NEARBY_TOLERANCE,
            beta=FBETA_BETA
        )

        losses = DC.make_decolle_losses(
            sites,
            hidden_loss=nn.MSELoss(),
            output_loss=output_loss
        )

        decolle = DC.DECOLLETrainer(
            model=net,
            sites=sites,
            losses=losses,
            optimizer=optimizer,
            train_readouts=False,
            detach_between_sites=True,
            detach_states=True
        )

        try:
            best_model_sd, results = decolle.fit(
                tr_dl=tr_dl,
                te_dl=te_dl,
                device=device,
                num_epochs=100,
                delay=DELAY,
                plot=True,
                nearby_tolerance=NEARBY_TOLERANCE,
                fbeta_beta=FBETA_BETA
            )

            net.load_state_dict(best_model_sd)

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}.pth")

        except KeyboardInterrupt:
            training_interrupted = True

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}_interrupted.pth")

            print("\nTraining interrupted.")
            print("Using current network state for gradient viewer.")

    else:
        try:
            best_model_sd = FT.normal_train(net, tr_dl, te_dl, optimizer, loss_fn, device, 100, DELAY)

            net.load_state_dict(best_model_sd)

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}.pth")

        except KeyboardInterrupt:
            training_interrupted = True

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}_interrupted.pth")

            print("\nTraining interrupted.")
            print("Using current network state for gradient viewer.")

if not TRAIN and SYMMETRIC:
    print("Executing one symmetric training test loop...")

    trainer_kwargs = {}
    PREFIX = "symmetric_" + PREFIX

    if not hasattr(net, "encoder_layers"):
        trainer_kwargs["encoder_layers"] = [net.encoder]
        trainer_kwargs["decoder_layers"] = [net.decoder]

        if hasattr(net, "lif1") and hasattr(net, "lif2"):
            trainer_kwargs["neuron_layers"] = [net.lif1, net.lif2]
        elif hasattr(net, "rlif1") and hasattr(net, "rlif2"):
            trainer_kwargs["neuron_layers"] = [net.rlif1, net.rlif2]
        else:
            raise AttributeError(
                "Single-layer network needs lif1/lif2 or rlif1/rlif2 neuron attributes for SymmetricTrainer.")

    symmetric_trainer = ST.ProgressiveSpikingAutoencoderTrainer(
        model=net,
        loss_fn=loss_fn,
        device=device,
        lr=1e-3,
        **trainer_kwargs
    )

    metrics = symmetric_trainer.test(test_loader=te_dl, plot=True)

out_dir = f"gradient_trace/{PREFIX + net.name}/"

paths = generate_gradient_view(
    model=net,
    test_loader=te_dl,
    output_dir=out_dir,
    loss_mode="vr",
    device=device
)

print(f"Gradient viewer written to: {paths['html']}")