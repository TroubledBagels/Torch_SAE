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
from utils import TreeTrainer as TT
from utils import LatentAnalysis as LA

import warnings
warnings.filterwarnings("ignore")


import re
import numpy as np
import torch
from pathlib import Path


def load_pretrained_layers(net, folder, device):
    """
    Load pretrained layer_N_weights.npy files into a multilayer autoencoder.

    Saved layers are assumed to be ordered along the forward path.

    Example
    -------
    Saved network:
        20 -> 16 -> 12 -> 16 -> 20

    Current network:
        20 -> 16 -> 12 -> 8 -> 12 -> 16 -> 20

    The loader will map:

        saved layer 1 -> encoder[0]   20 -> 16
        saved layer 2 -> encoder[1]   16 -> 12
        saved layer 3 -> decoder[1]   12 -> 16
        saved layer 4 -> decoder[2]   16 -> 20

    The new:
        encoder[2]  12 -> 8
        decoder[0]   8 -> 12

    remain at their normal initialization.
    """

    folder = Path(folder)

    # ----------------------------------------------------------
    # 1. Find all layer_N_weights.npy files
    # ----------------------------------------------------------
    weight_files = list(folder.glob("layer_*_weights.npy"))

    if not weight_files:
        raise FileNotFoundError(
            f"No layer_*_weights.npy files found in:\n{folder}"
        )

    # Sort numerically:
    # layer_2_weights.npy before layer_10_weights.npy
    def get_layer_number(path):
        match = re.search(r"layer_(\d+)_weights\.npy$", path.name)

        if match is None:
            raise ValueError(
                f"Could not extract layer number from {path.name}"
            )

        return int(match.group(1))

    weight_files = sorted(weight_files, key=get_layer_number)

    print(f"\nFound {len(weight_files)} pretrained layers in:")
    print(f"  {folder}")

    for path in weight_files:
        print(f"  {path.name}")

    # ----------------------------------------------------------
    # 2. Get model layers in forward-pass order
    # ----------------------------------------------------------
    if hasattr(net, "encoder_layers") and hasattr(net, "decoder_layers"):
        target_layers = (
            list(net.encoder_layers)
            + list(net.decoder_layers)
        )

        target_names = (
            [f"encoder[{i}]" for i in range(len(net.encoder_layers))]
            + [f"decoder[{i}]" for i in range(len(net.decoder_layers))]
        )

    elif hasattr(net, "encoder") and hasattr(net, "decoder"):
        target_layers = [
            net.encoder,
            net.decoder,
        ]

        target_names = [
            "encoder",
            "decoder",
        ]

    else:
        raise AttributeError(
            "Network must contain either:\n"
            "  encoder_layers / decoder_layers\n"
            "or:\n"
            "  encoder / decoder"
        )

    print(f"\nCurrent network has {len(target_layers)} trainable layers:")

    for name, layer in zip(target_names, target_layers):
        print(
            f"  {name:12s} "
            f"weight shape = {tuple(layer.weight.shape)}"
        )

    # ----------------------------------------------------------
    # 3. Load arrays
    # ----------------------------------------------------------
    saved_weights = []

    print("\nSaved weight shapes:")

    for path in weight_files:
        weights = np.load(path)
        saved_weights.append((path, weights))

        print(
            f"  {path.name:25s} "
            f"shape={weights.shape}"
        )

    # ----------------------------------------------------------
    # 4. Match each saved layer to a model layer
    #
    # Search forward through the model. This allows a shallower
    # pretrained network to be loaded into a deeper one.
    # ----------------------------------------------------------
    next_target_index = 0
    loaded_target_indices = []

    with torch.no_grad():

        for path, weights in saved_weights:

            matched = False

            for target_index in range(
                next_target_index,
                len(target_layers)
            ):
                target_layer = target_layers[target_index]

                target_shape = tuple(target_layer.weight.shape)
                saved_shape = tuple(weights.shape)
                transposed_shape = tuple(weights.T.shape)

                # ----------------------------------------------
                # Determine orientation automatically
                # ----------------------------------------------
                if saved_shape == target_shape:
                    tensor = torch.as_tensor(
                        weights,
                        dtype=target_layer.weight.dtype,
                        device=target_layer.weight.device,
                    )

                    orientation = "direct"

                elif transposed_shape == target_shape:
                    tensor = torch.as_tensor(
                        weights.T,
                        dtype=target_layer.weight.dtype,
                        device=target_layer.weight.device,
                    )

                    orientation = "transposed"

                else:
                    # This target doesn't match.
                    # Try the next network layer.
                    continue

                # ----------------------------------------------
                # Copy weights
                # ----------------------------------------------
                target_layer.weight.copy_(tensor)

                print(
                    f"\nLoaded {path.name}"
                    f"\n  saved shape:  {saved_shape}"
                    f"\n  -> {target_names[target_index]}"
                    f"\n  target shape: {target_shape}"
                    f"\n  orientation:  {orientation}"
                )

                loaded_target_indices.append(target_index)

                # Any following saved layer must occur later in
                # the forward path.
                next_target_index = target_index + 1

                matched = True
                break

            if not matched:
                remaining = [
                    (
                        target_names[i],
                        tuple(target_layers[i].weight.shape)
                    )
                    for i in range(
                        next_target_index,
                        len(target_layers)
                    )
                ]

                raise RuntimeError(
                    f"\nCould not match pretrained weight:\n"
                    f"  file:  {path}\n"
                    f"  shape: {weights.shape}\n"
                    f"  transposed shape: {weights.T.shape}\n\n"
                    f"Remaining model layers:\n"
                    + "\n".join(
                        f"  {name}: {shape}"
                        for name, shape in remaining
                    )
                )

    # ----------------------------------------------------------
    # 5. Report layers that remain newly initialized
    # ----------------------------------------------------------
    unloaded = [
        i
        for i in range(len(target_layers))
        if i not in loaded_target_indices
    ]

    print("\n========================================")
    print("Pretrained weight loading complete")
    print("========================================")
    print(
        f"Loaded {len(loaded_target_indices)} / "
        f"{len(target_layers)} model layers."
    )

    if unloaded:
        print("\nLayers left at their initial values:")

        for i in unloaded:
            print(
                f"  {target_names[i]:12s} "
                f"{tuple(target_layers[i].weight.shape)}"
            )
    else:
        print("\nAll model layers were loaded.")

    print("========================================\n")

    return loaded_target_indices

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
if device == torch.device('cpu'):
    tr_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TRAIN, transform=a_transform)
    te_ds = edd.URBANDataset(local_dir, split=edd.DatasetSplit.TEST, transform=a_transform)
elif device == torch.device('cuda:0'):
    tr_ds = edd.URBANDatasetGPU(local_dir, split=edd.DatasetSplit.TRAIN, transform=a_transform, device=device)
    te_ds = edd.URBANDatasetGPU(local_dir, split=edd.DatasetSplit.TEST, transform=a_transform, device=device)
else:
    raise ValueError(f"Unsupported device: {device}")

if isinstance(tr_ds, edd.URBANDataset) or isinstance(te_ds, edd.URBANDatasetGPU):
    PREFIX += "urban_"

tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=32, shuffle=True)
te_dl = torch.utils.data.DataLoader(te_ds, batch_size=32, shuffle=False)

# net = N.SingleLayerAutoencoder(20, 15)
# net = N.SingleLayerAutoencoderTrainable(20, 8)
# net = N.MultilayerAETrainable(20, [16, 12, 8])
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
net.load_state_dict(torch.load("output_models/symmetric_urban__20_16_12_16_20MultilayerAETrainable.pth"))

TRAIN = True
MULTILAYER = True
LOAD = True
DECOLLE = False
SYMMETRIC = True
TREE = False
ONLY_LATENT = False

if SYMMETRIC:
    PREFIX = "symmetric_" + PREFIX
elif DECOLLE:
    PREFIX = "decolle_" + PREFIX
elif TREE:
    PREFIX = "tree_" + PREFIX

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

print(f"Using device: {device}")

if LOAD:
    if MULTILAYER:
        dimensions = "20_16_12_8_12_16_20"
        folder = f"pretrained_weights/{PREFIX}multilayer_{dimensions}"

        load_pretrained_layers(
            net=net,
            folder=folder,
            device=device,
        )

        PREFIX += f"{dimensions}_"

    else:
        folder = f"pretrained_weights/{PREFIX}single_autoencoder_20_8"

        load_pretrained_layers(
            net=net,
            folder=folder,
            device=device,
        )
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
            reuse_initial_decoder_for_final_layer=True,
            # scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
            # scheduler_kwargs={"T_max": 100}
            scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={"patience": 5, "factor": 0.5, "min_lr": 1e-6},
            # scheduler_class=torch.optim.lr_scheduler.StepLR,
            # scheduler_kwargs={"step_size": 1, "gamma": 0.5},
            **trainer_kwargs
        )

        PREFIX += f"{symmetric_trainer.scheduler_class.__name__}_"

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
    elif TREE:
        trainer_kwargs = {}

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

        symmetric_trainer = TT.TreeSpikingAutoencoderTrainer(
            model=net,
            loss_fn=loss_fn,
            device=device,
            lr=1e-3,
            patience=8,
            patience_min_delta=0.0001,
            reuse_initial_decoder_for_final_layer=True,
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
            best_model_sd = FT.normal_train(net, tr_dl, te_dl, optimizer, loss_fn, device, 300, DELAY)

            net.load_state_dict(best_model_sd)

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}.pth")

        except KeyboardInterrupt:
            training_interrupted = True

            torch.save(net.state_dict(), f"output_models/{PREFIX + net.name}_interrupted.pth")

            print("\nTraining interrupted.")
            print("Using current network state for gradient viewer.")

if not TRAIN and SYMMETRIC and not ONLY_LATENT:
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

if ONLY_LATENT:
    app, bundle = LA.main(
        model=net,
        dataset=te_ds,
        samples=10,
        seed=1337
    )

out_dir = f"gradient_trace/{PREFIX + net.name}/"

paths = generate_gradient_view(
    model=net,
    test_loader=te_dl,
    output_dir=out_dir,
    loss_mode="vr",
    device=device
)

print(f"Gradient viewer written to: {paths['html']}")