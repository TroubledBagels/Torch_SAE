"""Tree-style progressive training for symmetric spiking autoencoders.

This module is a drop-in companion to ``utils/SymmetricTrainer.py``.  It keeps
that trainer's public ``fit``/``test`` API, metrics, plotting, patience handling,
and fine-tuning code, but changes the progressive schedule.

For a network

    A -> B -> C -> D -> C -> B -> A

training is performed as

    Stage 1: A -> B -> A
             train A->B and B->A

    Stage 2: A -> B -> C -> B -> A
             freeze A->B and B->A
             train B->C and C->B

    Stage 3: A -> B -> C -> D -> C -> B -> A
             freeze the four retained outer layers
             train C->D and D->C

The retained layers are the *actual model layers*, not temporary reconstruction
heads, so the weights learned by one stage are directly reused by the next.
"""

from dataclasses import dataclass

import torch

try:
    # Normal use when this file lives beside SymmetricTrainer.py in ``utils``.
    from .SymmetricTrainer import (
        ProgressiveSpikingAutoencoderTrainer as _BaseProgressiveTrainer,
    )
except ImportError:  # pragma: no cover - useful when running this file directly
    from SymmetricTrainer import (
        ProgressiveSpikingAutoencoderTrainer as _BaseProgressiveTrainer,
    )


@dataclass
class TreeStage:
    """One nested symmetric subnetwork used by tree training."""

    name: str
    structure: list
    active_indices: list
    train_indices: list

    # These fields are intentionally kept compatible with SymmetricTrainer's
    # helper methods.  Tree training never uses a temporary readout because the
    # matching real decoder layer is present from the first stage onward.
    target_mode: str = "global_input"
    use_temp_readout: bool = False
    temp_input_size: int | None = None
    temp_output_size: int | None = None


class TreeSpikingAutoencoderTrainer(_BaseProgressiveTrainer):
    """Progressively train nested symmetric autoencoders from outside inward.

    The model interface is the same as ``ProgressiveSpikingAutoencoderTrainer``:

    ``model.encoder_layers``
        ``ModuleList`` containing encoder synaptic layers from input to centre.

    ``model.decoder_layers``
        ``ModuleList`` containing decoder synaptic layers from centre to output.

    ``model.lif_layers``
        ``ModuleList`` containing one spiking-neuron module per synaptic layer,
        in the same order as ``encoder_layers + decoder_layers``.

    For encoder depth ``N``, stage ``k`` activates the first ``k`` encoders and
    the last ``k`` decoders.  Only the newly introduced mirrored pair is
    trainable.  Previously learned outer pairs are frozen but remain in the
    forward path so their weights are genuinely reused.

    All constructor arguments, ``fit(...)``, ``test(...)``, early stopping,
    plotting, tolerant F1 metrics, and optional final fine-tuning are inherited
    from the existing symmetric trainer.
    """

    def __init__(self, *args, **kwargs):
        # SymmetricTrainer has an option that copies a temporary first-stage
        # decoder into the final decoder.  Tree training does not need that:
        # B->A is the real final decoder and is trained in stage 1 itself.
        requested_legacy_reuse = bool(
            kwargs.get("reuse_initial_decoder_for_final_layer", False)
        )
        if "reuse_initial_decoder_for_final_layer" in kwargs:
            kwargs["reuse_initial_decoder_for_final_layer"] = False

        super().__init__(*args, **kwargs)
        self.requested_legacy_decoder_reuse = requested_legacy_reuse
        self.reuse_initial_decoder_for_final_layer = False

    def _validate_architecture(self):
        """Validate both the full network and every nested tree subnetwork."""
        super()._validate_architecture()

        total_layers = len(self.layers)

        for encoder_index in range(self.encoder_depth):
            decoder_index = total_layers - 1 - encoder_index
            encoder = self.layers[encoder_index]
            decoder = self.layers[decoder_index]

            # A nested stage connects encoder i directly to its mirrored
            # decoder, so those interfaces must be exact reverses.
            if decoder.in_features != encoder.out_features:
                raise ValueError(
                    "Tree training requires mirrored encoder/decoder widths. "
                    f"Encoder layer {encoder_index} is "
                    f"{encoder.in_features}->{encoder.out_features}, but its "
                    f"mirrored decoder layer {decoder_index} is "
                    f"{decoder.in_features}->{decoder.out_features}. "
                    f"Expected decoder input size {encoder.out_features}."
                )

            if decoder.out_features != encoder.in_features:
                raise ValueError(
                    "Tree training requires mirrored encoder/decoder widths. "
                    f"Encoder layer {encoder_index} is "
                    f"{encoder.in_features}->{encoder.out_features}, but its "
                    f"mirrored decoder layer {decoder_index} is "
                    f"{decoder.in_features}->{decoder.out_features}. "
                    f"Expected decoder output size {encoder.in_features}."
                )

    def _build_stages(self):
        """Build A-B-A, A-B-C-B-A, ... nested stages."""
        stages = []
        total_layers = len(self.layers)

        for depth in range(1, self.encoder_depth + 1):
            # Example for six layers and depth=2:
            # encoder side -> [0, 1]
            # decoder side -> [4, 5]
            # active path  -> [0, 1, 4, 5]
            encoder_indices = list(range(depth))
            decoder_indices = list(range(total_layers - depth, total_layers))
            active_indices = encoder_indices + decoder_indices

            new_encoder_index = depth - 1
            new_decoder_index = total_layers - depth
            train_indices = [new_encoder_index, new_decoder_index]

            structure = [self.layers[active_indices[0]].in_features]
            structure.extend(
                self.layers[index].out_features for index in active_indices
            )

            stages.append(
                TreeStage(
                    name=" -> ".join(str(width) for width in structure),
                    structure=structure,
                    active_indices=active_indices,
                    train_indices=train_indices,
                )
            )

        return stages

    def describe_schedule(self):
        """Print the nested subnetworks and newly trained mirrored pair."""
        print("\nTREE TRAINING SCHEDULE")
        for stage_number, stage in enumerate(self.stages, start=1):
            active_text = ", ".join(str(i) for i in stage.active_indices)
            train_text = ", ".join(str(i) for i in stage.train_indices)
            print(
                f"Stage {stage_number}: {stage.name} | "
                f"active actual layers: [{active_text}] | "
                f"train actual layers: [{train_text}]"
            )

        if self.requested_legacy_decoder_reuse:
            print(
                "Note: reuse_initial_decoder_for_final_layer is unnecessary in "
                "TreeTraining because the real outer decoder is trained and "
                "retained from stage 1."
            )

    def _is_initial_i_m_i_stage(self, stage, stage_number):
        """Disable the base trainer's temporary-decoder save path."""
        return False

    def _is_final_decoder_stage(self, stage):
        """Disable the base trainer's temporary-decoder restore path."""
        return False

    def _stage_forward(
        self,
        inputs,
        stage,
        temp_readout=None,
        record_spikes=False,
    ):
        """Run one nested tree stage using only its active real model layers.

        Frozen encoder layers *before* the new trainable pair are evaluated
        under ``torch.no_grad`` and detached because no gradient needs to pass
        through them.

        Frozen decoder layers *after* the new trainable pair are deliberately
        evaluated with autograd enabled.  Their parameters still have
        ``requires_grad=False``, but the computation graph must pass through
        them so reconstruction loss can reach the newly added inner pair.
        """
        if temp_readout is not None:
            raise RuntimeError(
                "TreeTraining does not use temporary readouts; each stage uses "
                "the model's real mirrored decoder layer."
            )

        timesteps = inputs.size(2)
        active_indices = list(stage.active_indices)
        train_indices = set(stage.train_indices)

        if not active_indices:
            raise RuntimeError("Tree stage has no active layers")

        train_positions = [
            position
            for position, layer_index in enumerate(active_indices)
            if layer_index in train_indices
        ]
        if not train_positions:
            raise RuntimeError("Tree stage has no trainable layers")

        first_train_position = min(train_positions)

        states = {
            index: self._init_neuron_state(self.neurons[index])
            for index in active_indices
        }

        outputs = []
        targets = []
        recorded = {}

        if record_spikes:
            for index in stage.train_indices:
                layer = self.layers[index]
                recorded[
                    f"Layer {index} ({layer.in_features}->{layer.out_features})"
                ] = []

        for timestep in range(timesteps):
            current = inputs[:, :, timestep]
            target_t = current.detach()

            for position, index in enumerate(active_indices):
                layer = self.layers[index]
                neuron = self.neurons[index]

                if position < first_train_position:
                    # These are retained outer encoder layers.  Nothing before
                    # them is trainable in this stage, so detaching is safe and
                    # avoids storing an unnecessary graph.
                    with torch.no_grad():
                        synaptic = layer(current)
                        current, states[index] = self._step_neuron(
                            neuron,
                            synaptic,
                            states[index],
                        )
                    current = current.detach()
                else:
                    # This includes both newly trainable layers and retained
                    # decoder layers after them.  Frozen decoder parameters do
                    # not receive gradients, but their operations remain in the
                    # graph so gradients can reach the new inner pair.
                    synaptic = layer(current)
                    current, states[index] = self._step_neuron(
                        neuron,
                        synaptic,
                        states[index],
                    )

                if record_spikes and index in train_indices:
                    key = (
                        f"Layer {index} "
                        f"({layer.in_features}->{layer.out_features})"
                    )
                    recorded[key].append(current.detach())

            outputs.append(current)
            targets.append(target_t)

        outputs = torch.stack(outputs, dim=2)
        targets = torch.stack(targets, dim=2)

        if record_spikes:
            recorded = {
                name: torch.stack(values, dim=2)
                for name, values in recorded.items()
            }

        return outputs, targets, recorded


# Drop-in name: existing code can switch only the imported module, e.g.
#
#     from utils import TreeTraining as TT
#     trainer = TT.ProgressiveSpikingAutoencoderTrainer(...)
#
# and keep the rest of the training call unchanged.
# ProgressiveSpikingAutoencoderTrainer = TreeSpikingAutoencoderTrainer
#
#
# __all__ = [
#     "TreeStage",
#     "TreeSpikingAutoencoderTrainer",
#     "ProgressiveSpikingAutoencoderTrainer",
# ]