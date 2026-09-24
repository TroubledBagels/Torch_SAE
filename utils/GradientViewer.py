import csv
import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Structure discovery / specification
# -----------------------------------------------------------------------------

def _get_module(model, path):
    module = model
    if path == "":
        return module

    for part in path.split("."):
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            module = module[int(part)]
        elif isinstance(module, nn.ModuleDict):
            module = module[part]
        else:
            module = getattr(module, part)

    return module


def _auto_linear_paths(model):
    """
    Find the explicit Linear layers that define the visible network structure.

    Direct children of the model and Linear layers inside ModuleList/Sequential/
    ModuleDict containers are included. Linear modules internal to another custom
    module (for example an RLeaky recurrent matrix) are deliberately excluded.
    Their effect is still present in the BPTT gradients, but they are not drawn as
    an extra feed-forward stage.
    """
    paths = []

    for path, module in model.named_modules():
        if path == "" or not isinstance(module, nn.Linear):
            continue

        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        parent = _get_module(model, parent_path)

        if parent is model or isinstance(parent, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
            paths.append(path)

    return paths


def _module_exists(model, path):
    try:
        _get_module(model, path)
        return True
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def _is_neuron_module(module):
    return hasattr(module, "beta") or hasattr(module, "threshold")


def _auto_neuron_path(model, linear_path):
    """Best-effort mapping from a visible Linear layer to its following neuron."""
    candidates = []

    if linear_path.startswith("encoder_layers."):
        index = linear_path.split(".")[-1]
        candidates += [f"encoder_lifs.{index}", f"encoder_neurons.{index}"]

    if linear_path.startswith("decoder_layers."):
        index = linear_path.split(".")[-1]
        candidates += [f"decoder_lifs.{index}", f"decoder_neurons.{index}"]

    if linear_path == "recurrent_linear":
        candidates += ["recurrent_lif", "rlif", "recurrent_neuron"]

    if linear_path == "output_layer":
        candidates += ["output_lif", "output_neuron", "lif_out"]

    if linear_path == "encoder":
        candidates += ["rlif1", "lif1", "encoder_lif", "encoder_neuron"]

    if linear_path == "decoder":
        candidates += ["rlif2", "lif2", "decoder_lif", "decoder_neuron", "output_lif"]

    candidates += [
        linear_path + "_lif",
        linear_path.replace("_linear", "_lif"),
        linear_path.replace("_layer", "_lif"),
        linear_path.replace("layers", "lifs"),
    ]

    seen = set()
    for path in candidates:
        if not path or path == linear_path or path in seen:
            continue
        seen.add(path)
        if _module_exists(model, path):
            module = _get_module(model, path)
            if _is_neuron_module(module):
                return path

    return None


def _expand_node_value(value, size):
    """Convert scalar/per-neuron beta or threshold values to a JSON-friendly list."""
    if value is None:
        return [None] * size

    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        try:
            values = [float(value)]
        except (TypeError, ValueError):
            return [None] * size

    values = [float(v) for v in values]

    if len(values) == 1:
        return values * size
    if len(values) == size:
        return values

    # If the neuron module stores an unexpected broadcast shape, use the first
    # value rather than failing the whole viewer.
    return [values[0]] * size


def _attach_neuron_metadata(model, layer_specs, stage_data):
    """
    Attach decay (beta) and threshold values to each destination neuron stage.

    A structure item may explicitly specify:
        {"module": "encoder", "neuron": "rlif1"}

    or override values directly with:
        {"module": "encoder", "decay": 0.9, "threshold": 0.5}

    Otherwise the viewer tries common snnTorch naming patterns automatically.
    """
    stage_data[0]["neuron_module"] = None
    stage_data[0]["decay"] = [None] * stage_data[0]["activation"].size(0)
    stage_data[0]["threshold"] = [None] * stage_data[0]["activation"].size(0)

    for edge_index, item in enumerate(layer_specs):
        stage_index = edge_index + 1
        size = stage_data[stage_index]["activation"].size(0)

        neuron_path = item.get("neuron")
        if neuron_path is None:
            neuron_path = _auto_neuron_path(model, item["module"])

        neuron = _get_module(model, neuron_path) if neuron_path is not None else None

        decay = item.get("decay", getattr(neuron, "beta", None) if neuron is not None else None)
        threshold = item.get("threshold", getattr(neuron, "threshold", None) if neuron is not None else None)

        stage_data[stage_index]["neuron_module"] = neuron_path
        stage_data[stage_index]["decay"] = _expand_node_value(decay, size)
        stage_data[stage_index]["threshold"] = _expand_node_value(threshold, size)

    return stage_data


def detect_gradient_viewer_structure(model):
    """
    Return an automatically detected viewer structure.

    The returned dictionary can be copied into the model as
    ``model.gradient_viewer_structure`` and edited if desired.
    """
    paths = _auto_linear_paths(model)

    if not paths:
        raise ValueError(
            "Could not auto-detect any explicit nn.Linear layers. "
            "Specify model.gradient_viewer_structure manually."
        )

    # U-Net style models need graph-aware handling because skip concatenation
    # breaks the simple-chain assumption used by the original viewer.
    if (hasattr(model, "encoder_layers") and hasattr(model, "decoder_layers")
            and hasattr(model, "lif_layers") and hasattr(model, "concat_mode")):
        layers = []
        lif_idx = 0
        for i in range(len(model.encoder_layers)):
            layers.append({"module": f"encoder_layers.{i}", "neuron": f"lif_layers.{lif_idx}", "name": f"Encoder {i}"})
            lif_idx += 1
        for i in range(len(model.decoder_layers)):
            layers.append({"module": f"decoder_layers.{i}", "neuron": f"lif_layers.{lif_idx}", "name": f"Decoder {i}"})
            lif_idx += 1
        return {
            "layers": layers, "stage_names": None,
            "latent_stage": 2 * len(model.encoder_layers) - 1,
            "spike_threshold": 0.5, "forward_kwargs": {}, "output_index": 0,
            "graph_mode": "unet", "concat_mode": model.concat_mode,
        }

    return {
        "layers": [{"module": path, "name": path} for path in paths],
        "stage_names": None,
        "latent_stage": None,
        "spike_threshold": 0.5,
        "forward_kwargs": {},
        "output_index": 0,
        "graph_mode": "chain",
    }


def _normalise_structure(model, structure=None):
    """
    Structure priority:
        1. Explicit ``structure=...`` argument.
        2. ``model.get_gradient_viewer_structure()``.
        3. ``model.get_gradient_viewer_spec()``.
        4. ``model.gradient_viewer_structure``.
        5. ``model.gradient_viewer_spec``.
        6. Automatic detection.

    Minimal manual form:

        {
            "layers": [
                {"module": "encoder", "neuron": "rlif1"},
                {"module": "decoder", "neuron": "rlif2"},
            ],
            "stage_names": ["Input", "Latent", "Output"],
            "latent_stage": 1,
        }

    ``layers`` may also contain dictionaries:

        {"module": "encoder_layers.0", "name": "Encoder 0", "neuron": "encoder_lifs.0"}

    ``neuron`` is optional. If omitted, common snnTorch naming patterns such as
    ``encoder_layers.0 -> encoder_lifs.0`` and ``encoder -> rlif1`` are detected
    automatically. ``decay`` and ``threshold`` can also be provided directly.
    """
    if structure is None and hasattr(model, "get_gradient_viewer_structure"):
        structure = model.get_gradient_viewer_structure()

    if structure is None and hasattr(model, "get_gradient_viewer_spec"):
        structure = model.get_gradient_viewer_spec()

    if structure is None and hasattr(model, "gradient_viewer_structure"):
        structure = model.gradient_viewer_structure

    if structure is None and hasattr(model, "gradient_viewer_spec"):
        structure = model.gradient_viewer_spec

    if structure is None:
        structure = detect_gradient_viewer_structure(model)

    if isinstance(structure, (list, tuple)):
        structure = {"layers": list(structure)}

    if not isinstance(structure, dict):
        raise TypeError("structure must be a dict, list of module paths, or None")

    structure = dict(structure)
    raw_layers = structure.get("layers")

    if not raw_layers:
        raise ValueError("gradient viewer structure must contain a non-empty 'layers' list")

    layers = []
    for item in raw_layers:
        if isinstance(item, str):
            item = {"module": item, "name": item}
        elif isinstance(item, dict):
            item = dict(item)
        else:
            raise TypeError("Each structure layer must be a module path string or dictionary")

        if "module" not in item:
            raise ValueError("Each structure layer dictionary needs a 'module' path")

        item.setdefault("name", item["module"])
        module = _get_module(model, item["module"])

        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Viewer edge module '{item['module']}' is {type(module).__name__}, not nn.Linear. "
                "The current edge-gradient calculation supports nn.Linear modules."
            )

        layers.append(item)

    structure["layers"] = layers
    structure.setdefault("stage_names", None)
    structure.setdefault("latent_stage", None)
    structure.setdefault("spike_threshold", 0.5)
    structure.setdefault("forward_kwargs", {})
    structure.setdefault("output_index", 0)
    structure.setdefault("graph_mode", "chain")
    structure.setdefault("concat_mode", getattr(model, "concat_mode", None))

    return structure


# -----------------------------------------------------------------------------
# Forward tracing
# -----------------------------------------------------------------------------

class _LinearTraceRecorder:
    def __init__(self, model, layer_specs):
        self.model = model
        self.layer_specs = layer_specs
        self.records = {item["module"]: [] for item in layer_specs}
        self.first_call_order = {}
        self.call_counter = 0
        self.handles = []

    def _hook(self, path):
        def hook(module, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                raise RuntimeError(f"Layer '{path}' did not receive/return a Tensor as expected")

            x = inputs[0]
            y = output

            if x.requires_grad:
                x.retain_grad()
            if y.requires_grad:
                y.retain_grad()

            if path not in self.first_call_order:
                self.first_call_order[path] = self.call_counter

            self.call_counter += 1
            self.records[path].append({"input": x, "output": y})

        return hook

    def __enter__(self):
        for item in self.layer_specs:
            path = item["module"]
            module = _get_module(self.model, path)
            self.handles.append(module.register_forward_hook(self._hook(path)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _extract_model_output(result, output_index=0):
    if torch.is_tensor(result):
        return result

    if isinstance(result, (tuple, list)):
        result = result[output_index]
        if not torch.is_tensor(result):
            raise TypeError("Selected model output is not a Tensor")
        return result

    raise TypeError("Model forward must return a Tensor or tuple/list containing a Tensor")


def _run_model(model, x, structure):
    kwargs = dict(structure.get("forward_kwargs", {}))

    # If a model has ret_lat but the structure does not explicitly ask for it,
    # leave it False. The viewer gets its latent activity directly from the
    # traced network stage, which is more general and does not require ret_lat.
    result = model(x, **kwargs)
    outputs = _extract_model_output(result, structure.get("output_index", 0))

    if outputs.ndim != 3:
        raise ValueError(f"Viewer expects model output [B, C, T], got {tuple(outputs.shape)}")

    if outputs.requires_grad:
        outputs.retain_grad()

    return outputs


def _order_layers(structure, recorder, automatic_structure):
    layers = structure["layers"]

    if not automatic_structure:
        return layers

    # For auto-detection, use actual first-execution order rather than registration order.
    return sorted(layers, key=lambda item: recorder.first_call_order.get(item["module"], float("inf")))


def _validate_trace(layer_specs, recorder, timesteps, structure=None):
    structure = structure or {}
    graph_mode = structure.get("graph_mode", "chain")
    concat_mode = structure.get("concat_mode")

    for item in layer_specs:
        path = item["module"]
        count = len(recorder.records[path])
        expected = timesteps
        if graph_mode == "unet" and concat_mode == "temporal" and path.startswith("decoder_layers."):
            idx = int(path.split(".")[-1])
            if idx > 0:
                expected = 2 * timesteps
        if count != expected:
            raise ValueError(
                f"Layer '{path}' executed {count} times; expected {expected} calls for "
                f"a {timesteps}-timestep sample in graph_mode={graph_mode!r}."
            )

    if graph_mode != "chain":
        return

    for i in range(len(layer_specs) - 1):
        left = _get_module(recorder.model, layer_specs[i]["module"])
        right = _get_module(recorder.model, layer_specs[i + 1]["module"])
        if left.out_features != right.in_features:
            raise ValueError(
                f"Detected structure is not a simple chain: '{layer_specs[i]['module']}' outputs "
                f"{left.out_features}, but '{layer_specs[i + 1]['module']}' expects {right.in_features}. "
                "Use graph_mode='unet' for skip-connected U-Net models."
            )


# -----------------------------------------------------------------------------
# Loss / gradient extraction
# -----------------------------------------------------------------------------

def _loss_per_timestep(outputs, targets, loss_mode="mse", loss_scale=1.0, tau=10.0):
    loss_mode = loss_mode.lower()

    if loss_mode == "mse":
        return ((outputs - targets) ** 2).mean(dim=(0, 1)) * loss_scale

    if loss_mode in ("vr", "van_rossum", "van-rossum"):
        alpha = torch.exp(torch.tensor(-1.0 / tau, device=outputs.device, dtype=outputs.dtype))
        pred_trace = outputs[..., 0]
        target_trace = targets[..., 0]
        losses = [((pred_trace - target_trace) ** 2).mean()]

        for t in range(1, outputs.size(2)):
            pred_trace = alpha * pred_trace + outputs[..., t]
            target_trace = alpha * target_trace + targets[..., t]
            losses.append(((pred_trace - target_trace) ** 2).mean())

        return torch.stack(losses) * loss_scale

    raise ValueError("loss_mode must be 'mse' or 'vr'")


def _safe_grad(tensor):
    return torch.zeros_like(tensor) if tensor.grad is None else tensor.grad


def _stack_call_tensor(records, key):
    return torch.stack([record[key][0] for record in records], dim=1)


def _build_stages(model, layer_specs, recorder, outputs, structure):
    """
    Build stage activations from the tensors that actually flow between visible
    Linear layers.

    For an edge A -> B, the intermediate stage uses the *input of the next Linear*
    rather than A's raw pre-activation. This means LIF spikes, tanh outputs, ReLU
    outputs, etc. between the two Linear layers are shown automatically.
    """
    timesteps = outputs.size(2)
    stage_tensors = []

    first_records = recorder.records[layer_specs[0]["module"]]
    stage_tensors.append(_stack_call_tensor(first_records, "input"))

    for i in range(1, len(layer_specs)):
        records = recorder.records[layer_specs[i]["module"]]
        stage_tensors.append(_stack_call_tensor(records, "input"))

    last_module = _get_module(model, layer_specs[-1]["module"])
    if outputs.size(1) == last_module.out_features:
        stage_tensors.append(outputs[0])
    else:
        last_records = recorder.records[layer_specs[-1]["module"]]
        stage_tensors.append(_stack_call_tensor(last_records, "output"))

    stage_sizes = [tensor.size(0) for tensor in stage_tensors]

    stage_names = structure.get("stage_names")
    if stage_names is not None:
        if len(stage_names) != len(stage_tensors):
            raise ValueError(
                f"stage_names has {len(stage_names)} entries but the viewer structure has "
                f"{len(stage_tensors)} stages"
            )
        stage_names = list(stage_names)
    else:
        stage_names = ["Input"]
        for i, item in enumerate(layer_specs):
            if i == len(layer_specs) - 1:
                stage_names.append("Output")
            else:
                pretty = item.get("name", item["module"])
                stage_names.append(pretty)

    latent_stage = structure.get("latent_stage")
    if latent_stage is None:
        if len(stage_sizes) > 2:
            intermediate = list(range(1, len(stage_sizes) - 1))
            latent_stage = min(intermediate, key=lambda idx: stage_sizes[idx])
        else:
            latent_stage = 0
    else:
        latent_stage = int(latent_stage)

    if not 0 <= latent_stage < len(stage_tensors):
        raise ValueError(f"latent_stage={latent_stage} is outside 0..{len(stage_tensors) - 1}")

    if structure.get("stage_names") is None and 0 < latent_stage < len(stage_tensors) - 1:
        stage_names[latent_stage] = "Latent"

    return stage_tensors, stage_names, latent_stage


def _stage_gradients(layer_specs, recorder, outputs, stage_tensors):
    grads = []

    first_records = recorder.records[layer_specs[0]["module"]]
    grads.append(torch.stack([_safe_grad(record["input"])[0] for record in first_records], dim=1))

    for i in range(1, len(layer_specs)):
        records = recorder.records[layer_specs[i]["module"]]
        grads.append(torch.stack([_safe_grad(record["input"])[0] for record in records], dim=1))

    if outputs.grad is not None and outputs.grad.shape == outputs.shape:
        grads.append(outputs.grad[0])
    else:
        last_records = recorder.records[layer_specs[-1]["module"]]
        grads.append(torch.stack([_safe_grad(record["output"])[0] for record in last_records], dim=1))

    return grads


def _edge_gradient_contributions(model, layer_specs, recorder):
    edge_data = []

    for item in layer_specs:
        path = item["module"]
        module = _get_module(model, path)
        records = recorder.records[path]
        per_timestep = []

        for record in records:
            src = record["input"][0].detach()
            pre = record["output"]
            grad_pre = _safe_grad(pre)[0]
            contribution = grad_pre.unsqueeze(1) * src.unsqueeze(0)
            per_timestep.append(contribution.detach().cpu())

        edge_data.append({
            "name": item.get("name", path),
            "module": path,
            "weight": module.weight.detach().cpu(),
            "grad": torch.stack(per_timestep, dim=0),
        })

    return edge_data



def _stack_records(records, key):
    return torch.stack([r[key][0] for r in records], dim=1)


def _unet_external_records(path, records, timesteps, concat_mode):
    """Return display input/grad tensors and per-timestep record groups."""
    if concat_mode == "temporal" and path.startswith("decoder_layers.") and int(path.split(".")[-1]) > 0:
        skip_records = records[:timesteps]
        dec_records = records[timesteps:]
        # Both halves use the same feature width. Sum activations/gradients so the
        # displayed external timestep represents the complete 2T layer usage.
        inp = _stack_records(skip_records, "input") + _stack_records(dec_records, "input")
        inp_grad = torch.stack([_safe_grad(a["input"])[0] + _safe_grad(b["input"])[0]
                                for a, b in zip(skip_records, dec_records)], dim=1)
        groups = list(zip(skip_records, dec_records))
        return inp, inp_grad, groups
    inp = _stack_records(records, "input")
    inp_grad = torch.stack([_safe_grad(r["input"])[0] for r in records], dim=1)
    return inp, inp_grad, [(r,) for r in records]


def _build_unet_graph(model, layer_specs, recorder, outputs, structure):
    """Build an edge-centric graph for skip-connected U-Nets.

    Each Linear gets its own source and destination stage. This keeps every
    weight matrix dimensionally correct even when a decoder source is a spatial
    concatenation. Temporal concatenation is folded back to T external steps.
    """
    T = outputs.size(2)
    concat_mode = structure.get("concat_mode", getattr(model, "concat_mode", None))
    nenc = len(getattr(model, "encoder_layers", []))
    stage_data, edge_data = [], []

    def post_activation_for(k, item, module):
        path = item["module"]
        # Final displayed layer uses the model output (post-LIF).
        if k == len(layer_specs) - 1 and outputs.size(1) == module.out_features:
            g = outputs.grad[0] if outputs.grad is not None else torch.zeros_like(outputs[0])
            return outputs[0], g
        # The next Linear input contains the post-neuron output. For spatial
        # decoder concatenation it is the first block; for temporal mode it is
        # the decoder half of the 2T call sequence.
        nxt = layer_specs[k + 1]
        nr = recorder.records[nxt["module"]]
        if concat_mode == "temporal" and nxt["module"].startswith("decoder_layers.") and int(nxt["module"].split(".")[-1]) > 0:
            nr = nr[T:]
        a = _stack_records(nr, "input")
        g = torch.stack([_safe_grad(r["input"])[0] for r in nr], dim=1)
        return a[:module.out_features], g[:module.out_features]

    for k, item in enumerate(layer_specs):
        path = item["module"]
        module = _get_module(model, path)
        records = recorder.records[path]
        src_act, src_grad, groups = _unet_external_records(path, records, T, concat_mode)
        dst_act, dst_grad = post_activation_for(k, item, module)

        src_name = f"{item.get('name', path)} input"
        if concat_mode == "spatial" and path.startswith("decoder_layers.") and int(path.split(".")[-1]) > 0:
            src_name += " (concat)"
        if concat_mode == "temporal" and path.startswith("decoder_layers.") and int(path.split(".")[-1]) > 0:
            src_name += " (skip + decoder)"

        src_idx = len(stage_data)
        stage_data.append({"name": src_name, "activation": src_act.detach().cpu(), "gradient": src_grad.detach().cpu(),
                           "neuron_module": None, "decay": [None]*src_act.size(0), "threshold": [None]*src_act.size(0)})
        dst_idx = len(stage_data)
        neuron_path = item.get("neuron")
        neuron = _get_module(model, neuron_path) if neuron_path else None
        stage_data.append({"name": item.get("name", path), "activation": dst_act.detach().cpu(), "gradient": dst_grad.detach().cpu(),
                           "neuron_module": neuron_path,
                           "decay": _expand_node_value(getattr(neuron, "beta", None), dst_act.size(0)),
                           "threshold": _expand_node_value(getattr(neuron, "threshold", None), dst_act.size(0))})

        per_t = []
        for grp in groups:
            contrib = None
            for r in grp:
                src = r["input"][0].detach()
                gp = _safe_grad(r["output"])[0]
                c = gp.unsqueeze(1) * src.unsqueeze(0)
                contrib = c if contrib is None else contrib + c
            per_t.append(contrib.detach().cpu())
        edge_data.append({"name": item.get("name", path), "module": path,
                          "weight": module.weight.detach().cpu(), "grad": torch.stack(per_t),
                          "src": src_idx, "dst": dst_idx})

    latent_stage = int(structure.get("latent_stage", 2*nenc-1))
    return stage_data, edge_data, latent_stage


def _spike_density_per_timestep(latent, threshold=0.5):
    # latent: [H, T]
    return (latent > threshold).float().mean(dim=0)


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

def _write_csvs(output_dir, losses, spike_density, stage_data, edge_data, target, latent_stage, structure_info):
    output_dir = Path(output_dir)

    loss_path = output_dir / "loss_per_timestep.csv"
    with loss_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "loss"])
        for t, loss in enumerate(losses.tolist()):
            writer.writerow([t, loss])

    density_path = output_dir / "spike_density_per_timestep.csv"
    with density_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "spike_density", "latent_stage"])
        for t, density in enumerate(spike_density.tolist()):
            writer.writerow([t, density, latent_stage])

    edge_path = output_dir / "edge_gradients.csv"
    with edge_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "edge_index", "layer_name", "module", "source_node", "destination_node", "weight", "gradient_contribution"])
        for edge_index, edge in enumerate(edge_data):
            grads = edge["grad"]
            weight = edge["weight"]
            for t in range(grads.size(0)):
                for dst in range(grads.size(1)):
                    for src in range(grads.size(2)):
                        writer.writerow([t, edge_index, edge["name"], edge["module"], src, dst, weight[dst, src].item(), grads[t, dst, src].item()])

    node_path = output_dir / "node_gradients.csv"
    with node_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "stage_index", "stage_name", "node", "activation", "gradient", "decay", "threshold", "target"])

        for stage_index, stage in enumerate(stage_data):
            activation = stage["activation"]
            gradient = stage["gradient"]

            for t in range(activation.size(1)):
                for node in range(activation.size(0)):
                    target_value = ""
                    if stage_index == len(stage_data) - 1 and node < target.size(1):
                        target_value = target[0, node, t].item()
                    elif stage_index == 0 and node < target.size(1):
                        target_value = target[0, node, t].item()

                    writer.writerow([
                        t,
                        stage_index,
                        stage["name"],
                        node,
                        activation[node, t].item(),
                        gradient[node, t].item(),
                        stage.get("decay", [None] * activation.size(0))[node],
                        stage.get("threshold", [None] * activation.size(0))[node],
                        target_value,
                    ])

    structure_path = output_dir / "viewer_structure.json"
    structure_path.write_text(json.dumps(structure_info, indent=2), encoding="utf-8")

    paths = {
        "loss": loss_path,
        "spike_density": density_path,
        "edges": edge_path,
        "nodes": node_path,
        "structure": structure_path,
    }

    # Preserve the old filenames for the common 3-stage architecture.
    if len(edge_data) == 2:
        old_names = ["encoder_edge_gradients.csv", "decoder_edge_gradients.csv"]
        for old_name, edge in zip(old_names, edge_data):
            old_path = output_dir / old_name
            with old_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "source_node", "destination_node", "weight", "gradient_contribution"])
                for t in range(edge["grad"].size(0)):
                    for dst in range(edge["grad"].size(1)):
                        for src in range(edge["grad"].size(2)):
                            writer.writerow([t, src, dst, edge["weight"][dst, src].item(), edge["grad"][t, dst, src].item()])
            paths["encoder_edges" if old_name.startswith("encoder") else "decoder_edges"] = old_path

    return paths


def _make_html(html_path, losses, spike_density, stage_data, edge_data, target, latent_stage, loss_mode, spike_threshold=0.5):
    stages_json = []
    for stage in stage_data:
        stages_json.append({
            "name": stage["name"],
            "size": stage["activation"].size(0),
            "activation": stage["activation"].transpose(0, 1).tolist(),
            "gradient": stage["gradient"].transpose(0, 1).tolist(),
            "neuronModule": stage.get("neuron_module"),
            "decay": stage.get("decay", [None] * stage["activation"].size(0)),
            "threshold": stage.get("threshold", [None] * stage["activation"].size(0)),
        })

    edges_json = []
    for i, edge in enumerate(edge_data):
        edges_json.append({
            "name": edge["name"],
            "module": edge["module"],
            "src": edge.get("src", i),
            "dst": edge.get("dst", i + 1),
            "weight": edge["weight"].tolist(),
            "grad": edge["grad"].tolist(),
        })

    data = {
        "loss": losses.tolist(),
        "spikeDensity": spike_density.tolist(),
        "stages": stages_json,
        "edges": edges_json,
        "target": target[0].transpose(0, 1).tolist(),
        "inputRaster": stage_data[0]["activation"].transpose(0, 1).tolist(),
        "outputRaster": stage_data[-1]["activation"].transpose(0, 1).tolist(),
        "latentStage": latent_stage,
        "timesteps": len(losses),
        "lossMode": loss_mode,
        "spikeThreshold": float(spike_threshold),
        "structureText": " → ".join(str(stage["size"]) for stage in stages_json),
    }

    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Temporal Gradient Viewer</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:6px;background:#fafafa;color:#222;overflow-x:hidden}
.card{max-width:1500px;margin:0 auto 6px;background:white;border:1px solid #ddd;border-radius:8px;padding:6px 10px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
h1{font-size:19px;margin:0}h2{font-size:15px;margin:0}.topline{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:2px}.controls{display:grid;grid-template-columns:minmax(240px,1fr) auto auto auto auto;gap:7px;align-items:center;margin:2px 0}
input[type=range]{width:100%}select,input[type=number]{font-size:11px;padding:1px 3px}svg{width:100%;height:auto;background:white;display:block}.edge{fill:none;stroke-linecap:round;pointer-events:stroke}.node{stroke:#333;stroke-width:1}.node-label{font-size:9px;dominant-baseline:middle}.param-label{font-size:7px;fill:#555;dominant-baseline:middle}.layer-label{font-size:13px;font-weight:600;text-anchor:middle}.small{font-size:10px;color:#666}.metric{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.legend{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:1px 0;font-size:10px}.swatch{width:16px;height:3px;display:inline-block;vertical-align:middle;margin-right:3px}
#network{max-height:57vh}.plot-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1500px;margin:0 auto}.plot-row .card{margin:0}#lossPlot,#densityPlot{max-height:16vh}.raster-card{max-width:1500px}.raster-wrap{width:100%;overflow-x:auto}.raster{width:100%;height:auto;display:block;border:1px solid #ddd;background:white;image-rendering:pixelated}@media(max-width:900px){.plot-row{grid-template-columns:1fr}#network{max-height:48vh}}
</style>
</head>
<body>
<div class="card">
<div class="topline"><h1>Temporal Gradient Viewer</h1><div class="small">Structure: <span id="structureText"></span> · edge width/opacity = per-timestep contribution to full-sample BPTT gradient</div></div>
<div class="controls"><input id="timeSlider" type="range" min="0" max="0" value="0" step="1"><label>Timestep <input id="timeNumber" type="number" min="0" value="0" style="width:68px"></label><label>View <select id="viewMode"><option value="gradient">Gradients</option><option value="spike">Spike routing</option></select></label><label>Normalise <select id="scaleMode"><option value="current">Current</option><option value="global">Global</option></select></label><label>Gradient scale <select id="gradScaleMode"><option value="linear">Linear</option><option value="log">Logarithmic</option></select></label></div>
<div class="legend"><span id="legendPositive"><span class="swatch" id="swatchPositive" style="background:#d62728"></span><span id="positiveLabel">positive gradient</span></span><span id="legendNegative"><span class="swatch" id="swatchNegative" style="background:#1f77b4"></span><span id="negativeLabel">negative gradient</span></span><span>Loss: <span id="currentLoss" class="metric"></span></span><span>Latent spike density: <span id="currentDensity" class="metric"></span></span><span>Latent stage: <span id="latentStage" class="metric"></span></span><span>Loss mode: <span id="lossMode" class="metric"></span></span></div>
<svg id="network" viewBox="0 0 1400 520"></svg>
</div>
<div class="plot-row"><div class="card"><div class="topline"><h2>Loss per timestep</h2><div class="small">Marker follows timestep</div></div><svg id="lossPlot" viewBox="0 0 680 165"></svg></div><div class="card"><div class="topline"><h2>Latent spike density</h2><div class="small">Fraction above spike threshold</div></div><svg id="densityPlot" viewBox="0 0 680 165"></svg></div></div>
<div class="card raster-card"><div class="topline"><h2>Input</h2><div class="small">Channel × timestep · red line follows selected timestep</div></div><div class="raster-wrap"><canvas id="inputRaster" class="raster" width="1400" height="250"></canvas></div></div>
<div class="card raster-card"><div class="topline"><h2>Reconstructed output</h2><div class="small">Channel × timestep · red line follows selected timestep</div></div><div class="raster-wrap"><canvas id="outputRaster" class="raster" width="1400" height="250"></canvas></div></div>
<script>
const D=__DATA__;
const ns="http://www.w3.org/2000/svg",network=document.getElementById("network"),slider=document.getElementById("timeSlider"),number=document.getElementById("timeNumber"),currentLoss=document.getElementById("currentLoss"),currentDensity=document.getElementById("currentDensity"),lossMode=document.getElementById("lossMode"),viewMode=document.getElementById("viewMode"),scaleMode=document.getElementById("scaleMode"),gradScaleMode=document.getElementById("gradScaleMode"),swatchPositive=document.getElementById("swatchPositive"),swatchNegative=document.getElementById("swatchNegative"),positiveLabel=document.getElementById("positiveLabel"),negativeLabel=document.getElementById("negativeLabel");
document.getElementById("structureText").textContent=D.structureText;document.getElementById("latentStage").textContent=`${D.latentStage}: ${D.stages[D.latentStage].name}`;slider.max=D.timesteps-1;number.max=D.timesteps-1;lossMode.textContent=D.lossMode;
function svgEl(name,attrs={}){const el=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,v);return el}
function yPositions(count,top=40,bottom=492){if(count===1)return[(top+bottom)/2];const step=(bottom-top)/(count-1);return Array.from({length:count},(_,i)=>top+i*step)}
const nStages=D.stages.length,left=72,right=1328,xStep=nStages===1?0:(right-left)/(nStages-1),stagePos=[],nodes=[],edgeLines=[];
for(let s=0;s<nStages;s++){const x=left+s*xStep,ys=yPositions(D.stages[s].size);stagePos.push(ys.map(y=>({x,y})));const label=svgEl("text",{x:x,y:22,class:"layer-label"});label.textContent=`${D.stages[s].name} (${D.stages[s].size})`;network.appendChild(label);nodes.push([])}
for(let e=0;e<D.edges.length;e++){const edge=D.edges[e],lines=[];for(let dst=0;dst<D.stages[edge.dst].size;dst++)for(let src=0;src<D.stages[edge.src].size;src++){const a=stagePos[edge.src][src],b=stagePos[edge.dst][dst],line=svgEl("line",{x1:a.x+9,y1:a.y,x2:b.x-9,y2:b.y,class:"edge"}),title=svgEl("title");line.appendChild(title);network.appendChild(line);lines.push({line,title,src,dst})}edgeLines.push(lines)}
function fmtParam(v){if(v===null||v===undefined)return null;const a=Math.abs(v);if((a!==0&&a<.001)||a>=100)return v.toExponential(1);return Number(v.toFixed(3)).toString()}
function addNode(p,label,side,decay,threshold){const g=svgEl("g"),c=svgEl("circle",{cx:p.x,cy:p.y,r:8,class:"node",fill:"#eee"}),title=svgEl("title");c.appendChild(title);g.appendChild(c);const anchor=side==="left"?"end":"start",x=side==="left"?p.x-12:p.x+12,hasParams=decay!==null||threshold!==null;const tx=svgEl("text",{x:x,y:hasParams?p.y-3:p.y,"text-anchor":anchor,class:"node-label"});tx.textContent=label;g.appendChild(tx);let paramText=null;if(hasParams){paramText=svgEl("text",{x:x,y:p.y+5.5,"text-anchor":anchor,class:"param-label"});const parts=[];if(decay!==null)parts.push(`β=${fmtParam(decay)}`);if(threshold!==null)parts.push(`θ=${fmtParam(threshold)}`);paramText.textContent=parts.join("  ");g.appendChild(paramText)}network.appendChild(g);return{circle:c,title,paramText}}
for(let s=0;s<nStages;s++)for(let i=0;i<D.stages[s].size;i++){const side=s===0?"left":"right",decay=D.stages[s].decay[i],threshold=D.stages[s].threshold[i];nodes[s].push(addNode(stagePos[s][i],`${s===0?"I":s===nStages-1?"O":"N"}${i}`,side,decay,threshold))}
let globalGradMax=0;for(const edge of D.edges)for(let t=0;t<D.timesteps;t++)for(let dst=0;dst<edge.grad[t].length;dst++)for(let src=0;src<edge.grad[t][dst].length;src++)globalGradMax=Math.max(globalGradMax,Math.abs(edge.grad[t][dst][src]));if(globalGradMax===0)globalGradMax=1;
let globalRouteMax=0;for(let e=0;e<D.edges.length;e++){const edge=D.edges[e],srcStage=D.stages[edge.src];for(let t=0;t<D.timesteps;t++)for(let src=0;src<srcStage.size;src++){const a=srcStage.activation[t][src];if(Math.abs(a)>D.spikeThreshold)for(let dst=0;dst<D.stages[edge.dst].size;dst++)globalRouteMax=Math.max(globalRouteMax,Math.abs(a*edge.weight[dst][src]))}}if(globalRouteMax===0)globalRouteMax=1;
function gradColor(v){return v>0?"#d62728":v<0?"#1f77b4":"#aaa"}function routeColor(v){return v>0?"#2ca02c":v<0?"#9467bd":"#aaa"}function gradientRatio(v,m){if(m<=0)return 0;const q=Math.min(1,Math.abs(v)/m);if(gradScaleMode.value==="log")return Math.log1p(999999*q)/Math.log(1000000);return q}function nodeFill(v,m){const r=gradientRatio(v,m),a=.12+.78*r;return v>=0?`rgba(214,39,40,${a})`:`rgba(31,119,180,${a})`}function spikeNodeFill(a,threshold){const active=Math.abs(a)>threshold;return active?"rgba(44,160,44,.88)":"rgba(180,180,180,.18)"}
function currentGradMax(t){let m=0;for(const edge of D.edges)for(let dst=0;dst<edge.grad[t].length;dst++)for(let src=0;src<edge.grad[t][dst].length;src++)m=Math.max(m,Math.abs(edge.grad[t][dst][src]));return m||1}
function currentRouteMax(t){let m=0;for(let e=0;e<D.edges.length;e++){const edge=D.edges[e],srcStage=D.stages[edge.src];for(let src=0;src<srcStage.size;src++){const a=srcStage.activation[t][src];if(Math.abs(a)>D.spikeThreshold)for(let dst=0;dst<D.stages[edge.dst].size;dst++)m=Math.max(m,Math.abs(a*edge.weight[dst][src]))}}return m||1}
function updateLegend(){if(viewMode.value==="gradient"){swatchPositive.style.background="#d62728";swatchNegative.style.background="#1f77b4";positiveLabel.textContent="positive gradient";negativeLabel.textContent="negative gradient"}else{swatchPositive.style.background="#2ca02c";swatchNegative.style.background="#9467bd";positiveLabel.textContent="positive routed signal";negativeLabel.textContent="negative routed signal"}}
function updateNetwork(t){const gradientMode=viewMode.value==="gradient",edgeMax=gradientMode?(scaleMode.value==="global"?globalGradMax:currentGradMax(t)):(scaleMode.value==="global"?globalRouteMax:currentRouteMax(t));for(let e=0;e<D.edges.length;e++){const edge=D.edges[e],srcStage=D.stages[edge.src];for(const item of edgeLines[e]){const weight=edge.weight[item.dst][item.src];if(gradientMode){const g=edge.grad[t][item.dst][item.src],r=gradientRatio(g,edgeMax);item.line.setAttribute("stroke",gradColor(g));item.line.setAttribute("stroke-width",.2+4.5*r);item.line.setAttribute("stroke-opacity",.025+.94*r);item.title.textContent=`${edge.name}: ${item.src} → ${item.dst}\nmodule=${edge.module}\nweight=${weight.toExponential(4)}\ngrad contribution=${g.toExponential(4)}`}else{const a=srcStage.activation[t][item.src],active=Math.abs(a)>D.spikeThreshold,routed=active?a*weight:0,r=Math.min(1,Math.abs(routed)/edgeMax);item.line.setAttribute("stroke",active?routeColor(routed):"#aaa");item.line.setAttribute("stroke-width",active?.35+4.5*r:.15);item.line.setAttribute("stroke-opacity",active?.08+.9*r:.015);item.title.textContent=`${edge.name}: ${item.src} → ${item.dst}\nmodule=${edge.module}\nsource activation=${a.toExponential(4)}\nspike active=${active}\nweight=${weight.toExponential(4)}\nrouted signal=${routed.toExponential(4)}`}}}
for(let s=0;s<nStages;s++){const grads=D.stages[s].gradient[t],acts=D.stages[s].activation[t],m=Math.max(...grads.map(Math.abs),1e-12);for(let i=0;i<D.stages[s].size;i++){const g=grads[i],a=acts[i];nodes[s][i].circle.setAttribute("fill",gradientMode?nodeFill(g,m):spikeNodeFill(a,D.spikeThreshold));let extra="";if(D.stages[s].decay[i]!==null)extra+=`\ndecay β=${D.stages[s].decay[i]}`;if(D.stages[s].threshold[i]!==null)extra+=`\nthreshold θ=${D.stages[s].threshold[i]}`;if(D.stages[s].neuronModule)extra+=`\nneuron=${D.stages[s].neuronModule}`;if(s===nStages-1&&i<D.target[t].length)extra+=`\ntarget=${D.target[t][i].toFixed(4)}`;nodes[s][i].title.textContent=gradientMode?`${D.stages[s].name} ${i}\nactivation=${a.toFixed(4)}\ngrad=${g.toExponential(4)}${extra}`:`${D.stages[s].name} ${i}\nactivation=${a.toFixed(4)}\nspike active=${Math.abs(a)>D.spikeThreshold}${extra}`}}
currentLoss.textContent=D.loss[t].toExponential(6);currentDensity.textContent=(100*D.spikeDensity[t]).toFixed(2)+"%";updateLegend()}
const lossSvg=document.getElementById("lossPlot"),densitySvg=document.getElementById("densityPlot"),PW=680,PH=165,PL=55,PR=14,PT=12,PB=31;function px(t){return PL+(PW-PL-PR)*(D.timesteps===1?0:t/(D.timesteps-1))}
function drawSeries(svg,values,yMin,yMax,formatY){const span=Math.max(yMax-yMin,1e-12),py=v=>PT+(PH-PT-PB)*(1-(v-yMin)/span);svg.appendChild(svgEl("line",{x1:PL,y1:PH-PB,x2:PW-PR,y2:PH-PB,stroke:"#444"}));svg.appendChild(svgEl("line",{x1:PL,y1:PT,x2:PL,y2:PH-PB,stroke:"#444"}));let d="";for(let t=0;t<D.timesteps;t++)d+=`${t===0?"M":"L"}${px(t)},${py(values[t])} `;svg.appendChild(svgEl("path",{d,fill:"none",stroke:"#222","stroke-width":"2"}));const marker=svgEl("circle",{cx:px(0),cy:py(values[0]),r:4.5,fill:"#d62728"});svg.appendChild(marker);const xlab=svgEl("text",{x:PW/2,y:PH-7,"text-anchor":"middle","font-size":"11"});xlab.textContent="Timestep";svg.appendChild(xlab);const top=svgEl("text",{x:3,y:PT+5,"font-size":"9"});top.textContent=formatY(yMax);svg.appendChild(top);const bot=svgEl("text",{x:3,y:PH-PB,"font-size":"9"});bot.textContent=formatY(yMin);svg.appendChild(bot);return{marker,py}}
const maxLoss=Math.max(...D.loss,1e-12),minLoss=Math.min(...D.loss),lossSeries=drawSeries(lossSvg,D.loss,minLoss,maxLoss,v=>v.toExponential(2)),densitySeries=drawSeries(densitySvg,D.spikeDensity,0,1,v=>(100*v).toFixed(0)+"%");

const inputRasterCanvas=document.getElementById("inputRaster"),outputRasterCanvas=document.getElementById("outputRaster");

function buildRaster(values){
    const W=1400,H=250,L=48,R=12,T=10,B=28,plotW=W-L-R,plotH=H-T-B;
    const nT=values.length,nC=values[0].length;
    const base=document.createElement("canvas");
    base.width=W;base.height=H;
    const ctx=base.getContext("2d");

    ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);

    let minV=Infinity,maxV=-Infinity,maxAbs=0;
    for(let t=0;t<nT;t++)for(let c=0;c<nC;c++){
        const v=values[t][c];
        minV=Math.min(minV,v);maxV=Math.max(maxV,v);maxAbs=Math.max(maxAbs,Math.abs(v));
    }

    const unitRange=minV>=0&&maxV<=1;
    const cellW=plotW/nT,cellH=plotH/nC;

    for(let t=0;t<nT;t++)for(let c=0;c<nC;c++){
        const v=values[t][c];
        if(unitRange){
            const q=Math.max(0,Math.min(1,v));
            const grey=Math.round(255-235*q);
            ctx.fillStyle=`rgb(${grey},${grey},${grey})`;
        }else{
            const q=maxAbs>0?Math.min(1,Math.abs(v)/maxAbs):0;
            const fade=Math.round(255-200*q);
            ctx.fillStyle=v>=0?`rgb(255,${fade},${fade})`:`rgb(${fade},${fade},255)`;
        }
        ctx.fillRect(L+t*cellW,T+c*cellH,Math.ceil(cellW)+.3,Math.ceil(cellH)+.3);
    }

    ctx.strokeStyle="#444";ctx.lineWidth=1;
    ctx.strokeRect(L,T,plotW,plotH);

    ctx.fillStyle="#444";ctx.font="10px system-ui";
    ctx.textAlign="right";ctx.textBaseline="middle";
    const channelTicks=[0,Math.floor((nC-1)/2),nC-1];
    for(const c of channelTicks){
        const y=T+(c+.5)*cellH;
        ctx.fillText(String(c),L-6,y);
    }

    ctx.textAlign="center";ctx.textBaseline="top";
    const timeTicks=[0,Math.floor((nT-1)/4),Math.floor((nT-1)/2),Math.floor(3*(nT-1)/4),nT-1];
    for(const t of timeTicks){
        const x=L+(t+.5)*cellW;
        ctx.fillText(String(t),x,T+plotH+6);
    }

    ctx.save();
    ctx.translate(11,T+plotH/2);
    ctx.rotate(-Math.PI/2);
    ctx.textAlign="center";ctx.textBaseline="top";
    ctx.fillText("Channel",0,0);
    ctx.restore();

    ctx.textAlign="center";ctx.textBaseline="top";
    ctx.fillText("Timestep",L+plotW/2,H-13);

    return{base,L,T,plotW,plotH,nT};
}

const inputRasterInfo=buildRaster(D.inputRaster),outputRasterInfo=buildRaster(D.outputRaster);

function drawRaster(canvas,info,t){
    const ctx=canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(info.base,0,0);
    const x=info.L+info.plotW*(info.nT===1?0.5:(t+.5)/info.nT);
    ctx.strokeStyle="#d62728";ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(x,info.T);ctx.lineTo(x,info.T+info.plotH);ctx.stroke();
}

function update(t){t=Math.max(0,Math.min(D.timesteps-1,Number(t)));slider.value=t;number.value=t;updateNetwork(t);lossSeries.marker.setAttribute("cx",px(t));lossSeries.marker.setAttribute("cy",lossSeries.py(D.loss[t]));densitySeries.marker.setAttribute("cx",px(t));densitySeries.marker.setAttribute("cy",densitySeries.py(D.spikeDensity[t]));drawRaster(inputRasterCanvas,inputRasterInfo,t);drawRaster(outputRasterCanvas,outputRasterInfo,t)}slider.addEventListener("input",()=>update(slider.value));number.addEventListener("change",()=>update(number.value));viewMode.addEventListener("change",()=>update(slider.value));scaleMode.addEventListener("change",()=>update(slider.value));gradScaleMode.addEventListener("change",()=>update(slider.value));update(0);
</script></body></html>'''

    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(html_path).write_text(html, encoding="utf-8")


def export_gradient_viewer(model, sample, output_dir="gradient_trace", loss_mode="mse", loss_scale=1.0, tau=10.0, device=None, structure=None):
    """
    Export an interactive HTML gradient viewer for a temporal network.

    The viewer can either:
      * automatically detect the main chain of explicit nn.Linear layers, or
      * use a structure passed here, or
      * use a structure stored on the model itself.

    Example model-side specification:

        self.gradient_viewer_structure = {
            "layers": ["encoder", "decoder"],
            "stage_names": ["Input", "Latent", "Output"],
            "latent_stage": 1,
        }

    For a deeper model:

        self.gradient_viewer_structure = {
            "layers": [
                {"module": "encoder_layers.0", "neuron": "encoder_lifs.0"},
                {"module": "encoder_layers.1", "neuron": "encoder_lifs.1"},
                {"module": "recurrent_linear", "neuron": "recurrent_lif"},
                {"module": "decoder_layers.0", "neuron": "decoder_lifs.0"},
                {"module": "output_layer", "neuron": "output_lif"},
            ],
            "stage_names": ["Input", "Enc 0", "Enc 1", "Latent", "Dec 0", "Output"],
            "latent_stage": 3,
        }

    Notes
    -----
    * Every displayed edge is an nn.Linear module.
    * Intermediate node activations are taken from the actual tensor entering the
      next Linear layer, so post-LIF spikes / tanh / ReLU values are shown when
      those operations sit between two displayed Linear layers.
    * One full-sequence backward pass is used. Therefore the local edge gradient
      contribution at timestep t contains gradient arriving from later timesteps
      through recurrent state.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)

    if sample.ndim == 2:
        sample = sample.unsqueeze(0)
    if sample.ndim != 3 or sample.size(0) != 1:
        raise ValueError("sample must have shape [C, T] or [1, C, T]")

    explicit_structure = structure is not None or any(
        hasattr(model, name)
        for name in (
            "get_gradient_viewer_structure",
            "get_gradient_viewer_spec",
            "gradient_viewer_structure",
            "gradient_viewer_spec",
        )
    )

    structure = _normalise_structure(model, structure)
    candidate_layers = structure["layers"]

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    target = sample.detach().clone().to(device)
    x = sample.detach().clone().to(device).requires_grad_(True)

    with _LinearTraceRecorder(model, candidate_layers) as recorder:
        outputs = _run_model(model, x, structure)

    layer_specs = _order_layers(structure, recorder, automatic_structure=not explicit_structure)
    _validate_trace(layer_specs, recorder, x.size(2), structure)

    first_layer = _get_module(model, layer_specs[0]["module"])
    if x.size(1) != first_layer.in_features:
        raise ValueError(
            f"sample has {x.size(1)} channels but first displayed layer "
            f"'{layer_specs[0]['module']}' expects {first_layer.in_features}"
        )

    if outputs.size(1) != target.size(1):
        raise ValueError(
            f"model output has {outputs.size(1)} channels but reconstruction target has {target.size(1)}"
        )

    timestep_losses = _loss_per_timestep(outputs, target, loss_mode=loss_mode, loss_scale=loss_scale, tau=tau)
    total_loss = timestep_losses.mean()
    total_loss.backward()

    if structure.get("graph_mode") == "unet":
        stage_data, edge_data, latent_stage = _build_unet_graph(model, layer_specs, recorder, outputs, structure)
    else:
        stage_tensors, stage_names, latent_stage = _build_stages(model, layer_specs, recorder, outputs, structure)
        stage_grads = _stage_gradients(layer_specs, recorder, outputs, stage_tensors)

        stage_data = []
        for name, activation, gradient in zip(stage_names, stage_tensors, stage_grads):
            stage_data.append({
                "name": name,
                "activation": activation.detach().cpu(),
                "gradient": gradient.detach().cpu(),
            })

        stage_data = _attach_neuron_metadata(model, layer_specs, stage_data)
        edge_data = _edge_gradient_contributions(model, layer_specs, recorder)

    losses_cpu = timestep_losses.detach().cpu()
    target_cpu = target.detach().cpu()
    latent_cpu = stage_data[latent_stage]["activation"]
    spike_density_cpu = _spike_density_per_timestep(latent_cpu, threshold=float(structure.get("spike_threshold", 0.5)))

    structure_info = {
        "stages": [{
            "index": i,
            "name": stage["name"],
            "size": int(stage["activation"].size(0)),
            "neuron_module": stage.get("neuron_module"),
            "decay": stage.get("decay"),
            "threshold": stage.get("threshold"),
        } for i, stage in enumerate(stage_data)],
        "edges": [{"index": i, "name": edge["name"], "module": edge["module"], "src": edge.get("src", i), "dst": edge.get("dst", i + 1)} for i, edge in enumerate(edge_data)],
        "latent_stage": latent_stage,
        "spike_threshold": float(structure.get("spike_threshold", 0.5)),
        "loss_mode": loss_mode,
    }

    csv_paths = _write_csvs(output_dir, losses_cpu, spike_density_cpu, stage_data, edge_data, target_cpu, latent_stage, structure_info)

    html_path = output_dir / "gradient_viewer.html"
    _make_html(html_path, losses_cpu, spike_density_cpu, stage_data, edge_data, target_cpu, latent_stage, loss_mode, spike_threshold=float(structure.get("spike_threshold", 0.5)))

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    print("Detected / selected structure:")
    print(" -> ".join(f"{stage['name']}({stage['activation'].size(0)})" for stage in stage_data))
    print(f"Latent stage: {latent_stage} ({stage_data[latent_stage]['name']})")
    print(f"Gradient viewer written to: {html_path}")
    print(f"Structure JSON written to: {csv_paths['structure']}")
    print(f"Loss CSV written to: {csv_paths['loss']}")
    print(f"Spike density CSV written to: {csv_paths['spike_density']}")
    print(f"Edge gradient CSV written to: {csv_paths['edges']}")
    print(f"Node gradient CSV written to: {csv_paths['nodes']}")

    return {"html": html_path, **csv_paths}


def export_recurrent_gradient_viewer(model, sample, output_dir="gradient_trace", loss_mode="mse", loss_scale=1.0, tau=10.0, device=None, structure=None):
    """Backward-compatible alias for export_gradient_viewer()."""
    return export_gradient_viewer(
        model=model,
        sample=sample,
        output_dir=output_dir,
        loss_mode=loss_mode,
        loss_scale=loss_scale,
        tau=tau,
        device=device,
        structure=structure,
    )