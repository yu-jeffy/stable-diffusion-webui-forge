import os
import re

import lora_patches
import functools
import network

import torch
from typing import Union

from modules import shared, sd_models, errors, scripts
from ldm_patched.modules.utils import load_torch_file
from ldm_patched.modules.sd import load_lora_for_models


@functools.lru_cache(maxsize=5)
def load_lora_state_dict(filename):
    return load_torch_file(filename, safe_load=True)


def convert_diffusers_name_to_compvis(key, is_sd2):
    pass


def assign_network_names_to_compvis_modules(sd_model):
    pass


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    return net


def purge_networks_from_memory():
    pass


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    global lora_state_dict_cache

    current_sd = sd_models.model_data.get_sd_model()
    if current_sd is None:
        return

    loaded_networks.clear()

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        try:
            net = load_network(name, network_on_disk)
        except Exception as e:
            errors.display(e, f"loading network {network_on_disk.filename}")
            continue
        net.mentioned_name = name
        network_on_disk.read_hash()
        loaded_networks.append(net)

    compiled_lora_targets = []
    for a, b, c in zip(networks_on_disk, unet_multipliers, te_multipliers):
        compiled_lora_targets.append([a.filename, b, c])

    compiled_lora_targets_hash = str(compiled_lora_targets)

    if current_sd.current_lora_hash == compiled_lora_targets_hash:
        return

    current_sd.current_lora_hash = compiled_lora_targets_hash
    current_sd.forge_objects.unet = current_sd.forge_objects_original.unet
    current_sd.forge_objects.clip = current_sd.forge_objects_original.clip

    for filename, strength_model, strength_clip in compiled_lora_targets:
        lora_sd = load_lora_state_dict(filename)
        current_sd.forge_objects.unet, current_sd.forge_objects.clip = load_lora_for_models(
            current_sd.forge_objects.unet, current_sd.forge_objects.clip, lora_sd, strength_model, strength_clip,
            filename=filename)

    current_sd.forge_objects_after_applying_lora = current_sd.forge_objects.shallow_copy()
    return


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    pass


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
<<<<<<< HEAD
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores original weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != ():
        if current_names != ():
            raise RuntimeError("no backup weights found and current weights are not unchanged")

        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.network_weights_backup = weights_backup

    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_backup = self.out_proj.bias.to(devices.cpu, copy=True)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = self.bias.to(devices.cpu, copy=True)
        else:
            bias_backup = None
        self.network_bias_backup = bias_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)

        for net in loaded_networks:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                try:
                    with torch.no_grad():
                        if getattr(self, 'fp16_weight', None) is None:
                            weight = self.weight
                            bias = self.bias
                        else:
                            weight = self.fp16_weight.clone().to(self.weight.device)
                            bias = getattr(self, 'fp16_bias', None)
                            if bias is not None:
                                bias = bias.clone().to(self.bias.device)
                        updown, ex_bias = module.calc_updown(weight)

                        if len(weight.shape) == 4 and weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))

                        self.weight.copy_((weight.to(dtype=updown.dtype) + updown).to(dtype=self.weight.dtype))
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias).to(self.weight.dtype)
                            else:
                                self.bias.copy_((bias + ex_bias).to(dtype=self.bias.dtype))
                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.in_proj_weight.chunk(3, 0)
                        updown_q, _ = module_q.calc_updown(qw)
                        updown_k, _ = module_k.calc_updown(kw)
                        updown_v, _ = module_v.calc_updown(vw)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight)

                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                    if ex_bias is not None:
                        if self.out_proj.bias is None:
                            self.out_proj.bias = torch.nn.Parameter(ex_bias)
                        else:
                            self.out_proj.bias += ex_bias

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if module is None:
                continue

            logging.debug(f"Network {net.name} layer {network_layer_name}: couldn't find supported operation")
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

        self.network_current_names = wanted_names
=======
    pass
>>>>>>> 29be1da7cf2b5dccfc70fbdd33eb35c56a31ffb7


def network_forward(org_module, input, original_forward):
    pass


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    pass


def network_Linear_forward(self, input):
    pass


def network_Linear_load_state_dict(self, *args, **kwargs):
    pass


def network_Conv2d_forward(self, input):
    pass


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    pass


def network_GroupNorm_forward(self, input):
    pass


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    pass


def network_LayerNorm_forward(self, input):
    pass


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    pass


def network_MultiheadAttention_forward(self, *args, **kwargs):
    pass


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    pass


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    for filename in candidates:
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
