# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
import argparse, os, sys, glob
import torch, torchvision
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
from torchvision.utils import make_grid
from datetime import datetime
from ldm.util import ismap
import time
import tempfile, typing
import subprocess

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

sys.path.append("latent-diffusion")

ckpt = "/root/.cache/ldm/sr/last.ckpt"

sizes = [128, 256, 384, 448, 512]


class Predictor(BasePredictor):
    def setup(self):
        subprocess.call(["pip", "install", "-e", "."])
        global config, model, global_step, device
        device = torch.device("cuda")
        config = OmegaConf.load("/src/configs/latent-diffusion/superres.yaml")
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cuda")
        global_step = pl_sd["global_step"]
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()

    def predict(
        self,
        image: Path = Input(description="Image"),
        up_f: int = Input(description="Upscale factor", default=4, choices=[2, 3, 4]),
        steps: int = Input(description="Sampling steps", default=100),
    ) -> Path:
        global config, model, global_step

        save_intermediate_vid = False
        n_runs = 1
        masked = False
        guider = None
        ckwargs = None
        mode = "ddim"
        ddim_use_x0_pred = False
        temperature = 1.0
        eta = 1.0
        make_progrow = True
        custom_shape = None
        custom_steps = steps

        c = Image.open(image).convert("RGBA")
        # Remove alpha channel if present
        bg = Image.new("RGBA", c.size, (255, 255, 255))
        c = Image.alpha_composite(bg, c).convert("RGB")
        c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
        c_up = rearrange(c_up, "1 c h w -> 1 h w c")
        c = rearrange(c, "1 c h w -> 1 h w c")
        c = 2.0 * c - 1.0
        c = c.to(torch.device("cuda"))
        batch = {"LR_image": c, "image": c_up}

        height, width = batch["image"].shape[1:3]
        split_input = height >= 128 and width >= 128

        if split_input:
            ks = 128
            stride = 64
            vqf = 4
            model.split_input_params = {
                "ks": (ks, ks),
                "stride": (stride, stride),
                "vqf": vqf,
                "patch_distributed_vq": True,
                "tie_braker": False,
                "clip_max_weight": 0.5,
                "clip_min_weight": 0.01,
                "clip_max_tie_weight": 0.5,
                "clip_min_tie_weight": 0.01,
            }
        else:
            if hasattr(model, "split_input_params"):
                delattr(model, "split_input_params")

        invert_mask = False

        x_T = None
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, "1 c h w -> b c h w", b=custom_shape[0])
        for n in trange(n_runs, desc="Sampling"):
            logs = make_convolutional_sample(
                batch,
                model,
                mode="superresolution",
                custom_steps=custom_steps,
                eta=eta,
                swap_mode=False,
                masked=masked,
                invert_mask=invert_mask,
                quantize_x0=False,
                custom_schedule=None,
                decode_interval=10,
                resize_enabled=False,
                custom_shape=custom_shape,
                temperature=temperature,
                noise_dropout=0.0,
                corrector=guider,
                corrector_kwargs=ckwargs,
                x_T=x_T,
                save_intermediate_vid=False,
                make_progrow=make_progrow,
                ddim_use_x0_pred=ddim_use_x0_pred,
            )

        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1.0, 1.0)
        sample = (sample + 1.0) / 2.0 * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        outfile = tempfile.mktemp(".png")
        a = Image.fromarray(sample[0]).save(outfile)

        return Path(outfile)


@torch.no_grad()
def convsample_ddim(
    model,
    cond,
    steps,
    shape,
    eta=1.0,
    callback=None,
    normals_sequence=None,
    mask=None,
    x0=None,
    quantize_x0=False,
    img_callback=None,
    temperature=1.0,
    noise_dropout=0.0,
    score_corrector=None,
    corrector_kwargs=None,
    x_T=None,
    log_every_t=None,
):

    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(
        steps,
        batch_size=bs,
        shape=shape,
        conditioning=cond,
        callback=callback,
        normals_sequence=normals_sequence,
        quantize_x0=quantize_x0,
        eta=eta,
        mask=mask,
        x0=x0,
        temperature=temperature,
        verbose=False,
        score_corrector=score_corrector,
        corrector_kwargs=corrector_kwargs,
        x_T=x_T,
    )

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(
    batch,
    model,
    mode="vanilla",
    custom_steps=None,
    eta=1.0,
    swap_mode=False,
    masked=False,
    invert_mask=True,
    quantize_x0=False,
    custom_schedule=None,
    decode_interval=1000,
    resize_enabled=False,
    custom_shape=None,
    temperature=1.0,
    noise_dropout=0.0,
    corrector=None,
    corrector_kwargs=None,
    x_T=None,
    save_intermediate_vid=False,
    make_progrow=True,
    ddim_use_x0_pred=False,
):
    log = dict()

    z, c, x, xrec, xc = model.get_input(
        batch,
        model.first_stage_key,
        return_first_stage_outputs=True,
        force_c_encode=not (hasattr(model, "split_input_params") and model.cond_stage_key == "coordinates_bbox"),
        return_original_cond=True,
    )

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, "cond_stage_key"):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key == "class_label":
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(
            model,
            c,
            steps=custom_steps,
            shape=z.shape,
            eta=eta,
            quantize_x0=quantize_x0,
            img_callback=img_cb,
            mask=None,
            x0=z0,
            temperature=temperature,
            noise_dropout=noise_dropout,
            score_corrector=corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
        )
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates["pred_x0"][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log
