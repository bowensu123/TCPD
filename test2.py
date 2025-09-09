#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU VAE Encoder Tiled Encoding Memory Sweep (CUDA)
=================================================

This script measures **peak GPU memory** for different tiling parameters when
encoding images with a VAE encoder (e.g., Diffusers AutoencoderKL) and helps you
find the **lowest-memory configuration** on CUDA GPUs.

Features
- Tests baseline (no tiling) vs tiled encoding (`encode_tiled`).
- Grid-search over tile sizes, halos, dtypes, pad modes, channels_last.
- Robust CUDA memory measurements; handles OOM and proceeds.
- Optional Diffusers integration or a DummyVAE fallback.
- Writes a CSV of results and prints the top-K configurations by peak memory.

Usage (examples)
----------------
# 1) With Diffusers VAE (local or HF model id):
python gpu_vae_mem_test.py --model-id stabilityai/sd-vae-ft-ema \
  --H 1536 --W 1536 --batch 1 --stride-total 8 --include-baseline --save-csv results.csv

# 2) With your own module exposing build_vae():
python gpu_vae_mem_test.py --vae-script /path/to/my_vae.py \
  --H 1024 --W 1024 --batch 1

# 3) Fallback DummyVAE (sanity check only):
python gpu_vae_mem_test.py --use-dummy --H 1024 --W 1024

Notes
-----
- For allocator fragmentation, you can optionally set before running:
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
- Peak memory is measured via torch.cuda; both allocated and reserved are reported.
- If you only care about **minimum memory**, start with smaller tile sizes and BF16/FP16.
"""

import argparse
import csv
import gc
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# encode_tiled implementation (CUDA)
# ------------------------------
@torch.no_grad()
def encode_tiled(
    vae,
    x: torch.Tensor,
    tile: int = 384,
    halo: int = 32,
    stride_total: int = 8,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    use_inference_mode: bool = True,
    channels_last: bool = True,
    pad_to_multiple: bool = True,
    pad_mode: str = "reflect",   # "reflect" | "replicate" | "constant"
    pad_value: float = 0.0,
    get_latents: Optional[Callable[[object], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Tile + halo VAE encoder for CUDA GPUs. Returns latents of shape
    (N, C_latent, H//s_ceil, W//s_ceil).
    """
    assert tile > 2 * halo, f"`tile` must be > 2*halo (tile={tile}, halo={halo})."
    assert x.dim() == 4, "x must be NCHW."

    # device & memory format
    if device is None:
        device = x.device if (hasattr(x, "device") and x.device.type == "cuda") else torch.device("cuda")
    x = x.to(device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    N, C, H, W = x.shape
    s = int(stride_total)
    assert s >= 1

    H_pad_mul = (s - (H % s)) % s if pad_to_multiple else 0
    W_pad_mul = (s - (W % s)) % s if pad_to_multiple else 0

    if pad_mode == "constant":
        x = F.pad(x, (0, W_pad_mul, 0, H_pad_mul), mode=pad_mode, value=pad_value)
        x = F.pad(x, (halo, halo, halo, halo), mode=pad_mode, value=pad_value)
    else:
        x = F.pad(x, (0, W_pad_mul, 0, H_pad_mul), mode=pad_mode)
        x = F.pad(x, (halo, halo, halo, halo), mode=pad_mode)

    H2, W2 = x.shape[-2], x.shape[-1]
    Hc, Wc = H + H_pad_mul, W + W_pad_mul
    central_y0, central_y1 = halo, halo + Hc
    central_x0, central_x1 = halo, halo + Wc

    def _encode_patch(patch: torch.Tensor) -> torch.Tensor:
        nonlocal vae, dtype, get_latents
        cm = torch.autocast(device_type="cuda", dtype=dtype)
        ctx = torch.inference_mode() if use_inference_mode else torch.enable_grad()
        with ctx, cm:
            out = vae.encode(patch)
            if get_latents is not None:
                z = get_latents(out)
            else:
                if hasattr(out, "latent_dist"):
                    ld = out.latent_dist
                    z = getattr(ld, "mode", lambda: None)()
                    if z is None:
                        z = getattr(ld, "mean", None)
                        if z is None:
                            z = ld.sample()
                elif isinstance(out, (tuple, list)) and len(out) >= 1:
                    z = out[0]
                else:
                    z = out
        return z

    # probe to know latent channels & dtype
    y_probe1 = min(tile, H2)
    x_probe1 = min(tile, W2)
    z_probe = _encode_patch(x[..., :y_probe1, :x_probe1])
    assert z_probe.dim() == 4, "Latent tensor must be 4D (N,C,H,W)."
    C_lat = z_probe.shape[1]

    H_out, W_out = (H + H_pad_mul) // s, (W + W_pad_mul) // s
    out = torch.empty((x.shape[0], C_lat, H_out, W_out), device=device, dtype=z_probe.dtype)
    if channels_last:
        out = out.to(memory_format=torch.channels_last)

    step = tile - 2 * halo
    for y0p in range(0, H2, step):
        y1p = min(y0p + tile, H2)
        y0v = max(y0p + halo, central_y0)
        y1v = min(y1p - halo, central_y1)
        if y1v <= y0v:
            continue
        for x0p in range(0, W2, step):
            x1p = min(x0p + tile, W2)
            x0v = max(x0p + halo, central_x0)
            x1v = min(x1p - halo, central_x1)
            if x1v <= x0v:
                continue

            patch = x[..., y0p:y1p, x0p:x1p]
            z = _encode_patch(patch)

            gy0_local = (y0v - y0p) // s
            gy1_local = (y1v - y0p) // s
            gx0_local = (x0v - x0p) // s
            gx1_local = (x1v - x0p) // s

            GY0 = (y0v - central_y0) // s
            GX0 = (x0v - central_x0) // s

            gy1_local = min(gy1_local, z.shape[-2])
            gx1_local = min(gx1_local, z.shape[-1])
            h_take = min(gy1_local - gy0_local, out.shape[-2] - GY0)
            w_take = min(gx1_local - gx0_local, out.shape[-1] - GX0)
            if h_take <= 0 or w_take <= 0:
                continue

            out[..., GY0:GY0 + h_take, GX0:GX0 + w_take] = z[..., gy0_local:gy0_local + h_take, gx0_local:gx0_local + w_take]
    return out

# ------------------------------
# Dummy VAE (fallback)
# ------------------------------
class DummyVAE(nn.Module):
    """A small conv encoder that downsamples by stride_total (default 8)."""
    def __init__(self, in_ch=3, lat_ch=4, stride_total=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(16, 256), nn.SiLU(),
            nn.Conv2d(256, lat_ch, 3, 1, 1)
        )
        self.latent_dist = None
        self.stride_total = stride_total

    def encode(self, x):
        z = self.enc(x)
        return z

# ------------------------------
# Utility: CUDA memory helpers
# ------------------------------

def cuda_empty_cache():
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

def cuda_reset_peak():
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

def cuda_max_allocated() -> Optional[int]:
    if hasattr(torch.cuda, "max_memory_allocated"):
        return int(torch.cuda.max_memory_allocated())
    return None

def cuda_max_reserved() -> Optional[int]:
    if hasattr(torch.cuda, "max_memory_reserved"):
        return int(torch.cuda.max_memory_reserved())
    return None


def cuda_synchronize():
    if hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()

# ------------------------------
# Trial definition
# ------------------------------
@dataclass
class TrialCfg:
    mode: str                 # 'baseline' | 'tiled'
    tile: Optional[int]
    halo: Optional[int]
    dtype: str                # 'bf16' | 'fp16' | 'fp32'
    channels_last: bool
    pad_mode: str
    stride_total: int

@dataclass
class TrialResult:
    mode: str
    tile: Optional[int]
    halo: Optional[int]
    dtype: str
    channels_last: bool
    pad_mode: str
    stride_total: int
    H: int
    W: int
    batch: int
    peak_alloc_MB: float
    peak_reserved_MB: float
    runtime_ms: float
    out_shape: Tuple[int, int, int, int]
    status: str  # 'ok' | 'oom' | 'error'
    error_msg: str = ""


def run_trial(vae: nn.Module, x: torch.Tensor, cfg: TrialCfg) -> TrialResult:
    assert x.device.type == "cuda", "Input must be on CUDA (GPU)."
    xin = x
    if cfg.channels_last:
        xin = xin.to(memory_format=torch.channels_last)
    else:
        xin = xin.contiguous(memory_format=torch.contiguous_format)

    if cfg.dtype == "bf16":
        dtype = torch.bfloat16
    elif cfg.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    cuda_empty_cache()
    gc.collect()
    cuda_reset_peak()

    start = time.perf_counter()
    try:
        if cfg.mode == "baseline":
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = vae.encode(xin)
                    if hasattr(out, "latent_dist"):
                        ld = out.latent_dist
                        out = getattr(ld, "mode", lambda: None)() or getattr(ld, "mean", None) or ld.sample()
                    elif isinstance(out, (tuple, list)):
                        out = out[0]
            out_shape = tuple(out.shape)
        else:
            out = encode_tiled(
                vae, xin,
                tile=cfg.tile or 384,
                halo=cfg.halo or 32,
                stride_total=cfg.stride_total,
                dtype=dtype,
                use_inference_mode=True,
                channels_last=cfg.channels_last,
                pad_to_multiple=True,
                pad_mode=cfg.pad_mode,
            )
            out_shape = tuple(out.shape)
        cuda_synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        peak_alloc = cuda_max_allocated() or 0
        peak_reserved = cuda_max_reserved() or 0
        del out
        gc.collect()
        cuda_empty_cache()
        return TrialResult(
            mode=cfg.mode,
            tile=cfg.tile,
            halo=cfg.halo,
            dtype=cfg.dtype,
            channels_last=cfg.channels_last,
            pad_mode=cfg.pad_mode,
            stride_total=cfg.stride_total,
            H=x.shape[-2], W=x.shape[-1], batch=x.shape[0],
            peak_alloc_MB=peak_alloc / 1024**2,
            peak_reserved_MB=peak_reserved / 1024**2,
            runtime_ms=elapsed,
            out_shape=out_shape,
            status="ok",
        )
    except RuntimeError as e:
        cuda_synchronize()
        peak_alloc = cuda_max_allocated() or 0
        peak_reserved = cuda_max_reserved() or 0
        gc.collect(); cuda_empty_cache()
        return TrialResult(
            mode=cfg.mode,
            tile=cfg.tile,
            halo=cfg.halo,
            dtype=cfg.dtype,
            channels_last=cfg.channels_last,
            pad_mode=cfg.pad_mode,
            stride_total=cfg.stride_total,
            H=x.shape[-2], W=x.shape[-1], batch=x.shape[0],
            peak_alloc_MB=peak_alloc / 1024**2,
            peak_reserved_MB=peak_reserved / 1024**2,
            runtime_ms=-1.0,
            out_shape=(0, 0, 0, 0),
            status="oom" if "out of memory" in str(e).lower() else "error",
            error_msg=str(e),
        )

# ------------------------------
# Build/load VAE
# ------------------------------

def load_from_script(path: str):
    spec = importlib.util.spec_from_file_location("user_vae_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_vae"):
        raise RuntimeError("Your script must define build_vae() returning a VAE.")
    return mod.build_vae()


def obtain_vae(args) -> nn.Module:
    if args.vae_script:
        vae = load_from_script(args.vae_script)
        print(f"Loaded VAE from {args.vae_script}")
        return vae
    if args.model_id and not args.use_dummy:
        try:
            from diffusers import AutoencoderKL
            if args.subfolder:
                vae = AutoencoderKL.from_pretrained(args.model_id, subfolder=args.subfolder)
            else:
                vae = AutoencoderKL.from_pretrained(args.model_id)
            print(f"Loaded Diffusers VAE: {args.model_id}")
            return vae
        except Exception as e:
            print(f"[Warn] Failed to load diffusers VAE: {e}. Falling back to DummyVAE.")
    print("Using DummyVAE (for sanity check only).")
    return DummyVAE(in_ch=args.in_ch, lat_ch=args.lat_ch, stride_total=args.stride_total)

# ------------------------------
# Argparse & grid
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CUDA VAE tiled encoding memory sweep")
    p.add_argument("--model-id", type=str, default=None, help="Diffusers model id or local path containing VAE")
    p.add_argument("--subfolder", type=str, default=None, help="Subfolder where VAE lives (if applicable)")
    p.add_argument("--vae-script", type=str, default=None, help="Path to a python file exposing build_vae() -> nn.Module")
    p.add_argument("--use-dummy", action="store_true", help="Force using DummyVAE")

    p.add_argument("--H", type=int, default=1536)
    p.add_argument("--W", type=int, default=1536)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--lat-ch", type=int, default=4)
    p.add_argument("--stride-total", type=int, default=8)

    p.add_argument("--tiles", type=str, default="256,320,384,448,512", help="Comma list of tile sizes")
    p.add_argument("--halos", type=str, default="16,24,32,40,48,64", help="Comma list of halos")
    p.add_argument("--pad-modes", type=str, default="reflect,replicate", help="Comma list: reflect,replicate,constant")
    p.add_argument("--dtypes", type=str, default="bf16,fp16", help="Comma list: bf16,fp16,fp32")
    p.add_argument("--channels-last", type=str, default="true,false", help="Comma list of booleans: true,false")

    p.add_argument("--include-baseline", action="store_true", help="Also test baseline (no tiling)")
    p.add_argument("--topk", type=int, default=10, help="Print top-K lowest-memory configs")
    p.add_argument("--save-csv", type=str, default=None, help="Save results to CSV path")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn benchmark (may change workspace/memory)")
    p.add_argument("--cudnn-deterministic", action="store_true", help="Enable cudnn deterministic (often smaller workspace)")
    return p.parse_args()

# ------------------------------
# Main
# ------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    assert torch.cuda.is_available(), "CUDA device not available."

    # Optional backend toggles
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(args.cudnn_deterministic)

    vae = obtain_vae(args)
    vae.eval()
    device = torch.device("cuda")
    vae.to(device)

    # Try channels_last for the model weights as well; ignore if not supported
    try:
        vae.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # Synthetic input
    x = torch.randn(args.batch, args.in_ch, args.H, args.W, device=device, dtype=torch.float32)

    tiles = [int(t) for t in args.tiles.split(',') if t]
    halos = [int(h) for h in args.halos.split(',') if h]
    pad_modes = [s.strip() for s in args.pad_modes.split(',') if s]
    dtypes = [s.strip() for s in args.dtypes.split(',') if s]
    chlast_opts = [(s.strip().lower() in ("1", "true", "yes", "y")) for s in args.__dict__["channels_last"].split(',')]

    trial_cfgs: List[TrialCfg] = []
    # Tiled trials
    for t in tiles:
        for h in halos:
            if t <= 2 * h:
                continue
            for pm in pad_modes:
                for dt in dtypes:
                    for cl in chlast_opts:
                        trial_cfgs.append(TrialCfg(
                            mode="tiled", tile=t, halo=h, dtype=dt,
                            channels_last=cl, pad_mode=pm, stride_total=args.stride_total
                        ))
    # Baseline trials (optional)
    if args.include_baseline:
        for dt in dtypes:
            for cl in chlast_opts:
                trial_cfgs.append(TrialCfg(
                    mode="baseline", tile=None, halo=None, dtype=dt,
                    channels_last=cl, pad_mode="n/a", stride_total=args.stride_total
                ))

    results: List[TrialResult] = []
    print(f"Total trials: {len(trial_cfgs)}")

    for i, cfg in enumerate(trial_cfgs, 1):
        tag = f"[{i}/{len(trial_cfgs)}] {cfg.mode} tile={cfg.tile} halo={cfg.halo} dtype={cfg.dtype} CL={cfg.channels_last} pad={cfg.pad_mode}"
        print(tag)
        res = run_trial(vae, x, cfg)
        results.append(res)
        if res.status != "ok":
            print(f"  -> {res.status.upper()} ({res.error_msg[:80]}...)")
        else:
            print(f"  -> peak_alloc={res.peak_alloc_MB:.1f} MB | peak_reserved={res.peak_reserved_MB:.1f} MB | time={res.runtime_ms:.1f} ms | out={res.out_shape}")

    # Sort by peak_alloc then reserved
    ok_results = [r for r in results if r.status == "ok"]
    ok_results.sort(key=lambda r: (r.peak_alloc_MB, r.peak_reserved_MB, r.runtime_ms))

    print("\n=== Top configs by minimal peak_alloc (MB) ===")
    for r in ok_results[: args.topk]:
        print(f"{r.mode:8s} tile={r.tile} halo={r.halo:>3} dtype={r.dtype:4s} CL={str(r.channels_last):5s} pad={r.pad_mode:9s} "
              f"alloc={r.peak_alloc_MB:8.1f} reserved={r.peak_reserved_MB:8.1f} time={r.runtime_ms:7.1f}ms out={r.out_shape}")

    if args.save_csv:
        fields = list(asdict(ok_results[0]).keys()) if ok_results else list(asdict(results[0]).keys())
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"Saved CSV -> {args.save_csv}")


if __name__ == "__main__":
    main()
