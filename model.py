"""
model.py -- model-agnostic loader for the residual-bus experiment.

Goals for Phase 1:
  * Load a small instruction model + tokenizer with FP32 weights.
  * Configure execution as deterministically as CUDA FP32 allows.
  * Use eager attention (transparent compute path, no fused kernels between blocks).
  * Freeze everything (eval + no grad).
  * Expose the decoder-layer list in an architecture-agnostic way so the same
    harness works for Qwen2.5-1.5B-Instruct today and Gemma-3-1B-it tomorrow.

Nothing here knows about interventions -- see intervention.py.
"""

from __future__ import annotations

# CUBLAS_WORKSPACE_CONFIG must be set BEFORE the CUDA context is created for
# torch.use_deterministic_algorithms to accept cuBLAS GEMMs. Set it at import.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SEED = 0


def configure_determinism(seed: int = DEFAULT_SEED) -> None:
    """Make execution as reproducible as FP32 CUDA permits.

    Note: this reduces run-to-run variance but does NOT guarantee bitwise
    identity across every CUDA execution path. The experiment therefore reports
    exact-equality *and* max-abs-diff / cosine similarity rather than relying on
    bitwise identity alone.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Kill the two most common sources of nondeterminism / silent precision loss.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # warn_only=True: some ops lack deterministic kernels; we prefer a warning
    # over a hard crash and capture the residual nondeterminism in telemetry.
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    model_id: str
    dtype: torch.dtype
    device: str

    @property
    def layers(self) -> torch.nn.ModuleList:
        return get_decoder_layers(self.model)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the decoder-layer ModuleList regardless of wrapper architecture.

    Handles:
      * model.model.layers                 (Qwen2, Llama, Gemma3ForCausalLM text)
      * model.model.language_model.layers  (Gemma3 multimodal wrappers)
      * model.transformer.h                (GPT-2 style, just in case)
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.transformer.h,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate decoder layers on this model. Inspect model.named_modules()."
    )


def load(
    model_id: str | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = DEFAULT_SEED,
) -> LoadedModel:
    # Allow a local path / alternate id via env (e.g. a ModelScope-downloaded dir)
    # so the harness stays model-agnostic and does not depend on the HF hub.
    if model_id is None:
        model_id = os.environ.get("RESIDUAL_BUS_MODEL", DEFAULT_MODEL)
    configure_determinism(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,  # transformers>=5 renamed torch_dtype -> dtype
        attn_implementation="eager",  # transparent path; no fused SDPA/flash
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        dtype=dtype,
        device=device,
    )


def build_inputs(lm: LoadedModel, prompt: str, system: str | None = None) -> torch.Tensor:
    """Apply the model's chat template and return input_ids on the model device.

    Using the chat template keeps generation on-distribution for the instruct
    model. For pure plumbing this only needs to be deterministic, but we do it
    correctly now so the same prompts stay valid for later semantic work.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    out = lm.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # transformers>=5 may return a BatchEncoding/dict; normalize to the id tensor.
    if hasattr(out, "input_ids"):
        input_ids = out.input_ids
    elif isinstance(out, dict):
        input_ids = out["input_ids"]
    else:
        input_ids = out
    return input_ids.to(lm.device)


@torch.no_grad()
def forward_logits(lm: LoadedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Single forward pass. Returns next-token logits at the final position.

    Shape: [batch, vocab]. No generation, no KV cache -- the cleanest possible
    causal readout for Phase 1.
    """
    out = lm.model(input_ids=input_ids, use_cache=False)
    return out.logits[:, -1, :]


def middle_layer_indices(num_layers: int, fractions=(0.4, 0.5, 0.6, 0.7)) -> list[int]:
    """Pick a band of middle-ish layers to sweep instead of assuming one.

    Extremes are avoided on purpose: very early layers wash out, very late
    layers leave too few blocks to propagate the change into the logits.
    """
    idxs = sorted({int(round(f * (num_layers - 1))) for f in fractions})
    return idxs
