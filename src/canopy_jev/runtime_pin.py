# SPDX-License-Identifier: MIT
"""Process-local FLA l2norm pin; call before the first model/head forward.

Only the existing forward/backward D<=512 autotuner config lists and their
selection caches are changed. No torch flags, environment or library files
are changed. Full-model repeatability is a separate validation requirement.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import threading

SCHEMA = "jev.l2norm_runtime_pin.v1"
SPEC = {"BT": 16, "num_warps": 8, "num_stages": 2}
EXPECTED_SOURCES = {
    "fla.modules.l2norm": "30f7feebdbfa87b8e90c143ec62a7fa0781bfd736c856c0ec69452f0c778be3f",
    "fla.ops.utils.cache": "a64b09ffd3f51aed547ad72559dee94cb9aeafd2a5f02c6c8e60a6e020147a9c",
    "triton.runtime.autotuner": "0497baf6ff4c48c4b3764bd84940ebbd7ac015e2d9d71e80fbb6949a9cf62ded",
}
ENV_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUDA_MODULE_LOADING",
    "FLA_CACHE_MODE", "FLA_CONFIG_DIR", "FLA_CACHE_RESULTS", "FLA_TILELANG",
    "FLA_USE_TMA", "FLA_DISABLE_TENSOR_CACHE", "FLA_TENSOR_CACHE_SIZE",
    "TRITON_CACHE_DIR", "TRITON_PRINT_AUTOTUNING",
    "TORCH_CUDNN_V8_API_DISABLED", "CUDNN_CONV_WSCAP_DBG",
)
_LOCK = threading.RLock()
_STATE = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _config(config):
    if config.pre_hook is not None:
        raise RuntimeError("Unexpected l2norm Config pre_hook")
    return {"kwargs": dict(config.kwargs), "all_kwargs": dict(config.all_kwargs())}


def _cache(tuner):
    entries = [{"key": repr(k), "config": _config(v)} for k, v in tuner.cache.items()]
    return {"count": len(entries), "entries": sorted(entries, key=lambda x: x["key"])}


def _select(tuner, spec):
    if list(tuner.keys) != ["D", "NB"] or not isinstance(tuner.cache, dict):
        raise RuntimeError("Unexpected l2norm autotuner structure")
    matches = [c for c in tuner.configs if dict(c.kwargs) == {"BT": spec["BT"]}
               and c.num_warps == spec["num_warps"] and c.num_stages == spec["num_stages"]]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one existing config {spec}, found {len(matches)}")
    _config(matches[0])
    return matches[0]


def _prepare(tuners, spec):
    """Validate every target before mutating any target."""
    return {name: {"tuner": tuner, "selected": _select(tuner, spec),
                   "before": {"configs": [_config(c) for c in tuner.configs],
                              "cache": _cache(tuner), "key_names": list(tuner.keys)}}
            for name, tuner in tuners.items()}


def _apply(prepared):
    for item in prepared.values():
        tuner, selected = item["tuner"], item["selected"]
        tuner.configs = [selected]  # Reuse the actual original Config object.
        tuner.cache.clear()
        tuner.best_config = selected
        if hasattr(tuner, "configs_timings"):
            tuner.configs_timings = {}


def _assert_targets(prepared):
    for name, item in prepared.items():
        tuner, selected = item["tuner"], item["selected"]
        if len(tuner.configs) != 1 or tuner.configs[0] is not selected:
            raise RuntimeError(f"Runtime pin overwritten: {name}.configs")
        if _config(selected) != item["selected_record"]:
            raise RuntimeError(f"Runtime pin Config mutated: {name}")
        if list(tuner.keys) != ["D", "NB"]:
            raise RuntimeError(f"Runtime pin key changed: {name}")
        if any(_config(c) != item["selected_record"] for c in tuner.cache.values()):
            raise RuntimeError(f"Conflicting runtime selection cache: {name}")


def _runtime_snapshot():
    import torch
    import triton
    return {
        "versions": {"torch": torch.__version__, "torch_git": torch.version.git_version,
                     "cuda_toolkit": torch.version.cuda, "triton": triton.__version__,
                     "fla": importlib.metadata.version("flash-linear-attention")},
        "flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        },
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }


def _source_identity(modules):
    result = {}
    for name, module in modules.items():
        path = Path(inspect.getsourcefile(module)).resolve()
        digest = _sha(path)
        if digest != EXPECTED_SOURCES[name]:
            raise RuntimeError(f"Unreviewed installed source for {name}: {digest}")
        result[name] = {"path": str(path), "sha256": digest}
    return result


def _require_disabled_cache(module):
    if module.FLA_CACHE_MODE.value != "disabled" or os.environ.get("FLA_CACHE_MODE", "disabled") != "disabled":
        raise RuntimeError("FLA_CACHE_MODE must already be disabled; runtime_pin does not change it")


def pin_l2norm_runtime(*, bt=16, num_warps=8, num_stages=2):
    """Pin both existing l2norm forward/backward configs; return provenance.

    The only accepted policy is BT16/W8/stages2. Repeating the same call is
    idempotent and checks for tampering. The backward configuration exists in
    the installed grid; its full training-path numeric validation is external.
    """
    global _STATE
    requested = {"BT": bt, "num_warps": num_warps, "num_stages": num_stages}
    if requested != SPEC:
        raise RuntimeError(f"Unsupported/conflicting runtime pin: {requested}; expected {SPEC}")
    with _LOCK:
        if _STATE is not None:
            return runtime_contract()
        # Lazy import: importing runtime_pin itself neither loads FLA nor uses CUDA.
        modules = {name: importlib.import_module(name) for name in EXPECTED_SOURCES}
        source_identity = _source_identity(modules)
        cache_module = modules["fla.ops.utils.cache"]
        _require_disabled_cache(cache_module)
        module = modules["fla.modules.l2norm"]
        tuners = {name: getattr(module, name) for name in ("l2norm_fwd_kernel", "l2norm_bwd_kernel")}
        before_runtime = _runtime_snapshot()
        prepared = _prepare(tuners, requested)
        for item in prepared.values():
            item["selected_record"] = _config(item["selected"])
            jit = item["tuner"].fn
            item["jit_object"] = jit
            item["jit_source_sha256"] = hashlib.sha256(jit.src.encode()).hexdigest()
        _apply(prepared)
        if _runtime_snapshot() != before_runtime:
            raise RuntimeError("Unexpected global runtime change during pin")
        _STATE = {"targets": prepared, "modules": modules, "sources": source_identity,
                  "runtime_at_pin": before_runtime, "self_sha256": _sha(__file__)}
        return runtime_contract()


def assert_runtime_pin():
    """Fail on conflicting config, key, cache, disk source or target replacement."""
    with _LOCK:
        if _STATE is None:
            raise RuntimeError("Call pin_l2norm_runtime before model/head first forward")
        _require_disabled_cache(_STATE["modules"]["fla.ops.utils.cache"])
        _assert_targets(_STATE["targets"])
        module = _STATE["modules"]["fla.modules.l2norm"]
        for name, item in _STATE["targets"].items():
            if getattr(module, name) is not item["tuner"]:
                raise RuntimeError(f"Pinned kernel replaced: {name}")
            jit = item["tuner"].fn
            if jit is not item["jit_object"] or hashlib.sha256(jit.src.encode()).hexdigest() != item["jit_source_sha256"]:
                raise RuntimeError(f"Pinned JIT source replaced: {name}")
        if _source_identity(_STATE["modules"]) != _STATE["sources"] or _sha(__file__) != _STATE["self_sha256"]:
            raise RuntimeError("Runtime pin source identity changed")


def runtime_contract():
    """JSON-ready manifest entry; call again after the last forward/backward."""
    with _LOCK:
        assert_runtime_pin()
        targets = {}
        for name, item in _STATE["targets"].items():
            targets[name] = {"before": item["before"], "selected": item["selected_record"],
                             "after_configs": [_config(c) for c in item["tuner"].configs],
                             "after_cache": _cache(item["tuner"]),
                             "key_names": list(item["tuner"].keys),
                             "jit_source_sha256": item["jit_source_sha256"]}
        current = _runtime_snapshot()
        identity = {"schema": SCHEMA, "policy": SPEC, "runtime_pin_sha256": _STATE["self_sha256"],
                    "sources": {k: v["sha256"] for k, v in _STATE["sources"].items()},
                    "runtime": current,
                    "targets": {k: {"config": v["selected"], "key_names": v["key_names"],
                                     "jit_source_sha256": v["jit_source_sha256"]} for k, v in targets.items()}}
        return {"schema": SCHEMA, "contract_sha256": _digest(identity), "identity": identity,
                "source_files": _STATE["sources"], "targets": targets,
                "runtime_at_pin": _STATE["runtime_at_pin"],
                "runtime_changed_since_pin": current != _STATE["runtime_at_pin"],
                "scope": "Process-local D<=512 l2norm fwd/bwd only; larger-D kernel1 and every other kernel unchanged",
                "validation": "Structural pin only; no assertion of full-model numerical repeatability",
                "integration": "Call before any model/head forward; attach contract at start/end; do not combine artifacts with different contract_sha256"}
