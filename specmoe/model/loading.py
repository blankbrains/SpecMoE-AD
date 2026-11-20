"""Strict M2B-MoE v3 checkpoint loading."""

from __future__ import annotations

import hashlib
import math
import pickle
import random
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import torch

from specmoe.data.preprocess import StandardScaler
from specmoe.data.protocol import (
    OFFICIAL_PROTOCOL_VERSION, PROTOCOL_VERSION, PAPER_PROTOCOL_VERSION,
    validate_paper_manifest,
)
from specmoe.model.m2b_moe_model import M2BMoEModel
from specmoe.train.trainer import (
    FIXED_VALIDATION_COMPONENT_WEIGHTS,
    FIXED_VALIDATION_OBJECTIVE_REVISION,
)


ARCHITECTURE = "m2b_moe_v3"

REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "rng_state",
        "model_config",
        "scaler_state",
        "calibrated_threshold",
        "resolved_config",
        "reproducibility",
        "seed",
        "git_revision",
        "source_fingerprint",
        "dataloader_generator_state",
        "checkpoint_kind",
        "trainer_state",
    }
)

REQUIRED_RESOLVED_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "git_revision",
        "source_fingerprint",
        "device",
        "dataset",
        "data_protocol",
        "model",
        "optimizer",
        "scheduler",
        "training",
        "validation",
        "calibration",
    }
)

_SOURCE_FINGERPRINT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "cache",
        "checkpoints",
        "docs",
        "env",
        "results",
        "tests",
        "venv",
    }
)


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_excluded_source_path(relative_path: Path) -> bool:
    excluded = {
        directory.casefold()
        for directory in _SOURCE_FINGERPRINT_EXCLUDED_DIRECTORIES
    }
    return any(part.casefold() in excluded for part in relative_path.parts)


def _source_fingerprint_paths(repository_root: Path) -> tuple[Path, ...]:
    candidates = set()
    for path in repository_root.rglob("*.py"):
        relative = path.relative_to(repository_root)
        if path.is_file() and not _is_excluded_source_path(relative):
            candidates.add(relative)

    config_root = repository_root / "configs"
    if config_root.is_dir():
        for path in config_root.glob("*.yaml"):
            if path.is_file():
                candidates.add(path.relative_to(repository_root))

    for relative in (
        Path("requirements.txt"),
        Path("run_specmoe_full.sh"),
    ):
        if (repository_root / relative).is_file():
            candidates.add(relative)
    return tuple(sorted(candidates, key=lambda path: path.as_posix()))


def compute_source_fingerprint(
    repository_root: str | Path | None = None,
) -> str:
    """Hash all runtime source and experiment inputs deterministically."""
    root = (
        _default_repository_root()
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    if not root.is_dir():
        raise ValueError(f"Repository root is not a directory: {root}")
    relative_paths = _source_fingerprint_paths(root)
    if not relative_paths:
        raise ValueError(
            f"Repository root contains no fingerprinted source files: {root}"
        )

    digest = hashlib.sha256()
    digest.update(b"SpecMoE-AD-source-fingerprint-v1\0")
    for relative_path in relative_paths:
        encoded_path = relative_path.as_posix().encode("utf-8")
        content = (root / relative_path).read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_source_fingerprint(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"Checkpoint field '{field_name}' must be a lowercase "
            "SHA256 hex digest"
        )
    return value

REQUIRED_MODEL_CONFIG_KEYS = frozenset(
    {
        "architecture",
        "time_len",
        "num_vars",
        "patch_len",
        "patch_stride",
        "d_model",
        "enc_hidden",
        "bottleneck_dims",
        "cross_var_heads",
        "use_revin",
        "input_mask_ratio",
        "mask_mode",
        "lambda_freq",
        "router_hidden",
        "router_mode",
        "expert_specialization",
        "diversity_weight",
        "load_balance_weight",
        "grad_weight",
        "score_freq_weight",
        "disagreement_weight",
        "diversity_mode",
        "use_contrastive_div",
        "use_grad_score",
        "use_adaptive_patches",
        "use_learnable_temp",
        "stft_window_sizes",
        "stft_hop_ratio",
        "frequency_score_aggregation",
    }
)

REQUIRED_REPRODUCIBILITY_KEYS = frozenset(
    {
        "seed",
        "validation_seed",
        "validation_rounds",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
)

REQUIRED_RNG_STATE_KEYS = frozenset(
    {"python", "numpy", "torch", "cuda"}
)

INTEGER_MODEL_CONFIG_KEYS = frozenset(
    {
        "time_len",
        "num_vars",
        "patch_len",
        "patch_stride",
        "d_model",
        "enc_hidden",
        "cross_var_heads",
        "router_hidden",
    }
)

FLOAT_MODEL_CONFIG_KEYS = frozenset(
    {
        "input_mask_ratio",
        "lambda_freq",
        "diversity_weight",
        "load_balance_weight",
        "grad_weight",
        "score_freq_weight",
        "disagreement_weight",
        "stft_hop_ratio",
    }
)

BOOLEAN_MODEL_CONFIG_KEYS = frozenset(
    {
        "use_revin",
        "use_contrastive_div",
        "use_grad_score",
        "use_adaptive_patches",
        "use_learnable_temp",
    }
)

REQUIRED_PLATEAU_STATE_KEYS = frozenset(
    {
        "factor",
        "min_lrs",
        "patience",
        "cooldown",
        "eps",
        "last_epoch",
        "mode_worse",
        "mode",
        "threshold",
        "threshold_mode",
        "best",
        "cooldown_counter",
        "num_bad_epochs",
    }
)


def _validate_weights_only_tree(value: Any, path: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        return
    if type(value) in {str, int, float, bool, type(None)}:
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_weights_only_tree(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            _validate_weights_only_tree(key, f"{path}.<key>")
            _validate_weights_only_tree(child, f"{path}.{key}")
        return
    raise ValueError(
        f"Unsafe checkpoint value at {path}: {type(value).__name__}; "
        "only tensors, primitives, lists, and dictionaries are accepted"
    )


def _list_tree_to_tuple(value: Any) -> Any:
    if type(value) is list:
        return tuple(_list_tree_to_tuple(child) for child in value)
    return value


def _require_mapping(
    container: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint field '{key}' must be a mapping")
    return value


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            set(left) == set(right)
            and all(
                _typed_equal(left[key], right[key]) for key in left
            )
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def _require_integer(
    name: str, value: Any, *, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Checkpoint field '{name}' must be an integer >= {minimum}"
        )
    return value


def _require_exact_keys(
    name: str,
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(
            f"Checkpoint field '{name}' schema mismatch: "
            f"missing={missing}, extra={extra}"
        )


def _finite_number(name: str, value: Any, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(
            f"Checkpoint field '{name}' must be finite and >= {minimum}"
        )
    return float(value)


def _positive_resolved_integer(name: str, value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(
            f"Checkpoint field '{name}' must be a positive integer"
        )
    return value


def _validate_dataloader_generator_state(value: Any) -> None:
    expected_length = torch.Generator(device="cpu").get_state().numel()
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() != expected_length
    ):
        raise ValueError(
            "Checkpoint field 'dataloader_generator_state' must be a "
            f"1D uint8 tensor of length {expected_length}"
        )
    try:
        torch.Generator(device="cpu").set_state(value.detach().cpu())
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint field 'dataloader_generator_state' is not "
            "restorable"
        ) from exc


def _validate_integer_list(
    config: Mapping[str, Any], key: str
) -> None:
    value = config[key]
    if type(value) is not list or not value:
        raise ValueError(
            f"Checkpoint model_config field '{key}' must be a non-empty list"
        )
    if any(type(item) is not int for item in value):
        raise ValueError(
            f"Checkpoint model_config field '{key}' must contain integers"
        )


def _validate_model_state_dict(
    model: M2BMoEModel,
    model_state: Mapping[str, Any],
) -> None:
    expected_state = model.state_dict()
    actual_keys = set(model_state)
    expected_keys = set(expected_state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Checkpoint model_state_dict keys do not match the model: "
            f"missing={missing}, extra={extra}"
        )
    for key, expected in expected_state.items():
        actual = model_state[key]
        if not isinstance(actual, torch.Tensor):
            raise ValueError(
                f"Checkpoint model_state_dict field '{key}' must be a tensor"
            )
        if actual.shape != expected.shape:
            raise ValueError(
                f"Checkpoint model_state_dict field '{key}' shape "
                f"{tuple(actual.shape)} does not match {tuple(expected.shape)}"
            )
        if actual.dtype != expected.dtype:
            raise ValueError(
                f"Checkpoint model_state_dict field '{key}' dtype "
                f"{actual.dtype} does not match {expected.dtype}"
            )
        if (actual.is_floating_point() or actual.is_complex()) and not bool(
            torch.isfinite(actual).all()
        ):
            raise ValueError(
                f"Checkpoint model_state_dict field '{key}' contains "
                "non-finite NaN/Inf values"
            )


def _validate_model_config(config: Mapping[str, Any]) -> None:
    architecture = config.get("architecture")
    if architecture == "m2b_moe_v2":
        raise ValueError(
            "m2b_moe_v2 checkpoints are incompatible with m2b_moe_v3 "
            "and must be retrained"
        )
    if architecture != ARCHITECTURE:
        raise ValueError(
            f"Unsupported checkpoint architecture {architecture!r}; "
            f"only {ARCHITECTURE} is accepted"
        )
    actual_keys = set(config)
    missing = sorted(REQUIRED_MODEL_CONFIG_KEYS - actual_keys)
    if missing:
        raise ValueError(
            "Checkpoint model_config is incomplete; missing: "
            + ", ".join(missing)
        )
    extra = sorted(actual_keys - REQUIRED_MODEL_CONFIG_KEYS)
    if extra:
        raise ValueError(
            "Checkpoint model_config contains unsupported fields: "
            + ", ".join(extra)
        )
    for key in INTEGER_MODEL_CONFIG_KEYS:
        if type(config[key]) is not int:
            raise ValueError(
                f"Checkpoint model_config field '{key}' must be an integer"
            )
    for key in FLOAT_MODEL_CONFIG_KEYS:
        if type(config[key]) is not float:
            raise ValueError(
                f"Checkpoint model_config field '{key}' must be a float"
            )
        if not bool(torch.isfinite(torch.tensor(config[key]))):
            raise ValueError(
                f"Checkpoint model_config field '{key}' must be finite"
            )
    for key in BOOLEAN_MODEL_CONFIG_KEYS:
        if type(config[key]) is not bool:
            raise ValueError(
                f"Checkpoint model_config field '{key}' must be a boolean"
            )
    _validate_integer_list(config, "bottleneck_dims")
    _validate_integer_list(config, "stft_window_sizes")
    if type(config["diversity_mode"]) is not str:
        raise ValueError(
            "Checkpoint model_config field 'diversity_mode' must be a string"
        )
    if config["diversity_mode"] not in {
        "none",
        "cosine",
        "contrastive",
    }:
        raise ValueError(
            "Checkpoint model_config field 'diversity_mode' is invalid"
        )
    expected_legacy_flag = config["diversity_mode"] == "contrastive"
    if config["use_contrastive_div"] != expected_legacy_flag:
        raise ValueError(
            "Checkpoint diversity_mode conflicts with use_contrastive_div"
        )
    if config["router_mode"] not in {"global", "variable"}:
        raise ValueError(
            "Checkpoint model_config field 'router_mode' is invalid"
        )
    if config["expert_specialization"] not in {
        "none",
        "frequency_bands",
    }:
        raise ValueError(
            "Checkpoint model_config field 'expert_specialization' is invalid"
        )
    if config["mask_mode"] not in {"random", "contiguous"}:
        raise ValueError(
            "Checkpoint model_config field 'mask_mode' is invalid"
        )
    if config["frequency_score_aggregation"] not in {"mean", "max"}:
        raise ValueError(
            "Checkpoint model_config field "
            "'frequency_score_aggregation' is invalid"
        )


def _validate_rng_state(rng_state: Mapping[str, Any]) -> None:
    actual_keys = set(rng_state)
    if actual_keys != REQUIRED_RNG_STATE_KEYS:
        missing = sorted(REQUIRED_RNG_STATE_KEYS - actual_keys)
        extra = sorted(actual_keys - REQUIRED_RNG_STATE_KEYS)
        raise ValueError(
            "Checkpoint field 'rng_state' schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    python_state = rng_state["python"]
    if type(python_state) is not list or len(python_state) != 3:
        raise ValueError(
            "Checkpoint field 'rng_state.python' must be a Python RNG "
            "list of length 3"
        )
    try:
        random.Random().setstate(_list_tree_to_tuple(python_state))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint field 'rng_state.python' is not restorable"
        ) from exc

    numpy_state = rng_state["numpy"]
    if type(numpy_state) is not list or len(numpy_state) != 5:
        raise ValueError(
            "Checkpoint field 'rng_state.numpy' must be a NumPy RNG "
            "list of length 5"
        )
    if type(numpy_state[0]) is not str:
        raise ValueError(
            "Checkpoint field 'rng_state.numpy[0]' must be a string"
        )
    numpy_keys = numpy_state[1]
    if (
        not isinstance(numpy_keys, torch.Tensor)
        or numpy_keys.ndim != 1
        or numpy_keys.dtype not in {torch.int32, torch.int64}
        or numpy_keys.numel() == 0
        or not bool((numpy_keys >= 0).all())
        or not bool((numpy_keys <= 2**32 - 1).all())
    ):
        raise ValueError(
            "Checkpoint field 'rng_state.numpy[1]' must be a valid "
            "integer state tensor"
        )
    if type(numpy_state[2]) is not int or type(numpy_state[3]) is not int:
        raise ValueError(
            "Checkpoint NumPy RNG position and Gaussian flag must be integers"
        )
    if (
        isinstance(numpy_state[4], bool)
        or not isinstance(numpy_state[4], (int, float))
        or not math.isfinite(float(numpy_state[4]))
    ):
        raise ValueError(
            "Checkpoint NumPy RNG cached Gaussian value must be finite"
        )
    restored_numpy_state = (
        numpy_state[0],
        numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
        numpy_state[2],
        numpy_state[3],
        float(numpy_state[4]),
    )
    try:
        np.random.RandomState().set_state(restored_numpy_state)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint field 'rng_state.numpy' is not restorable"
        ) from exc

    torch_state = rng_state["torch"]
    expected_torch_rng_length = torch.get_rng_state().numel()
    if (
        not isinstance(torch_state, torch.Tensor)
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
        or torch_state.numel() != expected_torch_rng_length
    ):
        raise ValueError(
            "Checkpoint field 'rng_state.torch' must be a 1D uint8 "
            f"tensor of length {expected_torch_rng_length}"
        )
    cuda_state = rng_state["cuda"]
    if type(cuda_state) is not list or any(
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or state.ndim != 1
        or state.numel() == 0
        for state in cuda_state
    ):
        raise ValueError(
            "Checkpoint field 'rng_state.cuda' must be a list of "
            "non-empty 1D uint8 tensors"
        )


def _validate_scheduler_state(
    scheduler_state: Mapping[str, Any],
    optimizer_param_groups: list,
) -> None:
    num_param_groups = len(optimizer_param_groups)
    missing = sorted(REQUIRED_PLATEAU_STATE_KEYS - set(scheduler_state))
    if missing:
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict' is missing "
            "ReduceLROnPlateau core state: " + ", ".join(missing)
        )
    if scheduler_state["mode"] not in {"min", "max"}:
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict.mode' is invalid"
        )
    if scheduler_state["threshold_mode"] not in {"rel", "abs"}:
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict.threshold_mode' is invalid"
        )
    factor = scheduler_state["factor"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(float(factor))
        or not 0.0 < float(factor) < 1.0
    ):
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict.factor' must be "
            "finite and satisfy 0 < factor < 1"
        )
    for key in ("min_lrs",):
        value = scheduler_state[key]
        if type(value) is not list or len(value) != num_param_groups:
            raise ValueError(
                f"Checkpoint field 'scheduler_state_dict.{key}' must "
                f"contain {num_param_groups} values"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in value
        ):
            raise ValueError(
                f"Checkpoint field 'scheduler_state_dict.{key}' must "
                "contain finite non-negative values"
            )
    for key in ("eps", "threshold"):
        value = scheduler_state[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"Checkpoint field 'scheduler_state_dict.{key}' must "
                "be finite and non-negative"
            )
    legal_initial = (
        float("inf")
        if scheduler_state["mode"] == "min"
        else float("-inf")
    )
    mode_worse = scheduler_state["mode_worse"]
    if (
        isinstance(mode_worse, bool)
        or not isinstance(mode_worse, (int, float))
        or float(mode_worse) != legal_initial
    ):
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict.mode_worse' must "
            f"equal {legal_initial!r} for mode={scheduler_state['mode']}"
        )
    best = scheduler_state["best"]
    if isinstance(best, bool) or not isinstance(best, (int, float)):
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict.best' must be numeric"
        )
    numeric_best = float(best)
    if not math.isfinite(numeric_best) and numeric_best != legal_initial:
        raise ValueError(
            "Checkpoint field 'scheduler_state_dict.best' must be finite "
            "or the legal initial value for its mode"
        )
    if "_last_lr" in scheduler_state:
        last_lrs = scheduler_state["_last_lr"]
        if type(last_lrs) is not list or len(last_lrs) != num_param_groups:
            raise ValueError(
                "Checkpoint field 'scheduler_state_dict._last_lr' must "
                f"contain {num_param_groups} values"
            )
        for group_index, (last_lr, optimizer_group) in enumerate(
            zip(last_lrs, optimizer_param_groups)
        ):
            if (
                isinstance(last_lr, bool)
                or not isinstance(last_lr, (int, float))
                or not math.isfinite(float(last_lr))
                or float(last_lr) < 0.0
            ):
                raise ValueError(
                    "Checkpoint field 'scheduler_state_dict._last_lr' must "
                    "contain finite non-negative values"
                )
            optimizer_lr = optimizer_group.get("lr")
            if (
                isinstance(optimizer_lr, bool)
                or not isinstance(optimizer_lr, (int, float))
                or not math.isfinite(float(optimizer_lr))
                or float(optimizer_lr) < 0.0
            ):
                raise ValueError(
                    "Checkpoint optimizer_state_dict param_group lr must "
                    "be finite and non-negative"
                )
            if not math.isclose(
                float(last_lr),
                float(optimizer_lr),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "Checkpoint scheduler_state_dict _last_lr does not match "
                    f"optimizer param_group lr at index {group_index}"
                )
    for key in (
        "patience",
        "cooldown",
        "last_epoch",
        "cooldown_counter",
        "num_bad_epochs",
    ):
        if (
            type(scheduler_state[key]) is not int
            or scheduler_state[key] < 0
        ):
            raise ValueError(
                f"Checkpoint field 'scheduler_state_dict.{key}' "
                "must be a non-negative integer"
            )


def _iter_state_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_state_tensors(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_state_tensors(child)


def _validate_adamw_parameter_state(
    parameter_id: int,
    state: Mapping[str, Any],
    parameter: torch.nn.Parameter,
    *,
    amsgrad: bool,
) -> None:
    for tensor in _iter_state_tensors(state):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                "Checkpoint optimizer_state_dict contains a non-finite "
                f"state tensor for parameter {parameter_id}"
            )

    step = state.get("step")
    if isinstance(step, torch.Tensor):
        if (
            step.numel() != 1
            or not bool(torch.isfinite(step).all())
            or float(step.item()) < 0.0
        ):
            raise ValueError(
                "Checkpoint optimizer_state_dict step must be a finite "
                "non-negative scalar"
            )
    elif (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(float(step))
        or float(step) < 0.0
    ):
        raise ValueError(
            "Checkpoint optimizer_state_dict step must be a finite "
            "non-negative scalar"
        )

    required_moments = ["exp_avg", "exp_avg_sq"]
    if amsgrad:
        required_moments.append("max_exp_avg_sq")
    for key in required_moments:
        if key not in state:
            raise ValueError(
                f"Checkpoint optimizer_state_dict is missing {key}"
            )
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key not in state:
            continue
        moment = state[key]
        if not isinstance(moment, torch.Tensor):
            raise ValueError(
                f"Checkpoint optimizer_state_dict {key} must be a tensor"
            )
        if moment.shape != parameter.shape:
            raise ValueError(
                f"Checkpoint optimizer_state_dict {key} shape "
                f"{tuple(moment.shape)} does not match parameter shape "
                f"{tuple(parameter.shape)}"
            )
        if moment.dtype != parameter.dtype:
            raise ValueError(
                f"Checkpoint optimizer_state_dict {key} dtype "
                f"{moment.dtype} does not match parameter dtype "
                f"{parameter.dtype}"
            )


def _validate_adamw_group(group: Mapping[str, Any]) -> None:
    for key in ("lr", "weight_decay"):
        if key not in group:
            raise ValueError(
                f"Checkpoint optimizer_state_dict is missing {key}"
            )
        value = group[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"Checkpoint optimizer_state_dict {key} must be finite "
                "and non-negative"
            )
    eps = group.get("eps")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        raise ValueError(
            "Checkpoint optimizer_state_dict eps must be finite and positive"
        )
    betas = group.get("betas")
    if (
        type(betas) is not list
        or len(betas) != 2
        or any(
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not math.isfinite(float(beta))
            or not 0.0 <= float(beta) < 1.0
            for beta in betas
        )
    ):
        raise ValueError(
            "Checkpoint optimizer_state_dict betas must be a list of "
            "two finite values in [0, 1)"
        )
    for key in (
        "amsgrad",
        "maximize",
        "capturable",
        "differentiable",
    ):
        if type(group.get(key)) is not bool:
            raise ValueError(
                f"Checkpoint optimizer_state_dict {key} must be boolean"
            )
    for key in ("fused", "foreach"):
        if key not in group:
            raise ValueError(
                f"Checkpoint optimizer_state_dict is missing {key}"
            )
        if group.get(key) is not None and type(group.get(key)) is not bool:
            raise ValueError(
                f"Checkpoint optimizer_state_dict {key} must be boolean or null"
            )
    if (
        "decoupled_weight_decay" in group
        and type(group["decoupled_weight_decay"]) is not bool
    ):
        raise ValueError(
            "Checkpoint optimizer_state_dict decoupled_weight_decay "
            "must be boolean"
        )


def _restore_optimizer_scheduler(
    model: M2BMoEModel,
    checkpoint: Mapping[str, Any],
) -> Tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]:
    optimizer_state = checkpoint["optimizer_state_dict"]
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError(
            "Checkpoint field 'optimizer_state_dict' must contain exactly "
            "'state' and 'param_groups'"
        )
    if not isinstance(optimizer_state["state"], Mapping):
        raise ValueError(
            "Checkpoint field 'optimizer_state_dict.state' must be a mapping"
        )
    param_groups = optimizer_state["param_groups"]
    if type(param_groups) is not list or not param_groups:
        raise ValueError(
            "Checkpoint field 'optimizer_state_dict.param_groups' "
            "must be a non-empty list"
        )
    if len(param_groups) != 1:
        raise ValueError(
            "Checkpoint optimizer_state_dict param_groups are incompatible "
            "with the model's single AdamW parameter group"
        )
    group = param_groups[0]
    if not isinstance(group, Mapping):
        raise ValueError(
            "Checkpoint optimizer_state_dict param_groups must be mappings"
        )
    _validate_adamw_group(group)
    parameter_ids = group.get("params")
    if type(parameter_ids) is not list or not parameter_ids:
        raise ValueError(
            "Checkpoint optimizer_state_dict param_groups params must "
            "be a non-empty list"
        )
    if any(type(parameter_id) is not int for parameter_id in parameter_ids):
        raise ValueError(
            "Checkpoint optimizer_state_dict params must contain integers"
        )
    model_parameter_count = len(list(model.parameters()))
    if (
        len(parameter_ids) != model_parameter_count
        or len(set(parameter_ids)) != model_parameter_count
    ):
        raise ValueError(
            "Checkpoint optimizer_state_dict params are incompatible with "
            f"the model's {model_parameter_count} parameters"
        )
    optimizer_state_parameter_ids = set(optimizer_state["state"])
    expected_parameter_ids = set(parameter_ids)
    if (
        checkpoint["epoch"] > 0
        and optimizer_state_parameter_ids != expected_parameter_ids
    ):
        raise ValueError(
            "Checkpoint optimizer_state_dict state parameter IDs must "
            "exactly match param_groups params when epoch is positive"
        )
    model_parameters = list(model.parameters())
    parameter_by_id = dict(zip(parameter_ids, model_parameters))
    amsgrad = group.get("amsgrad", False)
    if type(amsgrad) is not bool:
        raise ValueError(
            "Checkpoint optimizer_state_dict amsgrad must be boolean"
        )
    for parameter_id, parameter_state in optimizer_state["state"].items():
        if type(parameter_id) is not int or not isinstance(
            parameter_state, Mapping
        ):
            raise ValueError(
                "Checkpoint optimizer_state_dict state entries are invalid"
            )
        _validate_adamw_parameter_state(
            parameter_id,
            parameter_state,
            parameter_by_id[parameter_id],
            amsgrad=amsgrad,
        )

    scheduler_state = checkpoint["scheduler_state_dict"]
    _validate_scheduler_state(scheduler_state, param_groups)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    try:
        scheduler.load_state_dict(dict(scheduler_state))
        optimizer.load_state_dict(dict(optimizer_state))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "Checkpoint optimizer_state_dict or scheduler_state_dict "
            "cannot be restored"
        ) from exc
    return optimizer, scheduler


def _validate_entity_split_manifest(
    manifest: Any,
    *,
    seed: int,
    window_size: int,
) -> int:
    """Validate the checkpoint-only shape of the 60/10/30 split manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "Checkpoint unified entity-split manifest must be a mapping"
        )
    _require_exact_keys(
        "resolved_config.data_protocol.manifest",
        manifest,
        {
            "protocol_version",
            "seed",
            "train_fraction",
            "tail_block_count",
            "entities",
        },
    )
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(
            "Checkpoint unified entity-split manifest has an unsupported version"
        )
    if manifest["seed"] != seed:
        raise ValueError(
            "Checkpoint unified entity-split manifest seed must match checkpoint seed"
        )
    if manifest["train_fraction"] != [3, 5]:
        raise ValueError(
            "Checkpoint unified entity-split manifest train_fraction must be [3, 5]"
        )
    if manifest["tail_block_count"] != 4:
        raise ValueError(
            "Checkpoint unified entity-split manifest tail_block_count must be 4"
        )
    entities = manifest["entities"]
    if type(entities) is not dict or not entities:
        raise ValueError(
            "Checkpoint unified entity-split manifest entities must be a non-empty mapping"
        )
    if list(entities) != sorted(entities):
        raise ValueError(
            "Checkpoint unified entity-split manifest entity IDs must be sorted"
        )

    required_entity_fields = {
        "sources",
        "join_offset",
        "length",
        "train",
        "tail_blocks",
        "validation_block",
        "test_blocks",
        "validation_segment_id",
        "test_segment_ids",
    }
    for entity_id, entity in entities.items():
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(
                "Checkpoint unified entity-split manifest entity IDs must be strings"
            )
        if not isinstance(entity, Mapping):
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} must be a mapping"
            )
        _require_exact_keys(
            f"resolved_config.data_protocol.manifest.entities.{entity_id}",
            entity,
            required_entity_fields,
        )
        length = _positive_resolved_integer(
            f"resolved_config.data_protocol.manifest.entities.{entity_id}.length",
            entity["length"],
        )
        sources = entity["sources"]
        if type(sources) is not list or len(sources) not in {1, 2}:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} sources must contain one or two records"
            )
        source_lengths = []
        for index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"Checkpoint manifest entity {entity_id!r} source {index} must be a mapping"
                )
            _require_exact_keys(
                f"resolved_config.data_protocol.manifest.entities.{entity_id}.sources[{index}]",
                source,
                {"split", "segment_id", "length"},
            )
            if source["split"] not in {"train", "test"}:
                raise ValueError(
                    f"Checkpoint manifest entity {entity_id!r} source split is invalid"
                )
            if source["segment_id"] != entity_id:
                raise ValueError(
                    f"Checkpoint manifest entity {entity_id!r} source ID is invalid"
                )
            source_lengths.append(
                _positive_resolved_integer(
                    f"resolved_config.data_protocol.manifest.entities.{entity_id}.sources[{index}].length",
                    source["length"],
                )
            )
        if sum(source_lengths) != length:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} source lengths do not match entity length"
            )
        if len(sources) == 1:
            if entity["join_offset"] is not None:
                raise ValueError(
                    f"Checkpoint manifest entity {entity_id!r} standalone join_offset must be null"
                )
        elif (
            sources[0]["split"] != "train"
            or sources[1]["split"] != "test"
            or entity["join_offset"] != source_lengths[0]
        ):
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} paired source order or join_offset is invalid"
            )

        train_end = length * 3 // 5
        tail_length = length - train_end
        base, remainder = divmod(tail_length, 4)
        if train_end < window_size:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} training interval "
                f"is shorter than window_size={window_size}"
            )
        expected_blocks = []
        start = train_end
        for block_index in range(4):
            stop = start + base + int(block_index < remainder)
            expected_blocks.append([start, stop])
            start = stop
        if entity["train"] != [0, train_end]:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} train interval is invalid"
            )
        if entity["tail_blocks"] != expected_blocks:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} tail blocks are invalid"
            )
        validation_block = _require_integer(
            f"resolved_config.data_protocol.manifest.entities.{entity_id}.validation_block",
            entity["validation_block"],
        )
        if validation_block not in range(4):
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} validation block is invalid"
            )
        expected_test_blocks = [
            index for index in range(4) if index != validation_block
        ]
        if entity["test_blocks"] != expected_test_blocks:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} test blocks are invalid"
            )
        if entity["validation_segment_id"] != (
            f"{entity_id}:validation:{validation_block}"
        ):
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} validation segment ID is invalid"
            )
        expected_test_ids = [
            f"{entity_id}:test:{index}" for index in expected_test_blocks
        ]
        if entity["test_segment_ids"] != expected_test_ids:
            raise ValueError(
                f"Checkpoint manifest entity {entity_id!r} test segment IDs are invalid"
            )
    return len(entities)


def _validate_official_partition_manifest(
    manifest: Any,
    *,
    window_size: int,
) -> tuple[int, int]:
    """Validate the checkpoint-only official train/validation/test manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "Checkpoint official-partition manifest must be a mapping"
        )
    _require_exact_keys(
        "resolved_config.data_protocol.manifest",
        manifest,
        {
            "protocol_version",
            "validation_fraction",
            "training_labels_used",
            "train_segments",
            "test_segments",
        },
    )
    if manifest["protocol_version"] != OFFICIAL_PROTOCOL_VERSION:
        raise ValueError(
            "Checkpoint official-partition manifest has an unsupported version"
        )
    if manifest["validation_fraction"] != [1, 10]:
        raise ValueError(
            "Checkpoint official-partition validation_fraction must be [1, 10]"
        )
    if manifest["training_labels_used"] is not False:
        raise ValueError(
            "Checkpoint official-partition training_labels_used must be false"
        )
    train_segments = manifest["train_segments"]
    test_segments = manifest["test_segments"]
    if type(train_segments) is not dict or not train_segments:
        raise ValueError(
            "Checkpoint official-partition train_segments must be non-empty"
        )
    if type(test_segments) is not dict:
        raise ValueError(
            "Checkpoint official-partition test_segments must be a mapping"
        )
    if list(train_segments) != sorted(train_segments):
        raise ValueError(
            "Checkpoint official-partition train source IDs must be sorted"
        )
    if list(test_segments) != sorted(test_segments):
        raise ValueError(
            "Checkpoint official-partition test source IDs must be sorted"
        )

    validation_ids = {}
    for source_id, record in train_segments.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                "Checkpoint official-partition train source IDs must be strings"
            )
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Checkpoint official train source {source_id!r} must be a mapping"
            )
        _require_exact_keys(
            f"resolved_config.data_protocol.manifest.train_segments.{source_id}",
            record,
            {
                "source",
                "train_interval",
                "validation_interval",
                "train_segment_id",
                "validation_segment_id",
            },
        )
        source = record["source"]
        if not isinstance(source, Mapping):
            raise ValueError(
                f"Checkpoint official train source {source_id!r} identity must be a mapping"
            )
        _require_exact_keys(
            f"resolved_config.data_protocol.manifest.train_segments.{source_id}.source",
            source,
            {"split", "segment_id", "length"},
        )
        source_length = _positive_resolved_integer(
            f"resolved_config.data_protocol.manifest.train_segments.{source_id}.source.length",
            source["length"],
        )
        if source["split"] != "train" or source["segment_id"] != source_id:
            raise ValueError(
                f"Checkpoint official train source {source_id!r} identity is invalid"
            )
        validation_length = max(1, source_length // 10)
        train_stop = source_length - validation_length
        if train_stop < window_size:
            raise ValueError(
                f"Checkpoint official train source {source_id!r} interval is "
                f"shorter than window_size={window_size}"
            )
        if record["train_interval"] != [0, train_stop]:
            raise ValueError(
                f"Checkpoint official train source {source_id!r} train interval is invalid"
            )
        if record["validation_interval"] != [train_stop, source_length]:
            raise ValueError(
                f"Checkpoint official train source {source_id!r} validation interval is invalid"
            )
        expected_train_id = f"{source_id}:train"
        expected_validation_id = f"{source_id}:validation"
        if record["train_segment_id"] != expected_train_id:
            raise ValueError(
                f"Checkpoint official train source {source_id!r} train segment ID is invalid"
            )
        if record["validation_segment_id"] != expected_validation_id:
            raise ValueError(
                f"Checkpoint official train source {source_id!r} validation segment ID is invalid"
            )
        validation_ids[source_id] = expected_validation_id

    for source_id, record in test_segments.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                "Checkpoint official-partition test source IDs must be strings"
            )
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Checkpoint official test source {source_id!r} must be a mapping"
            )
        _require_exact_keys(
            f"resolved_config.data_protocol.manifest.test_segments.{source_id}",
            record,
            {
                "source",
                "interval",
                "test_segment_id",
                "validation_segment_id",
            },
        )
        source = record["source"]
        if not isinstance(source, Mapping):
            raise ValueError(
                f"Checkpoint official test source {source_id!r} identity must be a mapping"
            )
        _require_exact_keys(
            f"resolved_config.data_protocol.manifest.test_segments.{source_id}.source",
            source,
            {"split", "segment_id", "length"},
        )
        source_length = _positive_resolved_integer(
            f"resolved_config.data_protocol.manifest.test_segments.{source_id}.source.length",
            source["length"],
        )
        if source["split"] != "test" or source["segment_id"] != source_id:
            raise ValueError(
                f"Checkpoint official test source {source_id!r} identity is invalid"
            )
        if record["interval"] != [0, source_length]:
            raise ValueError(
                f"Checkpoint official test source {source_id!r} interval is invalid"
            )
        if record["test_segment_id"] != f"{source_id}:test":
            raise ValueError(
                f"Checkpoint official test source {source_id!r} segment ID is invalid"
            )
        if record["validation_segment_id"] != validation_ids.get(source_id):
            raise ValueError(
                f"Checkpoint official test source {source_id!r} validation reference is invalid"
            )
    return len(train_segments), len(test_segments)


def _validate_resolved_config(
    checkpoint: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    scaler: StandardScaler,
    seed: int,
    git_revision: str,
    source_fingerprint: str,
    calibrated_threshold: float | None,
    reproducibility: Mapping[str, Any],
    checkpoint_kind: str,
) -> None:
    resolved = _require_mapping(checkpoint, "resolved_config")
    _require_exact_keys(
        "resolved_config",
        resolved,
        REQUIRED_RESOLVED_CONFIG_KEYS,
    )
    if type(resolved["schema_version"]) is not int or resolved[
        "schema_version"
    ] != 1:
        raise ValueError(
            "Checkpoint field 'resolved_config.schema_version' must equal 1"
        )
    if resolved["seed"] != seed:
        raise ValueError(
            "Checkpoint field 'resolved_config.seed' must match seed"
        )
    if resolved["git_revision"] != git_revision:
        raise ValueError(
            "Checkpoint field 'resolved_config.git_revision' must match "
            "git_revision"
        )
    resolved_source_fingerprint = _validate_source_fingerprint(
        resolved["source_fingerprint"],
        field_name="resolved_config.source_fingerprint",
    )
    if resolved_source_fingerprint != source_fingerprint:
        raise ValueError(
            "Checkpoint field 'resolved_config.source_fingerprint' must "
            "match source_fingerprint"
        )

    device = _require_mapping(resolved, "device")
    _require_exact_keys(
        "resolved_config.device",
        device,
        {"requested", "resolved"},
    )
    for key in ("requested", "resolved"):
        if not isinstance(device[key], str) or not device[key].strip():
            raise ValueError(
                f"Checkpoint field 'resolved_config.device.{key}' "
                "must be a non-empty string"
            )

    dataset = _require_mapping(resolved, "dataset")
    _require_exact_keys(
        "resolved_config.dataset",
        dataset,
        {
            "name",
            "data_root",
            "n_features",
            "source_train_segment_count",
            "source_test_segment_count",
            "excluded_segments",
        },
    )
    if not isinstance(dataset["name"], str) or not dataset["name"]:
        raise ValueError(
            "Checkpoint field 'resolved_config.dataset.name' must be "
            "a non-empty string"
        )
    if not isinstance(dataset["data_root"], str) or not dataset["data_root"]:
        raise ValueError(
            "Checkpoint field 'resolved_config.dataset.data_root' must "
            "be a non-empty string"
        )
    if dataset["n_features"] != model_config["num_vars"]:
        raise ValueError(
            "Checkpoint resolved_config dataset n_features must match "
            "model_config.num_vars"
        )
    _positive_resolved_integer(
        "resolved_config.dataset.source_train_segment_count",
        dataset["source_train_segment_count"],
    )
    if (
        type(dataset["source_test_segment_count"]) is not int
        or dataset["source_test_segment_count"] < 0
    ):
        raise ValueError(
            "Checkpoint resolved_config dataset source_test_segment_count "
            "must be a non-negative integer"
        )
    if type(dataset["excluded_segments"]) is not list:
        raise ValueError(
            "Checkpoint field 'resolved_config.dataset.excluded_segments' "
            "must be a list"
        )

    data_protocol = _require_mapping(resolved, "data_protocol")
    _require_exact_keys(
        "resolved_config.data_protocol",
        data_protocol,
        {
            "protocol_version",
            "manifest",
            "normal_windows_only",
            "window_size",
            "stride",
            "short_segment_window_policy",
            "train_windows",
            "val_windows",
            "test_segment_count",
            "scaler",
        },
    )
    protocol_version = data_protocol["protocol_version"]
    if protocol_version not in {PROTOCOL_VERSION, OFFICIAL_PROTOCOL_VERSION, PAPER_PROTOCOL_VERSION}:
        raise ValueError(
            "Checkpoint resolved_config data_protocol version is unsupported"
        )
    if data_protocol["normal_windows_only"] is not True:
        raise ValueError(
            "Checkpoint resolved_config data_protocol normal_windows_only "
            "must be true"
        )
    if data_protocol["window_size"] != model_config["time_len"]:
        raise ValueError(
            "Checkpoint resolved_config window_size must match "
            "model_config.time_len"
        )
    _positive_resolved_integer(
        "resolved_config.data_protocol.stride",
        data_protocol["stride"],
    )
    if (
        data_protocol["short_segment_window_policy"]
        != "full_segment_when_shorter"
    ):
        raise ValueError(
            "Checkpoint resolved_config short_segment_window_policy is invalid"
        )
    if protocol_version == PAPER_PROTOCOL_VERSION:
        paper_manifest = validate_paper_manifest(
            data_protocol["manifest"], window_size=data_protocol["window_size"]
        )
        records = paper_manifest["sources"]
        for source_split in ("train", "test"):
            count = sum(record["source"]["split"] == source_split for record in records.values())
            if count != dataset[f"source_{source_split}_segment_count"]:
                raise ValueError("Paper protocol source count does not match dataset metadata")
        expected_test_segment_count = len(records)
    elif protocol_version == OFFICIAL_PROTOCOL_VERSION:
        train_source_count, expected_test_segment_count = (
            _validate_official_partition_manifest(
                data_protocol["manifest"],
                window_size=data_protocol["window_size"],
            )
        )
        if train_source_count != dataset["source_train_segment_count"]:
            raise ValueError(
                "Checkpoint official-partition train source count must match "
                "resolved_config.dataset"
            )
        if expected_test_segment_count != dataset["source_test_segment_count"]:
            raise ValueError(
                "Checkpoint official-partition test source count must match "
                "resolved_config.dataset"
            )
    else:
        entity_count = _validate_entity_split_manifest(
            data_protocol["manifest"],
            seed=seed,
            window_size=data_protocol["window_size"],
        )
        expected_test_segment_count = 3 * entity_count
    _positive_resolved_integer(
        "resolved_config.data_protocol.train_windows",
        data_protocol["train_windows"],
    )
    if (
        type(data_protocol["val_windows"]) is not int
        or data_protocol["val_windows"] < 1
    ):
        raise ValueError(
            "Checkpoint resolved_config val_windows must be a "
            "positive integer"
        )
    if (
        type(data_protocol["test_segment_count"]) is not int
        or data_protocol["test_segment_count"] != expected_test_segment_count
    ):
        raise ValueError(
            "Checkpoint resolved_config test_segment_count does not match "
            "the protocol manifest"
        )
    scaler_config = _require_mapping(data_protocol, "scaler")
    _require_exact_keys(
        "resolved_config.data_protocol.scaler",
        scaler_config,
        {"type", "fit_scope", "eps", "n_features"},
    )
    if scaler_config["type"] != "StandardScaler":
        raise ValueError(
            "Checkpoint resolved_config scaler type must be StandardScaler"
        )
    expected_scaler_scope = (
        "official_training_window_covered_points_only"
        if protocol_version == OFFICIAL_PROTOCOL_VERSION
        else "normal_training_window_covered_points_only"
    )
    if scaler_config["fit_scope"] != expected_scaler_scope:
        raise ValueError(
            "Checkpoint resolved_config scaler fit_scope is invalid"
        )
    if scaler_config["n_features"] != model_config["num_vars"]:
        raise ValueError(
            "Checkpoint resolved_config scaler n_features must match "
            "model_config.num_vars"
        )
    resolved_eps = _finite_number(
        "resolved_config.data_protocol.scaler.eps",
        scaler_config["eps"],
        minimum=0.0,
    )
    if resolved_eps <= 0.0 or not math.isclose(
        resolved_eps,
        scaler.eps,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Checkpoint resolved_config scaler eps must match scaler_state"
        )

    if not _typed_equal(dict(resolved["model"]), dict(model_config)):
        raise ValueError(
            "Checkpoint field 'resolved_config.model' must match model_config"
        )

    optimizer_config = _require_mapping(resolved, "optimizer")
    _require_exact_keys(
        "resolved_config.optimizer",
        optimizer_config,
        {"type", "lr", "weight_decay", "betas", "eps", "amsgrad"},
    )
    if optimizer_config["type"] != "AdamW":
        raise ValueError(
            "Checkpoint resolved_config optimizer type must be AdamW"
        )
    optimizer_state = _require_mapping(
        checkpoint, "optimizer_state_dict"
    )
    param_groups = optimizer_state.get("param_groups")
    if type(param_groups) is not list or len(param_groups) != 1:
        raise ValueError(
            "Checkpoint optimizer_state_dict must have one param group"
        )
    optimizer_group = param_groups[0]
    initial_lr = _finite_number(
        "resolved_config.optimizer.lr",
        optimizer_config["lr"],
    )
    current_lr = _finite_number(
        "optimizer_state_dict.param_groups.lr",
        optimizer_group["lr"],
    )
    if initial_lr <= 0.0 or current_lr <= 0.0:
        raise ValueError(
            "Checkpoint initial and current optimizer lr must be positive"
        )
    if current_lr > initial_lr and not math.isclose(
        current_lr,
        initial_lr,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Checkpoint current optimizer lr cannot exceed the "
            "resolved_config initial lr"
        )
    for key in ("weight_decay", "eps"):
        resolved_value = _finite_number(
            f"resolved_config.optimizer.{key}",
            optimizer_config[key],
        )
        if not math.isclose(
            resolved_value,
            float(optimizer_group[key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Checkpoint resolved_config optimizer {key} does not "
                "match optimizer_state_dict"
            )
    if (
        type(optimizer_config["betas"]) is not list
        or optimizer_config["betas"] != list(optimizer_group["betas"])
    ):
        raise ValueError(
            "Checkpoint resolved_config optimizer betas do not match "
            "optimizer_state_dict"
        )
    if optimizer_config["amsgrad"] is not optimizer_group["amsgrad"]:
        raise ValueError(
            "Checkpoint resolved_config optimizer amsgrad does not match "
            "optimizer_state_dict"
        )

    scheduler_config = _require_mapping(resolved, "scheduler")
    _require_exact_keys(
        "resolved_config.scheduler",
        scheduler_config,
        {"type", "mode", "factor", "patience", "min_lr"},
    )
    if scheduler_config["type"] != "ReduceLROnPlateau":
        raise ValueError(
            "Checkpoint resolved_config scheduler type must be "
            "ReduceLROnPlateau"
        )
    scheduler_state = _require_mapping(
        checkpoint, "scheduler_state_dict"
    )
    missing_scheduler_state = sorted(
        {"mode", "factor", "patience", "min_lrs"} - set(scheduler_state)
    )
    if missing_scheduler_state:
        raise ValueError(
            "Checkpoint scheduler_state_dict is missing resolved-config "
            "comparison fields: " + ", ".join(missing_scheduler_state)
        )
    if scheduler_config["mode"] != scheduler_state.get("mode"):
        raise ValueError(
            "Checkpoint resolved_config scheduler mode does not match "
            "scheduler_state_dict"
        )
    if scheduler_config["patience"] != scheduler_state.get("patience"):
        raise ValueError(
            "Checkpoint resolved_config scheduler patience does not match "
            "scheduler_state_dict"
        )
    for resolved_key, state_key in (
        ("factor", "factor"),
        ("min_lr", "min_lrs"),
    ):
        resolved_value = _finite_number(
            f"resolved_config.scheduler.{resolved_key}",
            scheduler_config[resolved_key],
        )
        state_value = scheduler_state[state_key]
        if state_key == "min_lrs":
            if type(state_value) is not list or len(state_value) != 1:
                raise ValueError(
                    "Checkpoint scheduler_state_dict min_lrs is invalid"
                )
            state_value = state_value[0]
        if not math.isclose(
            resolved_value,
            float(state_value),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Checkpoint resolved_config scheduler {resolved_key} "
                "does not match scheduler_state_dict"
            )

    training = _require_mapping(resolved, "training")
    _require_exact_keys(
        "resolved_config.training",
        training,
        {
            "batch_size",
            "num_epochs",
            "early_stopping_patience",
            "scheduler_patience",
            "grad_clip",
            "warmup_epochs",
            "warmup_start_factor",
            "gradient_accumulation_steps",
            "loss_warmup_epochs",
            "train_compute_scores",
        },
    )
    _positive_resolved_integer(
        "resolved_config.training.batch_size",
        training["batch_size"],
    )
    num_epochs = _positive_resolved_integer(
        "resolved_config.training.num_epochs",
        training["num_epochs"],
    )
    if num_epochs < checkpoint["epoch"]:
        raise ValueError(
            "Checkpoint resolved_config num_epochs cannot be less than epoch"
        )
    early_patience = _positive_resolved_integer(
        "resolved_config.training.early_stopping_patience",
        training["early_stopping_patience"],
    )
    scheduler_patience = _positive_resolved_integer(
        "resolved_config.training.scheduler_patience",
        training["scheduler_patience"],
    )
    if (
        scheduler_patience >= early_patience
        or scheduler_patience != scheduler_config["patience"]
    ):
        raise ValueError(
            "Checkpoint resolved_config scheduler_patience must match the "
            "scheduler and be smaller than early-stopping patience"
        )
    _finite_number(
        "resolved_config.training.grad_clip",
        training["grad_clip"],
    )
    warmup_epochs = _require_integer(
        "resolved_config.training.warmup_epochs",
        training["warmup_epochs"],
    )
    if warmup_epochs > num_epochs:
        raise ValueError(
            "Checkpoint resolved_config warmup_epochs cannot exceed "
            "num_epochs"
        )
    warmup_start_factor = _finite_number(
        "resolved_config.training.warmup_start_factor",
        training["warmup_start_factor"],
    )
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError(
            "Checkpoint resolved_config warmup_start_factor must be "
            "in (0, 1]"
        )
    _positive_resolved_integer(
        "resolved_config.training.gradient_accumulation_steps",
        training["gradient_accumulation_steps"],
    )
    loss_warmup_epochs = _require_integer(
        "resolved_config.training.loss_warmup_epochs",
        training["loss_warmup_epochs"],
    )
    if loss_warmup_epochs > num_epochs:
        raise ValueError(
            "Checkpoint resolved_config loss_warmup_epochs cannot exceed "
            "num_epochs"
        )
    if warmup_epochs > 0 and checkpoint["epoch"] <= warmup_epochs:
        progress = float(checkpoint["epoch"]) / float(warmup_epochs)
        expected_lr = initial_lr * (
            warmup_start_factor
            + (1.0 - warmup_start_factor) * progress
        )
        if not math.isclose(
            current_lr,
            expected_lr,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Checkpoint optimizer lr does not match the resolved "
                "warmup trajectory"
            )
        if (
            scheduler_state.get("last_epoch") != 0
            or scheduler_state.get("num_bad_epochs") != 0
            or float(scheduler_state.get("best")) != float("inf")
        ):
            raise ValueError(
                "Checkpoint plateau scheduler advanced during LR warmup"
            )
    if training["train_compute_scores"] is not False:
        raise ValueError(
            "Checkpoint resolved_config train_compute_scores must be false"
        )

    validation = _require_mapping(resolved, "validation")
    _require_exact_keys(
        "resolved_config.validation",
        validation,
        {
            "seed",
            "rounds",
            "masked",
            "source",
            "objective_revision",
            "component_weights",
        },
    )
    if validation["seed"] != reproducibility["validation_seed"]:
        raise ValueError(
            "Checkpoint resolved_config validation seed must match "
            "reproducibility"
        )
    if validation["rounds"] != reproducibility["validation_rounds"]:
        raise ValueError(
            "Checkpoint resolved_config validation rounds must match "
            "reproducibility"
        )
    if validation["masked"] is not True:
        raise ValueError(
            "Checkpoint resolved_config validation masked must be true"
        )
    if (
        validation["objective_revision"]
        != FIXED_VALIDATION_OBJECTIVE_REVISION
    ):
        raise ValueError(
            "Checkpoint resolved_config validation objective revision "
            "is invalid"
        )
    component_weights = validation["component_weights"]
    if (
        type(component_weights) is not dict
        or component_weights != FIXED_VALIDATION_COMPONENT_WEIGHTS
    ):
        raise ValueError(
            "Checkpoint resolved_config validation component weights "
            "are invalid"
        )
    expected_validation_source = (
        "official_train_tail"
        if protocol_version == OFFICIAL_PROTOCOL_VERSION
        else "source_chronological_validation_10_percent"
        if protocol_version == PAPER_PROTOCOL_VERSION
        else "protocol_random_tail_block"
    )
    if validation["source"] != expected_validation_source:
        raise ValueError(
            "Checkpoint resolved_config validation source is invalid"
        )

    calibration = _require_mapping(resolved, "calibration")
    _require_exact_keys(
        "resolved_config.calibration",
        calibration,
        {"source", "percentile", "threshold"},
    )
    if calibration["source"] != "normal_training_candidate_windows_only":
        raise ValueError(
            "Checkpoint resolved_config calibration source is invalid"
        )
    percentile = _finite_number(
        "resolved_config.calibration.percentile",
        calibration["percentile"],
    )
    if not 0.0 < percentile < 100.0:
        raise ValueError(
            "Checkpoint resolved_config calibration percentile must be "
            "in (0, 100)"
        )
    if checkpoint_kind == "formal":
        if (
            type(calibration["threshold"]) is not float
            or calibrated_threshold is None
            or not math.isclose(
                calibration["threshold"],
                calibrated_threshold,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(
                "Formal checkpoint resolved_config calibration threshold "
                "must match calibrated_threshold"
            )
    elif calibration["threshold"] is not None or calibrated_threshold is not None:
        raise ValueError(
            "Training recovery calibration threshold must be null"
        )


def _validate_trainer_state(checkpoint: Mapping[str, Any]) -> None:
    trainer_state = _require_mapping(checkpoint, "trainer_state")
    _require_exact_keys(
        "trainer_state",
        trainer_state,
        {
            "early_stopping",
            "best_epoch",
            "train_losses",
            "val_losses",
            "last_epoch",
            "dataloader_generator_state",
        },
    )
    early_stopping = _require_mapping(
        trainer_state, "early_stopping"
    )
    _require_exact_keys(
        "trainer_state.early_stopping",
        early_stopping,
        {"best_score", "counter"},
    )
    best_score = early_stopping["best_score"]
    if (
        best_score is not None
        and (
            isinstance(best_score, bool)
            or not isinstance(best_score, (int, float))
            or not math.isfinite(float(best_score))
        )
    ):
        raise ValueError(
            "Checkpoint trainer_state early-stopping best_score must "
            "be finite or null"
        )
    _require_integer(
        "trainer_state.early_stopping.counter",
        early_stopping["counter"],
    )
    best_epoch = _require_integer(
        "trainer_state.best_epoch",
        trainer_state["best_epoch"],
        minimum=1,
    )
    last_epoch = _require_integer(
        "trainer_state.last_epoch",
        trainer_state["last_epoch"],
        minimum=1,
    )
    if best_epoch > last_epoch:
        raise ValueError(
            "Checkpoint trainer_state best_epoch cannot exceed last_epoch"
        )
    if checkpoint["epoch"] != last_epoch:
        raise ValueError(
            "Checkpoint epoch must match trainer_state.last_epoch"
        )
    for key in ("train_losses", "val_losses"):
        history = trainer_state[key]
        if type(history) is not list:
            raise ValueError(
                f"Checkpoint trainer_state {key} must be a list"
            )
        for entry in history:
            if type(entry) is not dict or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in entry.values()
            ):
                raise ValueError(
                    f"Checkpoint trainer_state {key} contains an invalid "
                    "history entry"
                )
    if len(trainer_state["train_losses"]) != last_epoch:
        raise ValueError(
            "Checkpoint trainer_state train history length must equal "
            "last_epoch"
        )
    if len(trainer_state["val_losses"]) not in {
        0,
        last_epoch,
    }:
        raise ValueError(
            "Checkpoint trainer_state validation history length must be "
            "zero or last_epoch"
        )
    trainer_generator_state = trainer_state[
        "dataloader_generator_state"
    ]
    _validate_dataloader_generator_state(trainer_generator_state)
    if not torch.equal(
        trainer_generator_state.detach().cpu(),
        checkpoint["dataloader_generator_state"].detach().cpu(),
    ):
        raise ValueError(
            "Checkpoint trainer_state DataLoader generator state must "
            "match the root state"
        )


def _validate_checkpoint_protocol(
    checkpoint: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    repository_root: str | Path | None = None,
) -> None:
    _validate_weights_only_tree(checkpoint)
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise ValueError(
            "Checkpoint is incomplete; missing required fields: "
            + ", ".join(missing)
        )
    checkpoint_kind = checkpoint["checkpoint_kind"]
    if checkpoint_kind not in {"formal", "training_recovery"}:
        raise ValueError(
            "Checkpoint field 'checkpoint_kind' must be 'formal' or "
            "'training_recovery'"
        )
    if expected_kind is not None and checkpoint_kind != expected_kind:
        raise ValueError(
            f"Expected checkpoint_kind={expected_kind!r}, got "
            f"{checkpoint_kind!r}"
        )

    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError(
            "Checkpoint field 'model_state_dict' must be a mapping"
        )
    _require_mapping(checkpoint, "optimizer_state_dict")
    _require_mapping(checkpoint, "scheduler_state_dict")
    _require_integer("epoch", checkpoint["epoch"], minimum=1)
    rng_state = _require_mapping(checkpoint, "rng_state")
    _validate_rng_state(rng_state)
    _validate_dataloader_generator_state(
        checkpoint["dataloader_generator_state"]
    )
    _validate_trainer_state(checkpoint)
    config = _require_mapping(checkpoint, "model_config")
    _validate_model_config(config)

    scaler_state = checkpoint["scaler_state"]
    if not isinstance(scaler_state, dict):
        raise ValueError(
            "Checkpoint field 'scaler_state' must be a valid scaler mapping"
        )
    if type(scaler_state.get("_fitted")) is not bool:
        raise ValueError(
            "Checkpoint field 'scaler_state._fitted' must be a boolean"
        )
    try:
        scaler = StandardScaler()
        scaler.load_state_dict(scaler_state)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint field 'scaler_state' is invalid"
        ) from exc
    if not scaler._fitted:
        raise ValueError(
            "Checkpoint field 'scaler_state' must be fitted"
        )
    expected_scaler_shape = (1, 1, config["num_vars"])
    if (
        scaler.mean is None
        or scaler.std is None
        or scaler.mean.shape != expected_scaler_shape
        or scaler.std.shape != expected_scaler_shape
    ):
        raise ValueError(
            "Checkpoint field 'scaler_state' width must match "
            f"model_config.num_vars={config['num_vars']}"
        )

    threshold = checkpoint["calibrated_threshold"]
    if checkpoint_kind == "formal":
        if (
            type(threshold) is not float
            or not torch.isfinite(torch.tensor(float(threshold)))
        ):
            raise ValueError(
                "Formal checkpoint calibrated_threshold must be finite"
            )
    elif threshold is not None:
        raise ValueError(
            "Training recovery calibrated_threshold must be null"
        )

    seed = _require_integer("seed", checkpoint["seed"])
    git_revision = checkpoint["git_revision"]
    if not isinstance(git_revision, str) or not git_revision.strip():
        raise ValueError(
            "Checkpoint field 'git_revision' must be a non-empty string"
        )
    source_fingerprint = _validate_source_fingerprint(
        checkpoint["source_fingerprint"],
        field_name="source_fingerprint",
    )
    current_source_fingerprint = compute_source_fingerprint(repository_root)
    if source_fingerprint != current_source_fingerprint:
        raise ValueError(
            "Checkpoint source_fingerprint does not match the current "
            "runtime source fingerprint"
        )

    reproducibility = _require_mapping(checkpoint, "reproducibility")
    reproducibility_missing = sorted(
        REQUIRED_REPRODUCIBILITY_KEYS - set(reproducibility)
    )
    if reproducibility_missing:
        raise ValueError(
            "Checkpoint field 'reproducibility' is incomplete; missing: "
            + ", ".join(reproducibility_missing)
        )
    if reproducibility.get("seed") != seed:
        raise ValueError(
            "Checkpoint field 'reproducibility.seed' must match seed"
        )
    _require_integer(
        "reproducibility.validation_seed",
        reproducibility["validation_seed"],
    )
    _require_integer(
        "reproducibility.validation_rounds",
        reproducibility["validation_rounds"],
        minimum=1,
    )
    if type(reproducibility["deterministic_algorithms"]) is not bool:
        raise ValueError(
            "Checkpoint field 'reproducibility.deterministic_algorithms' "
            "must be boolean"
        )
    for key in ("cudnn_deterministic", "cudnn_benchmark"):
        if type(reproducibility[key]) is not bool:
            raise ValueError(
                f"Checkpoint field 'reproducibility.{key}' must be boolean"
            )
    _validate_resolved_config(
        checkpoint,
        model_config=config,
        scaler=scaler,
        seed=seed,
        git_revision=git_revision,
        source_fingerprint=source_fingerprint,
        calibrated_threshold=threshold,
        reproducibility=reproducibility,
        checkpoint_kind=checkpoint_kind,
    )


def _model_kwargs(config: Mapping[str, Any]) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key != "architecture"
    }


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: str | torch.device = "cpu",
    *,
    repository_root: str | Path | None = None,
) -> M2BMoEModel:
    """Restore a complete v3 checkpoint using strict parameter matching."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    _validate_checkpoint_protocol(
        checkpoint,
        expected_kind="formal",
        repository_root=repository_root,
    )
    config = checkpoint["model_config"]
    model = M2BMoEModel(**_model_kwargs(config))
    if not _typed_equal(model.get_config(), dict(config)):
        raise ValueError(
            "Checkpoint model_config is not canonical: normalized "
            "get_config() does not match the checkpoint"
        )
    _validate_model_state_dict(model, checkpoint["model_state_dict"])
    try:
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint parameters do not match its M2B-MoE v3 configuration"
        ) from exc
    _restore_optimizer_scheduler(model, checkpoint)
    model.to(device)
    model.eval()
    return model


def validate_checkpoint_for_model(
    checkpoint: Mapping[str, Any],
    model: M2BMoEModel,
    *,
    expected_kind: str,
    repository_root: str | Path | None = None,
) -> None:
    """Strictly validate checkpoint schema and active-model compatibility."""
    if not isinstance(model, M2BMoEModel):
        raise TypeError(
            "Strict checkpoint restore requires an active M2BMoEModel"
        )
    _validate_checkpoint_protocol(
        checkpoint,
        expected_kind=expected_kind,
        repository_root=repository_root,
    )
    if not _typed_equal(
        model.get_config(),
        dict(checkpoint["model_config"]),
    ):
        raise ValueError(
            "Active model configuration does not exactly match checkpoint "
            "model_config"
        )
    _validate_model_state_dict(model, checkpoint["model_state_dict"])
    _restore_optimizer_scheduler(model, checkpoint)


def restore_training_state_from_checkpoint(
    model: M2BMoEModel,
    checkpoint: Mapping[str, Any],
    *,
    restore_rng: bool = True,
    train_generator: torch.Generator | None = None,
    repository_root: str | Path | None = None,
) -> Tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    int,
]:
    """Restore AdamW, ReduceLROnPlateau, epoch, and optionally RNG state."""
    if type(restore_rng) is not bool:
        raise ValueError("restore_rng must be a boolean")
    if train_generator is not None and not isinstance(
        train_generator, torch.Generator
    ):
        raise TypeError("train_generator must be a torch.Generator")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    _validate_checkpoint_protocol(
        checkpoint,
        repository_root=repository_root,
    )
    if not _typed_equal(model.get_config(), dict(checkpoint["model_config"])):
        raise ValueError(
            "The supplied model configuration does not match the checkpoint"
        )
    optimizer, scheduler = _restore_optimizer_scheduler(model, checkpoint)
    if restore_rng:
        rng_state = checkpoint["rng_state"]
        random.setstate(_list_tree_to_tuple(rng_state["python"]))
        numpy_state = rng_state["numpy"]
        np.random.set_state(
            (
                numpy_state[0],
                numpy_state[1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint32, copy=False),
                numpy_state[2],
                numpy_state[3],
                float(numpy_state[4]),
            )
        )
        torch.set_rng_state(rng_state["torch"].cpu())
        cuda_states = rng_state["cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA RNG state cannot be restored because CUDA "
                    "is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(
                    "CUDA RNG state count does not match available devices"
                )
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in cuda_states]
            )
        reproducibility = checkpoint["reproducibility"]
        torch.use_deterministic_algorithms(
            reproducibility["deterministic_algorithms"]
        )
        torch.backends.cudnn.deterministic = reproducibility[
            "cudnn_deterministic"
        ]
        torch.backends.cudnn.benchmark = reproducibility["cudnn_benchmark"]
    if train_generator is not None:
        train_generator.set_state(
            checkpoint["dataloader_generator_state"].detach().cpu()
        )
    return optimizer, scheduler, checkpoint["epoch"]


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    repository_root: str | Path | None = None,
) -> Tuple[M2BMoEModel, Mapping[str, Any]]:
    """Load and validate a v3 checkpoint from disk."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=True
        )
    except (pickle.UnpicklingError, RuntimeError, TypeError) as exc:
        raise ValueError(
            "Checkpoint was rejected by torch.load(weights_only=True); "
            "unsafe or unsupported serialized objects are not accepted"
        ) from exc
    if type(checkpoint) is not dict:
        raise ValueError("Checkpoint root must be a dictionary")
    model = build_model_from_checkpoint(
        checkpoint,
        device=device,
        repository_root=repository_root,
    )
    return model, checkpoint
