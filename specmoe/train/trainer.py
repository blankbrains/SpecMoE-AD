"""Deterministic training, validation, and checkpointing for M2B-MoE."""

from __future__ import annotations

import copy
import os
import random
import tempfile
import time
from contextlib import contextmanager
from numbers import Integral, Real
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


FIXED_VALIDATION_OBJECTIVE_REVISION = (
    "fixed_recon1_freq1_diversity0.3_load_balance0.01_v1"
)
FIXED_VALIDATION_COMPONENT_WEIGHTS = {
    "recon": 1.0,
    "freq": 1.0,
    "diversity": 0.3,
    "load_balance": 0.01,
}

_CHECKPOINT_STATE_KEYS = {
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state",
    "score",
    "model_config",
    "reproducibility",
    "dataloader_generator_state",
    "trainer_state",
    "checkpoint_kind",
}
_REQUIRED_CHECKPOINT_METADATA = {
    "scaler_state",
    "calibrated_threshold",
    "seed",
    "resolved_config",
    "git_revision",
}


def _validated_seed(seed: int, *, name: str = "seed") -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    resolved = int(seed)
    if resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return resolved


def set_deterministic_seed(seed: int) -> None:
    """Seed every supported RNG and enable deterministic torch execution."""
    resolved = _validated_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return an explicit CPU generator for deterministic DataLoader order."""
    return torch.Generator(device="cpu").manual_seed(_validated_seed(seed))


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a DataLoader worker process."""
    if isinstance(worker_id, bool) or not isinstance(worker_id, Integral):
        raise ValueError("worker_id must be a non-negative integer")
    if int(worker_id) < 0:
        raise ValueError("worker_id must be a non-negative integer")
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _list_tree(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_list_tree(child) for child in value]
    if isinstance(value, list):
        return [_list_tree(child) for child in value]
    return value


def _tuple_tree(value: Any) -> Any:
    if type(value) is list:
        return tuple(_tuple_tree(child) for child in value)
    return value


def _weights_only_safe(value: Any) -> Any:
    """Convert nested runtime state to torch weights-only-safe values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        array = value
        if array.dtype == np.uint32:
            array = array.astype(np.int64, copy=True)
        return torch.from_numpy(np.array(array, copy=True))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            _weights_only_safe(key): _weights_only_safe(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_weights_only_safe(child) for child in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(
        f"Checkpoint value of type {type(value).__name__} is not "
        "compatible with torch.load(weights_only=True)"
    )


def capture_rng_state() -> Dict[str, Any]:
    """Capture all process RNG streams using weights-only-safe values."""
    numpy_state = np.random.get_state()
    return {
        "python": _weights_only_safe(random.getstate()),
        "numpy": [
            str(numpy_state[0]),
            torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch": torch.get_rng_state().detach().cpu().clone(),
        "cuda": (
            [
                state.detach().cpu().clone()
                for state in torch.cuda.get_rng_state_all()
            ]
            if torch.cuda.is_available()
            else []
        ),
    }


def atomic_torch_save(value: Mapping[str, Any], path: str) -> None:
    """Atomically replace a checkpoint after flushing it to stable storage."""
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=directory,
            prefix=".checkpoint-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = file.name
            torch.save(dict(value), file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def restore_rng_state(
    rng_state: Mapping[str, Any],
    reproducibility: Optional[Mapping[str, Any]] = None,
) -> None:
    """Restore RNG streams and deterministic backend flags."""
    if not isinstance(rng_state, Mapping):
        raise ValueError("rng_state must be a mapping")
    required = {"python", "numpy", "torch", "cuda"}
    if set(rng_state) != required:
        raise ValueError(
            "rng_state must contain exactly python, numpy, torch, and cuda"
        )
    random.setstate(_tuple_tree(rng_state["python"]))
    numpy_state = rng_state["numpy"]
    if type(numpy_state) is not list or len(numpy_state) != 5:
        raise ValueError("rng_state.numpy must be a five-item list")
    numpy_keys = numpy_state[1]
    if not isinstance(numpy_keys, torch.Tensor):
        raise ValueError("rng_state.numpy keys must be a tensor")
    np.random.set_state(
        (
            str(numpy_state[0]),
            numpy_keys.detach()
            .cpu()
            .numpy()
            .astype(np.uint32, copy=False),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    torch_state = rng_state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError("rng_state.torch must be a tensor")
    torch.set_rng_state(torch_state.detach().cpu())
    cuda_states = rng_state["cuda"]
    if type(cuda_states) is not list:
        raise ValueError("rng_state.cuda must be a list")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cannot restore CUDA RNG state because CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG state count does not match available devices"
            )
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() for state in cuda_states]
        )
    if reproducibility is not None:
        torch.use_deterministic_algorithms(
            bool(reproducibility["deterministic_algorithms"])
        )
        torch.backends.cudnn.deterministic = bool(
            reproducibility["cudnn_deterministic"]
        )
        torch.backends.cudnn.benchmark = bool(
            reproducibility["cudnn_benchmark"]
        )


@contextmanager
def _preserved_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


class EarlyStopping:
    """Track finite validation loss improvements."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        if isinstance(patience, bool) or not isinstance(patience, Integral):
            raise ValueError("patience must be a positive integer")
        if int(patience) < 1:
            raise ValueError("patience must be a positive integer")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, Real)
            or not np.isfinite(float(min_delta))
            or float(min_delta) < 0.0
        ):
            raise ValueError("min_delta must be finite and non-negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)
        self.counter = 0
        self.best_score: Optional[float] = None
        self.improved = False

    def __call__(self, score: float, _model: Optional[nn.Module] = None) -> bool:
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not np.isfinite(float(score))
        ):
            raise FloatingPointError(
                "Early stopping received a non-finite score"
            )
        resolved_score = float(score)
        self.improved = (
            self.best_score is None
            or resolved_score < self.best_score - self.min_delta
        )
        if self.improved:
            self.best_score = resolved_score
            self.counter = 0
            if self.verbose:
                print(
                    f"  [EarlyStopping] Improved to {resolved_score:.6f}"
                )
            return False
        self.counter += 1
        if self.verbose:
            print(
                "  [EarlyStopping] No improvement "
                f"({self.counter}/{self.patience})"
            )
        return self.counter >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_score": self.best_score,
            "counter": self.counter,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("early_stopping state must be a mapping")
        if set(state) != {"best_score", "counter"}:
            raise ValueError("early_stopping state schema is invalid")
        best_score = state["best_score"]
        if best_score is not None and (
            isinstance(best_score, bool)
            or not isinstance(best_score, Real)
            or not np.isfinite(float(best_score))
        ):
            raise ValueError("early_stopping best_score must be finite")
        counter = state["counter"]
        if (
            isinstance(counter, bool)
            or not isinstance(counter, Integral)
            or int(counter) < 0
        ):
            raise ValueError(
                "early_stopping counter must be a non-negative integer"
            )
        self.best_score = (
            None if best_score is None else float(best_score)
        )
        self.counter = int(counter)
        self.improved = False


class Trainer:
    """Production training loop for the maintained M2B-MoE model."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        patience: int = 10,
        device: Optional[str] = None,
        log_interval: int = 10,
        checkpoint_metadata: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        validation_seed: Optional[int] = None,
        validation_rounds: Optional[int] = None,
        scheduler_patience: Optional[int] = None,
        train_generator: Optional[torch.Generator] = None,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        gradient_accumulation_steps: int = 1,
        loss_warmup_epochs: int = 0,
        validation_component_weights: Optional[
            Mapping[str, Real]
        ] = None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, Real)
            or not np.isfinite(float(lr))
            or float(lr) <= 0.0
        ):
            raise ValueError("lr must be finite and positive")
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, Real)
            or not np.isfinite(float(weight_decay))
            or float(weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            isinstance(grad_clip, bool)
            or not isinstance(grad_clip, Real)
            or not np.isfinite(float(grad_clip))
            or float(grad_clip) < 0.0
        ):
            raise ValueError("grad_clip must be finite and non-negative")
        if isinstance(log_interval, bool) or not isinstance(
            log_interval, Integral
        ) or int(log_interval) < 1:
            raise ValueError("log_interval must be a positive integer")
        if (
            isinstance(warmup_epochs, bool)
            or not isinstance(warmup_epochs, Integral)
            or int(warmup_epochs) < 0
        ):
            raise ValueError("warmup_epochs must be a non-negative integer")
        if (
            isinstance(warmup_start_factor, bool)
            or not isinstance(warmup_start_factor, Real)
            or not np.isfinite(float(warmup_start_factor))
            or not 0.0 < float(warmup_start_factor) <= 1.0
        ):
            raise ValueError(
                "warmup_start_factor must be finite and in (0, 1]"
            )
        if (
            isinstance(gradient_accumulation_steps, bool)
            or not isinstance(gradient_accumulation_steps, Integral)
            or int(gradient_accumulation_steps) < 1
        ):
            raise ValueError(
                "gradient_accumulation_steps must be a positive integer"
            )
        if (
            isinstance(loss_warmup_epochs, bool)
            or not isinstance(loss_warmup_epochs, Integral)
            or int(loss_warmup_epochs) < 0
        ):
            raise ValueError(
                "loss_warmup_epochs must be a non-negative integer"
            )
        resolved_validation_weights = None
        if validation_component_weights is not None:
            if not isinstance(validation_component_weights, Mapping):
                raise TypeError(
                    "validation_component_weights must be a mapping"
                )
            expected_components = set(
                FIXED_VALIDATION_COMPONENT_WEIGHTS
            )
            if set(validation_component_weights) != expected_components:
                raise ValueError(
                    "validation_component_weights must contain exactly "
                    + ", ".join(sorted(expected_components))
                )
            resolved_validation_weights = {}
            for name in FIXED_VALIDATION_COMPONENT_WEIGHTS:
                value = validation_component_weights[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not np.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        "validation_component_weights values must be "
                        "finite and non-negative"
                    )
                resolved_validation_weights[name] = float(value)
            if not any(
                value > 0.0
                for value in resolved_validation_weights.values()
            ):
                raise ValueError(
                    "validation_component_weights must include a "
                    "positive weight"
                )

        metadata = dict(checkpoint_metadata or {})
        reproducibility = metadata.get("reproducibility")
        reproducibility = (
            dict(reproducibility)
            if isinstance(reproducibility, Mapping)
            else {}
        )
        resolved_seed = (
            metadata.get("seed", 0) if seed is None else seed
        )
        resolved_validation_seed = (
            reproducibility.get(
                "validation_seed",
                _validated_seed(resolved_seed) + 1_000_003,
            )
            if validation_seed is None
            else validation_seed
        )
        resolved_validation_rounds = (
            reproducibility.get("validation_rounds", 1)
            if validation_rounds is None
            else validation_rounds
        )
        self.seed = _validated_seed(resolved_seed)
        self.validation_seed = _validated_seed(
            resolved_validation_seed,
            name="validation_seed",
        )
        if (
            isinstance(resolved_validation_rounds, bool)
            or not isinstance(resolved_validation_rounds, Integral)
            or int(resolved_validation_rounds) < 1
        ):
            raise ValueError(
                "validation_rounds must be a positive integer"
            )
        self.validation_rounds = int(resolved_validation_rounds)
        self.early_stopping = EarlyStopping(patience=patience)
        resolved_scheduler_patience = (
            max(1, self.early_stopping.patience // 2)
            if scheduler_patience is None
            else scheduler_patience
        )
        if (
            isinstance(resolved_scheduler_patience, bool)
            or not isinstance(resolved_scheduler_patience, Integral)
            or int(resolved_scheduler_patience) < 1
            or int(resolved_scheduler_patience)
            >= self.early_stopping.patience
        ):
            raise ValueError(
                "scheduler_patience must be a positive integer smaller "
                "than early-stopping patience"
            )
        self.scheduler_patience = int(resolved_scheduler_patience)
        if train_generator is not None and not isinstance(
            train_generator, torch.Generator
        ):
            raise TypeError("train_generator must be a torch.Generator")
        self.train_generator = (
            make_dataloader_generator(self.seed)
            if train_generator is None
            else train_generator
        )
        self._train_generator_explicit = train_generator is not None

        self.model = model
        self.base_lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.log_interval = int(log_interval)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_start_factor = float(warmup_start_factor)
        self.gradient_accumulation_steps = int(
            gradient_accumulation_steps
        )
        self.loss_warmup_epochs = int(loss_warmup_epochs)
        self.validation_component_weights = (
            resolved_validation_weights
        )
        self.checkpoint_metadata = metadata
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        if self.warmup_epochs > 0:
            self._set_optimizer_lr(
                self.base_lr * self.warmup_start_factor
            )
        self.train_losses: List[Dict[str, float]] = []
        self.val_losses: List[Dict[str, float]] = []
        self.best_epoch = -1
        self.last_epoch = 0
        self._best_checkpoint: Optional[Dict[str, Any]] = None

    @staticmethod
    def _batch_input(batch) -> torch.Tensor:
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    @staticmethod
    def _average_losses(
        totals: Dict[str, float],
        n_samples: int,
    ) -> Dict[str, float]:
        if n_samples < 1:
            raise RuntimeError("Data loader produced no samples")
        return {
            key: value / float(n_samples)
            for key, value in totals.items()
        }

    @staticmethod
    def _format_losses(losses: Mapping[str, float]) -> str:
        fields = [
            "total",
            "recon",
            "freq",
            "diversity",
            "load_balance",
            "auxiliary_scale",
        ]
        return " | ".join(
            f"{field}={losses[field]:.4f}"
            for field in fields
            if field in losses
        )

    def _set_optimizer_lr(self, lr: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(lr)
        self.scheduler._last_lr = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]

    def _warmup_lr_after_epoch(self, epoch: int) -> float:
        if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
            return self.base_lr
        progress = float(epoch) / float(self.warmup_epochs)
        factor = self.warmup_start_factor + (
            1.0 - self.warmup_start_factor
        ) * progress
        return self.base_lr * factor

    def _loss_warmup_scale(self, epoch: int) -> float:
        if self.loss_warmup_epochs == 0:
            return 1.0
        return min(1.0, float(epoch) / float(self.loss_warmup_epochs))

    def _validate_recovery_training_config(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        resolved = checkpoint.get("resolved_config")
        training = (
            resolved.get("training")
            if isinstance(resolved, Mapping)
            else None
        )
        if not isinstance(training, Mapping):
            raise ValueError(
                "Recovery checkpoint resolved_config.training is missing"
            )
        expected = {
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_factor": self.warmup_start_factor,
            "gradient_accumulation_steps": (
                self.gradient_accumulation_steps
            ),
            "loss_warmup_epochs": self.loss_warmup_epochs,
        }
        for key, expected_value in expected.items():
            actual_value = training.get(key)
            if (
                isinstance(expected_value, float)
                and isinstance(actual_value, Real)
                and not isinstance(actual_value, bool)
            ):
                matches = float(actual_value) == expected_value
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(
                    "Recovery checkpoint training configuration does not "
                    f"match Trainer.{key}"
                )
        validation = resolved.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError(
                "Recovery checkpoint resolved_config.validation is missing"
            )
        actual_weights = validation.get("component_weights")
        if self.validation_component_weights is None:
            if actual_weights is not None:
                raise ValueError(
                    "Recovery checkpoint validation configuration does not "
                    "match Trainer.validation_component_weights"
                )
        elif actual_weights != self.validation_component_weights:
            raise ValueError(
                "Recovery checkpoint validation configuration does not "
                "match Trainer.validation_component_weights"
            )

    def _validation_monitor_value(
        self,
        losses: Mapping[str, float],
    ) -> float:
        if self.validation_component_weights is None:
            value = losses.get("total")
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    "Validation total loss must be finite"
                )
            return float(value)
        objective = 0.0
        for name, weight in self.validation_component_weights.items():
            value = losses.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    f"Validation component {name!r} must be finite"
                )
            objective += weight * float(value)
        if not np.isfinite(objective):
            raise FloatingPointError(
                "Validation monitoring objective is non-finite"
            )
        return objective

    @staticmethod
    def _accumulate_losses(
        totals: Dict[str, float],
        losses: Mapping[str, float],
        batch_size: int,
    ) -> None:
        for key, value in losses.items():
            numeric = float(value)
            if not np.isfinite(numeric):
                raise FloatingPointError(
                    f"Non-finite loss component {key!r}"
                )
            totals[key] = totals.get(key, 0.0) + numeric * batch_size

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        n_samples = 0
        total_batches = len(train_loader)
        if total_batches < 1:
            raise RuntimeError("Data loader produced no batches")
        optimizer_updates = 0
        group_sample_count = 0
        auxiliary_scale = self._loss_warmup_scale(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            x = self._batch_input(batch).to(self.device)
            output = self.model(x, compute_scores=False)
            if self.loss_warmup_epochs > 0:
                total_loss, loss_dict = self.model.compute_loss(
                    x,
                    output,
                    auxiliary_loss_scale=auxiliary_scale,
                )
            else:
                total_loss, loss_dict = self.model.compute_loss(x, output)
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(
                    "Non-finite training loss at "
                    f"epoch {epoch}, batch {batch_index}"
                )

            group_start = (
                batch_index // self.gradient_accumulation_steps
            ) * self.gradient_accumulation_steps
            group_size = min(
                self.gradient_accumulation_steps,
                total_batches - group_start,
            )
            batch_size = int(x.shape[0])
            if batch_size < 1:
                raise RuntimeError("Training batch must contain samples")
            (total_loss * float(batch_size)).backward()
            group_sample_count += batch_size
            group_position = batch_index - group_start + 1
            should_step = (
                group_position == group_size
                or batch_index + 1 == total_batches
            )
            if should_step:
                if group_sample_count < 1:
                    raise RuntimeError(
                        "Gradient accumulation group contained no samples"
                    )
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(group_sample_count))
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                        error_if_nonfinite=True,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                group_sample_count = 0

            self._accumulate_losses(totals, loss_dict, batch_size)
            n_samples += batch_size
            if batch_index % self.log_interval == 0:
                print(
                    f"  Epoch {epoch:4d} | Batch {batch_index:4d}/"
                    f"{len(train_loader)} | "
                    f"Accum {group_position}/{group_size} | "
                    f"{self._format_losses(loss_dict)}"
                )

        averaged = self._average_losses(totals, n_samples)
        self.train_losses.append(averaged)
        print(
            f"         Optimizer updates={optimizer_updates} | "
            f"accumulation={self.gradient_accumulation_steps}"
        )
        return averaged

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        n_samples = 0
        original_loader_generator = val_loader.generator
        val_loader.generator = make_dataloader_generator(
            self.validation_seed + 10_000_019
        )
        try:
            pass_index = 0
            for _round_index in range(self.validation_rounds):
                for batch in val_loader:
                    x = self._batch_input(batch).to(self.device)
                    mask_seed = self.validation_seed + pass_index
                    forward_masked = getattr(
                        self.model, "forward_masked", None
                    )
                    if not callable(forward_masked):
                        raise TypeError(
                            "Validation requires model.forward_masked("
                            "x, seed=..., compute_scores=False)"
                        )
                    forward_kwargs = {
                        "seed": mask_seed,
                        "compute_scores": False,
                    }
                    if x.shape[1] < int(
                        getattr(self.model, "time_len", x.shape[1])
                    ):
                        forward_kwargs["allow_shorter_time_length"] = True
                    output = forward_masked(x, **forward_kwargs)
                    total_loss, loss_dict = self.model.compute_loss(
                        x, output
                    )
                    if not bool(torch.isfinite(total_loss)):
                        raise FloatingPointError(
                            "Non-finite validation loss"
                        )
                    batch_size = int(x.shape[0])
                    self._accumulate_losses(
                        totals, loss_dict, batch_size
                    )
                    n_samples += batch_size
                    pass_index += 1
        finally:
            val_loader.generator = original_loader_generator

        averaged = self._average_losses(totals, n_samples)
        self.val_losses.append(averaged)
        return averaged

    def _reproducibility_state(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "validation_seed": self.validation_seed,
            "validation_rounds": self.validation_rounds,
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(
                torch.backends.cudnn.deterministic
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }

    def _validated_checkpoint_metadata(
        self,
        *,
        checkpoint_kind: str,
    ) -> Dict[str, Any]:
        metadata = copy.deepcopy(self.checkpoint_metadata)
        missing = sorted(_REQUIRED_CHECKPOINT_METADATA - set(metadata))
        if missing:
            raise ValueError(
                "Checkpoint metadata is incomplete; missing: "
                + ", ".join(missing)
            )
        threshold = metadata["calibrated_threshold"]
        if checkpoint_kind == "formal":
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, Real)
                or not np.isfinite(float(threshold))
            ):
                raise ValueError(
                    "formal calibrated_threshold must be finite"
                )
            metadata["calibrated_threshold"] = float(threshold)
        elif checkpoint_kind == "training_recovery":
            metadata["calibrated_threshold"] = None
            resolved = metadata.get("resolved_config")
            if isinstance(resolved, Mapping):
                resolved = copy.deepcopy(dict(resolved))
                calibration = resolved.get("calibration")
                if isinstance(calibration, Mapping):
                    calibration = dict(calibration)
                    calibration["threshold"] = None
                    resolved["calibration"] = calibration
                metadata["resolved_config"] = resolved
        else:
            raise ValueError(
                "checkpoint_kind must be 'formal' or 'training_recovery'"
            )
        metadata["seed"] = _validated_seed(metadata["seed"])
        if metadata["seed"] != self.seed:
            raise ValueError(
                "Checkpoint metadata seed must match Trainer seed"
            )
        if not isinstance(metadata["resolved_config"], Mapping):
            raise ValueError("resolved_config must be a mapping")
        if metadata["resolved_config"].get("seed") != self.seed:
            raise ValueError(
                "resolved_config.seed must match Trainer seed"
            )
        if not isinstance(metadata["git_revision"], str) or not metadata[
            "git_revision"
        ].strip():
            raise ValueError("git_revision must be a non-empty string")
        metadata.pop("reproducibility", None)
        return metadata

    def _checkpoint_payload(
        self,
        epoch: int,
        score: float,
        *,
        checkpoint_kind: str = "formal",
    ) -> Dict[str, Any]:
        if isinstance(epoch, bool) or not isinstance(epoch, Integral):
            raise ValueError("epoch must be a positive integer")
        if int(epoch) < 1:
            raise ValueError("epoch must be a positive integer")
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not np.isfinite(float(score))
        ):
            raise ValueError("checkpoint score must be finite")
        checkpoint = {
            "checkpoint_kind": checkpoint_kind,
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "score": float(score),
            "model_config": self.model.get_config(),
            "reproducibility": self._reproducibility_state(),
            "dataloader_generator_state": (
                self.train_generator.get_state().detach().cpu().clone()
            ),
            "trainer_state": {
                "early_stopping": self.early_stopping.state_dict(),
                "best_epoch": self.best_epoch,
                "train_losses": copy.deepcopy(self.train_losses),
                "val_losses": copy.deepcopy(self.val_losses),
                "last_epoch": self.last_epoch,
                "dataloader_generator_state": (
                    self.train_generator.get_state()
                    .detach()
                    .cpu()
                    .clone()
                ),
            },
            **self._validated_checkpoint_metadata(
                checkpoint_kind=checkpoint_kind
            ),
        }
        return _weights_only_safe(checkpoint)

    def _restore_checkpoint_payload(
        self,
        checkpoint: Mapping[str, Any],
        *,
        restore_rng: bool,
        restore_trainer_state: bool = True,
    ) -> None:
        self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        generator_state = checkpoint["dataloader_generator_state"]
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
            or generator_state.numel() == 0
        ):
            raise ValueError(
                "dataloader_generator_state must be a non-empty "
                "1D uint8 tensor"
            )
        self.train_generator.set_state(generator_state.detach().cpu())
        trainer_state = checkpoint.get("trainer_state")
        if not isinstance(trainer_state, Mapping):
            raise ValueError("trainer_state must be a mapping")
        expected_trainer_keys = {
            "early_stopping",
            "best_epoch",
            "train_losses",
            "val_losses",
            "last_epoch",
            "dataloader_generator_state",
        }
        if set(trainer_state) != expected_trainer_keys:
            raise ValueError("trainer_state schema is invalid")
        trainer_generator_state = trainer_state[
            "dataloader_generator_state"
        ]
        if (
            not isinstance(trainer_generator_state, torch.Tensor)
            or not torch.equal(
                trainer_generator_state.detach().cpu(),
                generator_state.detach().cpu(),
            )
        ):
            raise ValueError(
                "trainer_state DataLoader generator state must match "
                "the checkpoint generator state"
            )
        metadata = {
            key: copy.deepcopy(value)
            for key, value in checkpoint.items()
            if key not in _CHECKPOINT_STATE_KEYS
        }
        self.checkpoint_metadata = metadata
        self.seed = _validated_seed(checkpoint["seed"])
        reproducibility = checkpoint["reproducibility"]
        self.validation_seed = _validated_seed(
            reproducibility["validation_seed"],
            name="validation_seed",
        )
        self.validation_rounds = int(
            reproducibility["validation_rounds"]
        )
        if restore_trainer_state:
            self.early_stopping.load_state_dict(
                trainer_state["early_stopping"]
            )
            best_epoch = trainer_state["best_epoch"]
            last_epoch = trainer_state["last_epoch"]
            if (
                type(best_epoch) is not int
                or best_epoch < -1
                or type(last_epoch) is not int
                or last_epoch < 0
            ):
                raise ValueError(
                    "trainer_state best_epoch/last_epoch are invalid"
                )
            if best_epoch > last_epoch:
                raise ValueError(
                    "trainer_state best_epoch cannot exceed last_epoch"
                )
            if not isinstance(trainer_state["train_losses"], list) or not (
                isinstance(trainer_state["val_losses"], list)
            ):
                raise ValueError(
                    "trainer_state histories must be lists"
                )
            self.best_epoch = best_epoch
            self.last_epoch = last_epoch
            self.train_losses = copy.deepcopy(
                trainer_state["train_losses"]
            )
            self.val_losses = copy.deepcopy(
                trainer_state["val_losses"]
            )
        if restore_rng:
            restore_rng_state(
                checkpoint["rng_state"],
                reproducibility,
            )

    @staticmethod
    def _write_checkpoint(path: str, checkpoint: Mapping[str, Any]) -> None:
        atomic_torch_save(checkpoint, path)
        print(f"  Checkpoint saved to {path}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: Optional[str] = None,
        save_best: bool = True,
        start_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(epochs, bool) or not isinstance(epochs, Integral):
            raise ValueError("epochs must be a positive integer")
        if int(epochs) < 1:
            raise ValueError("epochs must be a positive integer")
        resolved_start_epoch = (
            self.last_epoch + 1
            if start_epoch is None
            else start_epoch
        )
        if (
            isinstance(resolved_start_epoch, bool)
            or not isinstance(resolved_start_epoch, Integral)
            or int(resolved_start_epoch) < 1
        ):
            raise ValueError("start_epoch must be a positive integer")
        resolved_start_epoch = int(resolved_start_epoch)
        if self.last_epoch > 0 and resolved_start_epoch != self.last_epoch + 1:
            raise ValueError(
                "start_epoch must continue immediately after last_epoch"
            )
        if resolved_start_epoch > int(epochs):
            raise ValueError("start_epoch cannot exceed epochs")
        started = time.time()
        loader_generator = getattr(train_loader, "generator", None)
        if loader_generator is not None:
            if (
                self._train_generator_explicit
                and loader_generator is not self.train_generator
            ):
                raise ValueError(
                    "Trainer train_generator must be the same generator "
                    "used by the training DataLoader"
                )
            self.train_generator = loader_generator
        elif self._train_generator_explicit:
            raise ValueError(
                "Training DataLoader must use Trainer train_generator"
            )

        for epoch in range(resolved_start_epoch, int(epochs) + 1):
            self.last_epoch = epoch
            epoch_started = time.time()
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self.train_epoch(train_loader, epoch)
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"{self._format_losses(train_loss)} | "
                f"lr={epoch_lr:.8g} | "
                f"time={time.time() - epoch_started:.1f}s"
            )

            monitored = train_loss["total"]
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                monitored = self._validation_monitor_value(val_loss)
                print(f"         Val | {self._format_losses(val_loss)}")
                print(
                    "         Monitor | "
                    f"{monitored:.6f}"
                )
            if epoch <= self.warmup_epochs:
                self._set_optimizer_lr(
                    self._warmup_lr_after_epoch(epoch)
                )
            else:
                self.scheduler.step(monitored)

            should_stop = self.early_stopping(monitored)
            if self.early_stopping.improved:
                self.best_epoch = epoch
                self._best_checkpoint = self._checkpoint_payload(
                    epoch,
                    monitored,
                    checkpoint_kind="training_recovery",
                )
                if save_best and save_dir:
                    self._write_checkpoint(
                        os.path.join(save_dir, "recovery_best.pt"),
                        self._best_checkpoint,
                    )
            latest_checkpoint = self._checkpoint_payload(
                epoch,
                monitored,
                checkpoint_kind="training_recovery",
            )
            if save_best and save_dir:
                self._write_checkpoint(
                    os.path.join(save_dir, "recovery_latest.pt"),
                    latest_checkpoint,
                )
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if self._best_checkpoint is None:
            raise RuntimeError("Training produced no best checkpoint state")
        self._restore_checkpoint_payload(
            self._best_checkpoint,
            restore_rng=True,
            restore_trainer_state=False,
        )
        print(
            f"Training completed in {time.time() - started:.1f}s. "
            f"Best epoch: {self.best_epoch}"
        )
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "last_epoch": self.last_epoch,
            "best_score": self.early_stopping.best_score,
        }

    def save_checkpoint(self, path: str, epoch: int, score: float) -> None:
        if self.last_epoch < int(epoch):
            self.last_epoch = int(epoch)
        checkpoint = self._checkpoint_payload(
            epoch,
            score,
            checkpoint_kind="formal",
        )
        self._write_checkpoint(path, checkpoint)

    def save_best_checkpoint(
        self,
        path: str,
        *,
        metadata_updates: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Save the frozen best state with post-training metadata updates."""
        if self._best_checkpoint is None:
            raise RuntimeError("No best checkpoint state is available")
        checkpoint = copy.deepcopy(self._best_checkpoint)
        updates = dict(metadata_updates or {})
        forbidden = sorted(set(updates) & _CHECKPOINT_STATE_KEYS)
        if forbidden:
            raise ValueError(
                "metadata_updates cannot replace frozen best state: "
                + ", ".join(forbidden)
            )
        checkpoint["checkpoint_kind"] = "formal"
        checkpoint.update(_weights_only_safe(updates))
        threshold = checkpoint.get("calibrated_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not np.isfinite(float(threshold))
        ):
            raise ValueError(
                "calibrated_threshold must be finite"
            )
        if not isinstance(checkpoint.get("resolved_config"), Mapping):
            raise ValueError("resolved_config must be a mapping")
        self._write_checkpoint(path, checkpoint)
        return checkpoint

    def load_checkpoint(
        self,
        path: str,
        *,
        restore_rng: bool = True,
    ) -> Mapping[str, Any]:
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                "Checkpoint is not compatible with "
                "torch.load(weights_only=True)"
            ) from exc
        if type(checkpoint) is not dict:
            raise ValueError("Checkpoint root must be a dictionary")
        from specmoe.model.loading import validate_checkpoint_for_model

        validate_checkpoint_for_model(
            checkpoint,
            self.model,
            expected_kind="training_recovery",
        )
        self._validate_recovery_training_config(checkpoint)
        self._restore_checkpoint_payload(
            checkpoint,
            restore_rng=restore_rng,
        )
        if self.best_epoch == self.last_epoch:
            self._best_checkpoint = copy.deepcopy(checkpoint)
        else:
            best_path = os.path.join(
                os.path.dirname(os.path.abspath(path)),
                "recovery_best.pt",
            )
            if not os.path.isfile(best_path):
                raise FileNotFoundError(
                    "Recovery checkpoint references an earlier best epoch, "
                    f"but {best_path} is missing"
                )
            best_checkpoint = torch.load(
                best_path,
                map_location=self.device,
                weights_only=True,
            )
            if type(best_checkpoint) is not dict:
                raise ValueError(
                    "Best recovery checkpoint root must be a dictionary"
                )
            validate_checkpoint_for_model(
                best_checkpoint,
                self.model,
                expected_kind="training_recovery",
            )
            if (
                best_checkpoint["trainer_state"]["last_epoch"]
                != self.best_epoch
            ):
                raise ValueError(
                    "Best recovery checkpoint epoch does not match "
                    "trainer_state.best_epoch"
                )
            self._best_checkpoint = copy.deepcopy(best_checkpoint)
        print(
            f"Checkpoint loaded from {path} "
            f"(epoch {checkpoint['epoch']})"
        )
        return checkpoint
