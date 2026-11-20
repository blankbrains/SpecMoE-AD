"""M2B-MoE v3 for frequency-aware time-series anomaly detection.

The model encodes rFFT patches along the frequency axis, applies cross-variable
attention separately, routes every frequency patch to heterogeneous
reconstruction branches, overlap-averages patch routing weights onto rFFT
bins, and fuses the branch spectra before the inverse FFT.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specmoe.data.preprocess import RevIN


_DIVERSITY_MODES = frozenset({"none", "cosine", "contrastive"})
_ROUTER_MODES = frozenset({"global", "variable"})
_EXPERT_SPECIALIZATION_MODES = frozenset({"none", "frequency_bands"})
_MASK_MODES = frozenset({"random", "contiguous"})
_FREQUENCY_SCORE_AGGREGATIONS = frozenset({"mean", "max"})
_UNSET = object()


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_float(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    converted = float(value)
    if not torch.isfinite(torch.tensor(converted)) or converted < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return converted


def _strict_bool(
    name: str, value: Any
) -> Optional[bool]:
    if value is _UNSET:
        return None
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _assert_finite(name: str, value: Any) -> None:
    """Fail fast when a tensor tree contains NaN or Inf."""
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains non-finite NaN/Inf values")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(f"{name}.{key}", child)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(f"{name}[{index}]", child)


class CrossVariableAttention(nn.Module):
    """Self-attention over variables for one frequency patch at a time."""

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attended)


class BottleneckBranch(nn.Module):
    """Heterogeneous bottleneck branch that reconstructs a full rFFT."""

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        freq_bins: int,
        num_patches: int,
        branch_id: int,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.branch_id = branch_id
        self.freq_bins = freq_bins
        self.num_patches = num_patches
        self.encoder = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, d_model),
        )
        reconstruction_width = d_model * num_patches
        hidden_width = max(reconstruction_width // 2, freq_bins * 4)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(reconstruction_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, freq_bins * 2),
        )

    def forward(
        self,
        shared: torch.Tensor,
        *,
        output_freq_bins: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_vars, num_patches, d_model = shared.shape
        flat = shared.reshape(batch_size * num_vars * num_patches, d_model)
        latent = self.encoder(flat)
        decoded = self.decoder(latent).reshape(
            batch_size, num_vars, num_patches, d_model
        )
        decoded_for_head = decoded
        if num_patches != self.num_patches:
            decoded_for_head = F.interpolate(
                decoded.permute(0, 1, 3, 2).reshape(
                    batch_size * num_vars,
                    d_model,
                    num_patches,
                ),
                size=self.num_patches,
                mode="linear",
                align_corners=False,
            ).reshape(
                batch_size,
                num_vars,
                d_model,
                self.num_patches,
            ).permute(0, 1, 3, 2)
        coefficients = self.reconstruction_head(
            decoded_for_head.reshape(
                batch_size,
                num_vars,
                self.num_patches * d_model,
            )
        )
        resolved_freq_bins = (
            self.freq_bins
            if output_freq_bins is None
            else _positive_int("output_freq_bins", output_freq_bins)
        )
        if resolved_freq_bins != self.freq_bins:
            coefficients = F.interpolate(
                coefficients.reshape(
                    batch_size * num_vars,
                    2,
                    self.freq_bins,
                ),
                size=resolved_freq_bins,
                mode="linear",
                align_corners=False,
            ).reshape(batch_size, num_vars, 2 * resolved_freq_bins)
        spectrum = torch.complex(
            coefficients[..., :resolved_freq_bins],
            coefficients[..., resolved_freq_bins:],
        )
        return spectrum, latent.reshape(
            batch_size,
            num_vars,
            num_patches,
            self.bottleneck_dim,
        )


class SpectralRouter(nn.Module):
    """Produce expert weights independently for every frequency patch."""

    min_temperature = 0.05
    max_temperature = 5.0

    def __init__(
        self,
        d_model: int,
        num_branches: int,
        hidden_dim: int = 32,
        learnable_temp: bool = False,
        mode: str = "global",
    ):
        super().__init__()
        if type(mode) is not str or mode not in _ROUTER_MODES:
            raise ValueError(
                "mode must be one of: " + ", ".join(sorted(_ROUTER_MODES))
            )
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_branches),
        )
        self.learnable_temp = learnable_temp
        self.mode = mode
        if learnable_temp:
            self.raw_temperature = nn.Parameter(torch.zeros(()))

    def current_temperature(self) -> torch.Tensor:
        reference = self.network[0].weight
        if not self.learnable_temp:
            return reference.new_tensor(1.0)
        temperature_range = self.max_temperature - self.min_temperature
        return self.min_temperature + temperature_range * torch.sigmoid(
            self.raw_temperature
        )

    def forward(
        self, shared: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_features = (
            shared.mean(dim=1)
            if self.mode == "global"
            else shared
        )
        logits = self.network(patch_features)
        temperature = self.current_temperature().to(dtype=logits.dtype)
        weights = F.softmax(logits / temperature, dim=-1)
        return weights, logits


class M2BMoEModel(nn.Module):
    """Multi-branch bottleneck spectral mixture of experts, version 3."""

    def __init__(
        self,
        time_len: int = 200,
        num_vars: int = 3,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 32,
        enc_hidden: int = 64,
        bottleneck_dims: Optional[List[int]] = None,
        cross_var_heads: int = 4,
        use_revin: bool = False,
        input_mask_ratio: float = 0.5,
        mask_mode: str = "random",
        lambda_freq: float = 1.0,
        router_hidden: int = 32,
        router_mode: str = "global",
        expert_specialization: str = "none",
        diversity_weight: float = 0.1,
        load_balance_weight: float = 0.01,
        grad_weight: float = 0.3,
        score_freq_weight: float = 0.3,
        disagreement_weight: float = 0.1,
        diversity_mode: Optional[str] = None,
        use_contrastive_div: Any = _UNSET,
        use_grad_score: bool = False,
        use_adaptive_patches: bool = False,
        use_learnable_temp: bool = False,
        stft_window_sizes: Optional[Sequence[int]] = None,
        stft_hop_ratio: float = 0.25,
        frequency_score_aggregation: str = "mean",
    ):
        super().__init__()
        self.time_len = _positive_int("time_len", time_len)
        if self.time_len < 2:
            raise ValueError("time_len must be at least 2")
        self.num_vars = _positive_int("num_vars", num_vars)
        self.patch_len = _positive_int("patch_len", patch_len)
        self.patch_stride = _positive_int("patch_stride", patch_stride)
        if self.patch_stride > self.patch_len:
            raise ValueError("patch_stride cannot exceed patch_len")
        self.d_model = _positive_int("d_model", d_model)
        self.enc_hidden = _positive_int("enc_hidden", enc_hidden)
        self.cross_var_heads = _positive_int(
            "cross_var_heads", cross_var_heads
        )
        if self.d_model % self.cross_var_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"cross_var_heads ({self.cross_var_heads})"
            )
        self.router_hidden = _positive_int("router_hidden", router_hidden)
        if type(router_mode) is not str or router_mode not in _ROUTER_MODES:
            raise ValueError(
                "router_mode must be one of: "
                + ", ".join(sorted(_ROUTER_MODES))
            )
        self.router_mode = router_mode
        if (
            type(expert_specialization) is not str
            or expert_specialization not in _EXPERT_SPECIALIZATION_MODES
        ):
            raise ValueError(
                "expert_specialization must be one of: "
                + ", ".join(sorted(_EXPERT_SPECIALIZATION_MODES))
            )
        self.expert_specialization = expert_specialization

        resolved_bottlenecks = (
            [2, 8, 32] if bottleneck_dims is None else list(bottleneck_dims)
        )
        if not resolved_bottlenecks:
            raise ValueError("bottleneck_dims cannot be empty")
        self.bottleneck_dims = [
            _positive_int("bottleneck_dim", dimension)
            for dimension in resolved_bottlenecks
        ]
        self.num_branches = len(self.bottleneck_dims)

        if isinstance(input_mask_ratio, bool) or not isinstance(
            input_mask_ratio, Real
        ):
            raise ValueError("input_mask_ratio must be in [0, 1]")
        self.input_mask_ratio = float(input_mask_ratio)
        if (
            not torch.isfinite(torch.tensor(self.input_mask_ratio))
            or not 0.0 <= self.input_mask_ratio <= 1.0
        ):
            raise ValueError("input_mask_ratio must be in [0, 1]")
        if type(mask_mode) is not str or mask_mode not in _MASK_MODES:
            raise ValueError(
                "mask_mode must be one of: " + ", ".join(sorted(_MASK_MODES))
            )
        self.mask_mode = mask_mode

        self.lambda_freq = _nonnegative_float("lambda_freq", lambda_freq)
        self.diversity_weight = _nonnegative_float(
            "diversity_weight", diversity_weight
        )
        self.load_balance_weight = _nonnegative_float(
            "load_balance_weight", load_balance_weight
        )
        self.grad_weight = _nonnegative_float("grad_weight", grad_weight)
        self.score_freq_weight = _nonnegative_float(
            "score_freq_weight", score_freq_weight
        )
        self.disagreement_weight = _nonnegative_float(
            "disagreement_weight", disagreement_weight
        )
        if (
            isinstance(stft_hop_ratio, bool)
            or not isinstance(stft_hop_ratio, Real)
            or not 0.0 < float(stft_hop_ratio) <= 1.0
        ):
            raise ValueError("stft_hop_ratio must be in (0, 1]")
        self.stft_hop_ratio = float(stft_hop_ratio)
        if (
            type(frequency_score_aggregation) is not str
            or frequency_score_aggregation
            not in _FREQUENCY_SCORE_AGGREGATIONS
        ):
            raise ValueError(
                "frequency_score_aggregation must be one of: "
                + ", ".join(sorted(_FREQUENCY_SCORE_AGGREGATIONS))
            )
        self.frequency_score_aggregation = frequency_score_aggregation

        legacy_contrastive = _strict_bool(
            "use_contrastive_div", use_contrastive_div
        )
        if diversity_mode is None:
            resolved_diversity_mode = (
                "contrastive" if legacy_contrastive else "cosine"
            )
        else:
            if type(diversity_mode) is not str:
                raise ValueError("diversity_mode must be a string")
            resolved_diversity_mode = diversity_mode
        if resolved_diversity_mode not in _DIVERSITY_MODES:
            raise ValueError(
                "diversity_mode must be one of: "
                + ", ".join(sorted(_DIVERSITY_MODES))
            )
        expected_legacy_flag = resolved_diversity_mode == "contrastive"
        if (
            legacy_contrastive is not None
            and legacy_contrastive != expected_legacy_flag
        ):
            raise ValueError(
                "diversity_mode conflicts with use_contrastive_div"
            )
        self.diversity_mode = resolved_diversity_mode
        self.use_contrastive_div = expected_legacy_flag
        self.use_revin = _strict_bool("use_revin", use_revin)
        self.use_grad_score = _strict_bool(
            "use_grad_score", use_grad_score
        )
        self.use_adaptive_patches = _strict_bool(
            "use_adaptive_patches", use_adaptive_patches
        )
        self.use_learnable_temp = _strict_bool(
            "use_learnable_temp", use_learnable_temp
        )

        self.full_freq_bins = self.time_len // 2 + 1
        self.register_buffer(
            "expert_band_prior",
            self._build_expert_band_prior(self.full_freq_bins),
            persistent=False,
        )
        patch_rows, patch_groups = self._build_patch_layout()
        self.full_num_patches = len(patch_rows)
        self._patch_groups = patch_groups

        max_patch_width = max(len(row) for row in patch_rows)
        patch_bin_indices = torch.full(
            (self.full_num_patches, max_patch_width),
            -1,
            dtype=torch.long,
        )
        patch_coverage_matrix = torch.zeros(
            self.full_num_patches,
            self.full_freq_bins,
            dtype=torch.float32,
        )
        for patch_index, bins in enumerate(patch_rows):
            patch_bin_indices[patch_index, : len(bins)] = torch.tensor(
                bins, dtype=torch.long
            )
            patch_coverage_matrix[patch_index, bins] = 1.0
        if not bool((patch_coverage_matrix.sum(dim=0) > 0).all()):
            raise RuntimeError("Frequency patch layout does not cover every rFFT bin")
        self.register_buffer(
            "patch_bin_indices", patch_bin_indices, persistent=False
        )
        self.register_buffer(
            "patch_coverage_matrix",
            patch_coverage_matrix,
            persistent=False,
        )

        if self.use_adaptive_patches:
            self.band_num_patches = [
                group_stop - group_start
                for group_start, group_stop, _ in self._patch_groups
            ]
            self.band_patch_lens = [
                width for _, _, width in self._patch_groups
            ]
            self.band_embeds = nn.ModuleList(
                [
                    nn.Linear(width * 2, self.d_model)
                    for _, _, width in self._patch_groups
                ]
            )
            self.band_pos = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            1,
                            1,
                            stop - start,
                            self.d_model,
                        )
                        * 0.02
                    )
                    for start, stop, _ in self._patch_groups
                ]
            )
            self.band_mask_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, 1, 1, width * 2))
                    for _, _, width in self._patch_groups
                ]
            )
        else:
            self.patch_embed = nn.Linear(self.patch_len * 2, self.d_model)
            self.patch_pos = nn.Parameter(
                torch.randn(
                    1, 1, self.full_num_patches, self.d_model
                )
                * 0.02
            )
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, 1, self.patch_len * 2)
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.cross_var_heads,
            dim_feedforward=self.enc_hidden,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.patch_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.cross_var = CrossVariableAttention(
            self.d_model, self.cross_var_heads
        )
        self.branches = nn.ModuleList(
            [
                BottleneckBranch(
                    self.d_model,
                    bottleneck_dim,
                    self.full_freq_bins,
                    self.full_num_patches,
                    branch_id,
                )
                for branch_id, bottleneck_dim in enumerate(
                    self.bottleneck_dims
                )
            ]
        )
        self.router = SpectralRouter(
            self.d_model,
            self.num_branches,
            self.router_hidden,
            self.use_learnable_temp,
            self.router_mode,
        )
        self.revin = RevIN(
            num_features=self.num_vars, eps=1e-5, affine=False
        )
        self.register_buffer(
            "contrast_temp", torch.tensor(0.07), persistent=False
        )
        self.stft_window_sizes = self._resolve_stft_window_sizes(
            stft_window_sizes
        )
        for index, window_size in enumerate(self.stft_window_sizes):
            self.register_buffer(
                f"_stft_window_{index}",
                torch.hann_window(window_size),
                persistent=False,
            )

    @staticmethod
    def _cover_end_starts(
        length: int, patch_width: int, stride: int
    ) -> List[int]:
        if patch_width > length:
            raise ValueError(
                f"patch width {patch_width} exceeds available length {length}"
            )
        starts = list(range(0, length - patch_width + 1, stride))
        final_start = length - patch_width
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def _build_patch_layout(
        self,
        full_freq_bins: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[Tuple[int, int, int]]]:
        resolved_freq_bins = (
            self.full_freq_bins
            if full_freq_bins is None
            else _positive_int("full_freq_bins", full_freq_bins)
        )
        if not self.use_adaptive_patches:
            if self.patch_len > resolved_freq_bins:
                raise ValueError(
                    f"patch_len ({self.patch_len}) exceeds rFFT bins "
                    f"({resolved_freq_bins})"
                )
            starts = self._cover_end_starts(
                resolved_freq_bins,
                self.patch_len,
                self.patch_stride,
            )
            rows = [
                list(range(start, start + self.patch_len))
                for start in starts
            ]
            return rows, [(0, len(rows), self.patch_len)]

        if resolved_freq_bins < 3:
            raise ValueError(
                "Adaptive frequency patching requires at least three rFFT bins"
            )
        boundaries = [
            0,
            resolved_freq_bins // 3,
            2 * resolved_freq_bins // 3,
            resolved_freq_bins,
        ]
        requested_widths = [
            self.patch_len + 8,
            self.patch_len,
            max(self.patch_len - 4, 4),
        ]
        requested_strides = [
            self.patch_stride + 4,
            self.patch_stride,
            max(self.patch_stride - 2, 2),
        ]
        rows: List[List[int]] = []
        groups: List[Tuple[int, int, int]] = []
        for band_index in range(3):
            band_start = boundaries[band_index]
            band_stop = boundaries[band_index + 1]
            band_width = band_stop - band_start
            patch_width = min(requested_widths[band_index], band_width)
            stride = min(requested_strides[band_index], patch_width)
            local_starts = self._cover_end_starts(
                band_width, patch_width, stride
            )
            group_start = len(rows)
            rows.extend(
                [
                    list(
                        range(
                            band_start + local_start,
                            band_start + local_start + patch_width,
                        )
                    )
                    for local_start in local_starts
                ]
            )
            groups.append((group_start, len(rows), patch_width))
        return rows, groups

    def _build_expert_band_prior(self, freq_bins: int) -> torch.Tensor:
        """Build normalized overlapping expert feasibility bands."""
        freq_bins = _positive_int("freq_bins", freq_bins)
        if self.num_branches == 1:
            return torch.ones(freq_bins, 1, dtype=torch.float32)
        positions = torch.linspace(0.0, 1.0, freq_bins)
        centers = torch.linspace(0.0, 1.0, self.num_branches)
        spacing = 1.0 / float(self.num_branches - 1)
        prior = (
            1.0
            - (positions[:, None] - centers[None, :]).abs() / spacing
        ).clamp_min(0.0)
        if not bool((prior.sum(dim=-1) > 0.0).all()):
            raise RuntimeError(
                "Expert frequency-band prior does not cover every rFFT bin"
            )
        return prior / prior.sum(dim=-1, keepdim=True)

    @staticmethod
    def _patch_layout_tensors(
        patch_rows: Sequence[Sequence[int]],
        full_freq_bins: int,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_patch_width = max(len(row) for row in patch_rows)
        patch_bin_indices = torch.full(
            (len(patch_rows), max_patch_width),
            -1,
            dtype=torch.long,
            device=device,
        )
        patch_coverage_matrix = torch.zeros(
            len(patch_rows),
            full_freq_bins,
            dtype=torch.float32,
            device=device,
        )
        for patch_index, bins in enumerate(patch_rows):
            patch_bin_indices[patch_index, : len(bins)] = torch.tensor(
                bins,
                dtype=torch.long,
                device=device,
            )
            patch_coverage_matrix[patch_index, bins] = 1.0
        return patch_bin_indices, patch_coverage_matrix

    def _runtime_patch_layout(
        self,
        time_len: int,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int]]]:
        runtime_freq_bins = _positive_int("runtime_time_len", time_len) // 2 + 1
        if runtime_freq_bins == self.full_freq_bins:
            return (
                self.patch_bin_indices.to(device=device),
                self.patch_coverage_matrix.to(device=device),
                list(self._patch_groups),
            )
        patch_rows, patch_groups = self._build_patch_layout(runtime_freq_bins)
        patch_bin_indices, patch_coverage_matrix = self._patch_layout_tensors(
            patch_rows,
            runtime_freq_bins,
            device=device,
        )
        if not bool((patch_coverage_matrix.sum(dim=0) > 0).all()):
            raise RuntimeError(
                "Runtime frequency patch layout does not cover every rFFT bin"
            )
        return patch_bin_indices, patch_coverage_matrix, patch_groups

    def _runtime_expert_band_prior(
        self,
        freq_bins: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the specialization prior at the current rFFT resolution."""
        freq_bins = _positive_int("freq_bins", freq_bins)
        prior = self.expert_band_prior
        if freq_bins != self.full_freq_bins:
            prior = F.interpolate(
                prior.transpose(0, 1).unsqueeze(0),
                size=freq_bins,
                mode="linear",
                align_corners=True,
            ).squeeze(0).transpose(0, 1)
        prior = prior.to(dtype=dtype, device=device).clamp_min(0.0)
        return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _apply_expert_specialization(
        self,
        routing_bin_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Restrict router weights to the selected expert frequency bands."""
        if self.expert_specialization == "none":
            return routing_bin_weights
        if self.expert_specialization != "frequency_bands":
            raise RuntimeError("Unsupported expert specialization mode")
        if routing_bin_weights.ndim not in {3, 4}:
            raise ValueError(
                "routing_bin_weights must have shape [B, F, K] or "
                "[B, V, F, K]"
            )
        freq_bins = routing_bin_weights.shape[-2]
        prior = self._runtime_expert_band_prior(
            freq_bins,
            dtype=routing_bin_weights.dtype,
            device=routing_bin_weights.device,
        )
        prior_shape = (
            (1, freq_bins, self.num_branches)
            if routing_bin_weights.ndim == 3
            else (1, 1, freq_bins, self.num_branches)
        )
        expanded_prior = prior.reshape(prior_shape)
        restricted = routing_bin_weights * expanded_prior
        denominator = restricted.sum(dim=-1, keepdim=True)
        normalized = restricted / denominator.clamp_min(1e-8)
        return torch.where(
            denominator > 1e-8,
            normalized,
            expanded_prior.expand_as(routing_bin_weights),
        )

    def _routing_sample_weights(
        self,
        routing_weights: torch.Tensor,
        routing_bin_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the actual routing allocation to one expert weight per sample."""
        if self.expert_specialization == "frequency_bands":
            return routing_bin_weights.mean(
                dim=(1, 2) if self.router_mode == "variable" else 1
            )
        return routing_weights.mean(
            dim=(1, 2) if self.router_mode == "variable" else 1
        )

    def _resolve_stft_window_sizes(
        self, requested: Optional[Sequence[int]]
    ) -> List[int]:
        candidates = [8, 16, 32] if requested is None else list(requested)
        if not candidates:
            raise ValueError("stft_window_sizes cannot be empty")
        resolved: List[int] = []
        for raw_size in candidates:
            size = _positive_int("stft_window_size", raw_size)
            size = min(size, self.time_len)
            if size < 2:
                raise ValueError("STFT window sizes must be at least 2")
            if size not in resolved:
                resolved.append(size)
        return resolved

    def _validate_input(
        self,
        x: torch.Tensor,
        *,
        check_finite: bool = True,
        allow_shorter_time_length: bool = False,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("M2BMoEModel input must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, T, N], got {tuple(x.shape)}"
            )
        if x.shape[0] < 1:
            raise ValueError("Input batch cannot be empty")
        if not isinstance(allow_shorter_time_length, bool):
            raise TypeError("allow_shorter_time_length must be boolean")
        valid_time_length = x.shape[1] == self.time_len or (
            allow_shorter_time_length and 2 <= x.shape[1] < self.time_len
        )
        if not valid_time_length:
            raise ValueError(
                f"Expected time length {self.time_len}, got {x.shape[1]}. "
                "Shorter inputs require allow_shorter_time_length=True"
            )
        if x.shape[2] != self.num_vars:
            raise ValueError(
                f"Expected {self.num_vars} variables, got {x.shape[2]}"
            )
        if not x.is_floating_point():
            raise TypeError("M2BMoEModel input must use a floating-point dtype")
        if check_finite:
            _assert_finite("input", x)

    def _resolve_generator(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator],
        seed: Optional[int],
    ) -> Optional[torch.Generator]:
        if generator is not None and seed is not None:
            raise ValueError("Provide either mask seed or generator, not both")
        if seed is None:
            return generator
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("mask seed must be a non-negative integer")
        return torch.Generator(device=x.device).manual_seed(seed)

    def _extract_patch_group(
        self,
        spectrum: torch.Tensor,
        patch_bin_indices: torch.Tensor,
        group_start: int,
        group_stop: int,
        patch_width: int,
    ) -> torch.Tensor:
        indices = patch_bin_indices[
            group_start:group_stop, :patch_width
        ]
        complex_patches = spectrum[:, :, indices]
        return torch.cat(
            [complex_patches.real, complex_patches.imag], dim=-1
        )

    def _sample_group_mask(
        self,
        *,
        batch_size: int,
        patch_count: int,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.mask_mode == "random":
            return (
                torch.rand(
                    batch_size,
                    1,
                    patch_count,
                    1,
                    device=device,
                    generator=generator,
                )
                < self.input_mask_ratio
            )

        masked_count = max(
            1,
            int(round(patch_count * self.input_mask_ratio)),
        )
        masked_count = min(masked_count, patch_count)
        max_start = patch_count - masked_count
        starts = torch.randint(
            0,
            max_start + 1,
            (batch_size, 1),
            device=device,
            generator=generator,
        )
        patch_positions = torch.arange(
            patch_count, device=device
        ).unsqueeze(0)
        mask = (patch_positions >= starts) & (
            patch_positions < starts + masked_count
        )
        return mask.unsqueeze(1).unsqueeze(-1)

    def _embed_patches(
        self,
        spectrum: torch.Tensor,
        apply_mask: bool,
        generator: Optional[torch.Generator],
        patch_bin_indices: torch.Tensor,
        patch_groups: Sequence[Tuple[int, int, int]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = spectrum.shape[0]
        embedded_groups: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for group_index, (
            group_start,
            group_stop,
            patch_width,
        ) in enumerate(patch_groups):
            patches = self._extract_patch_group(
                spectrum,
                patch_bin_indices,
                group_start,
                group_stop,
                patch_width,
            )
            group_mask: Optional[torch.Tensor] = None
            if apply_mask and self.input_mask_ratio > 0.0:
                group_mask = self._sample_group_mask(
                    batch_size=batch_size,
                    patch_count=group_stop - group_start,
                    device=spectrum.device,
                    generator=generator,
                )
                mask_token = (
                    self.band_mask_tokens[group_index]
                    if self.use_adaptive_patches
                    else self.mask_token
                )
                patches = torch.where(group_mask, mask_token, patches)
            elif apply_mask:
                group_mask = torch.zeros(
                    batch_size,
                    1,
                    group_stop - group_start,
                    1,
                    dtype=torch.bool,
                    device=spectrum.device,
                )

            if self.use_adaptive_patches:
                position = self.band_pos[group_index]
                if position.shape[2] != group_stop - group_start:
                    position = F.interpolate(
                        position.permute(0, 1, 3, 2).reshape(
                            1,
                            self.d_model,
                            position.shape[2],
                        ),
                        size=group_stop - group_start,
                        mode="linear",
                        align_corners=False,
                    ).reshape(
                        1,
                        1,
                        self.d_model,
                        group_stop - group_start,
                    ).permute(0, 1, 3, 2)
                embedded = (
                    self.band_embeds[group_index](patches)
                    + position
                )
            else:
                position = self.patch_pos
                if position.shape[2] != patch_bin_indices.shape[0]:
                    position = F.interpolate(
                        position.permute(0, 1, 3, 2).reshape(
                            1,
                            self.d_model,
                            position.shape[2],
                        ),
                        size=patch_bin_indices.shape[0],
                        mode="linear",
                        align_corners=False,
                    ).reshape(
                        1,
                        1,
                        self.d_model,
                        patch_bin_indices.shape[0],
                    ).permute(0, 1, 3, 2)
                embedded = self.patch_embed(patches) + position
            embedded_groups.append(embedded)
            if group_mask is not None:
                masks.append(group_mask[:, 0, :, 0])

        encoded = torch.cat(embedded_groups, dim=2)
        input_mask = torch.cat(masks, dim=1) if masks else None
        return encoded, input_mask

    def _project_patch_weights_to_bins(
        self,
        patch_weights: torch.Tensor,
        patch_coverage_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coverage = (
            self.patch_coverage_matrix
            if patch_coverage_matrix is None
            else patch_coverage_matrix
        ).to(
            dtype=patch_weights.dtype
        )
        if patch_weights.ndim == 3:
            weighted_sum = torch.einsum(
                "pf,bpk->bfk", coverage, patch_weights
            )
            count_shape = (1, -1, 1)
        elif patch_weights.ndim == 4:
            weighted_sum = torch.einsum(
                "pf,bvpk->bvfk", coverage, patch_weights
            )
            count_shape = (1, 1, -1, 1)
        else:
            raise ValueError(
                "patch_weights must have shape [B, P, K] or [B, V, P, K]"
            )
        counts = coverage.sum(dim=0).clamp_min(1.0)
        return weighted_sum / counts.reshape(count_shape)

    def _multi_resolution_stft_residual(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_len, num_vars = target.shape
        residual_flat = (target - reconstruction).permute(0, 2, 1).reshape(
            batch_size * num_vars, time_len
        )
        localized_scores: List[torch.Tensor] = []
        for index, configured_window_size in enumerate(self.stft_window_sizes):
            window_size = min(configured_window_size, time_len)
            hop_length = max(
                1, int(round(window_size * self.stft_hop_ratio))
            )
            if window_size == configured_window_size:
                window = getattr(self, f"_stft_window_{index}").to(
                    dtype=target.dtype,
                    device=target.device,
                )
            else:
                window = torch.hann_window(
                    window_size,
                    dtype=target.dtype,
                    device=target.device,
                )
            residual_stft = torch.stft(
                residual_flat,
                n_fft=window_size,
                hop_length=hop_length,
                win_length=window_size,
                window=window,
                center=True,
                pad_mode="reflect",
                normalized=True,
                onesided=True,
                return_complex=True,
            )
            frame_error = (
                residual_stft.abs()
                .square()
                .reshape(
                    batch_size,
                    num_vars,
                    residual_stft.shape[-2],
                    residual_stft.shape[-1],
                )
                .mean(dim=(1, 2))
            )
            localized_scores.append(
                F.interpolate(
                    frame_error.unsqueeze(1),
                    size=time_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            )
        all_scale_scores = torch.stack(localized_scores, dim=0)
        if self.frequency_score_aggregation == "mean":
            return all_scale_scores.mean(dim=0)
        return all_scale_scores.max(dim=0).values

    def forward(
        self,
        x: torch.Tensor,
        *,
        apply_mask: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        compute_scores: Optional[bool] = None,
        allow_shorter_time_length: bool = False,
    ) -> dict:
        """Run one forward pass.

        Masking defaults to the module's training state. Evaluation and
        detection are unmasked unless ``apply_mask=True`` is requested.
        """
        self._validate_input(
            x,
            allow_shorter_time_length=allow_shorter_time_length,
        )
        resolved_apply_mask = self.training if apply_mask is None else apply_mask
        resolved_compute_scores = (
            not self.training if compute_scores is None else compute_scores
        )
        if not isinstance(resolved_apply_mask, bool):
            raise TypeError("apply_mask must be boolean")
        if not isinstance(resolved_compute_scores, bool):
            raise TypeError("compute_scores must be boolean")
        if not resolved_apply_mask and (generator is not None or seed is not None):
            raise ValueError(
                "A mask seed or generator requires apply_mask=True"
            )
        mask_generator = self._resolve_generator(x, generator, seed)

        batch_size, time_len, num_vars = x.shape
        (
            runtime_patch_bin_indices,
            runtime_patch_coverage,
            runtime_patch_groups,
        ) = self._runtime_patch_layout(
            time_len,
            device=x.device,
        )
        runtime_freq_bins = time_len // 2 + 1
        runtime_num_patches = runtime_patch_bin_indices.shape[0]
        target_norm = self.revin(x, mode="norm") if self.use_revin else x
        target_spectrum = torch.fft.rfft(
            target_norm.permute(0, 2, 1),
            dim=-1,
            norm="ortho",
        )
        patch_tokens, input_mask = self._embed_patches(
            target_spectrum,
            resolved_apply_mask,
            mask_generator,
            runtime_patch_bin_indices,
            runtime_patch_groups,
        )

        # Transformer sequence length is P for every (batch, variable) pair.
        frequency_tokens = patch_tokens.reshape(
            batch_size * num_vars,
            runtime_num_patches,
            self.d_model,
        )
        encoded_frequency = self.patch_transformer(frequency_tokens)
        restored = encoded_frequency.reshape(
            batch_size,
            num_vars,
            runtime_num_patches,
            self.d_model,
        )

        # Cross-variable attention sequence length is N for every patch.
        cross_variable_input = restored.permute(0, 2, 1, 3).reshape(
            batch_size * runtime_num_patches,
            num_vars,
            self.d_model,
        )
        shared = self.cross_var(cross_variable_input).reshape(
            batch_size,
            runtime_num_patches,
            num_vars,
            self.d_model,
        ).permute(0, 2, 1, 3)

        branch_spectrum_list: List[torch.Tensor] = []
        branch_latents: List[torch.Tensor] = []
        for branch in self.branches:
            if (
                runtime_num_patches == self.full_num_patches
                and runtime_freq_bins == self.full_freq_bins
            ):
                branch_spectrum, branch_latent = branch(shared)
            else:
                branch_spectrum, branch_latent = branch(
                    shared,
                    output_freq_bins=runtime_freq_bins,
                )
            branch_spectrum_list.append(branch_spectrum)
            branch_latents.append(branch_latent)
        branch_spectra = torch.stack(branch_spectrum_list, dim=1)

        routing_weights, routing_logits = self.router(shared)
        routing_bin_weights = self._project_patch_weights_to_bins(
            routing_weights,
            runtime_patch_coverage,
        )
        routing_bin_weights = self._apply_expert_specialization(
            routing_bin_weights
        )
        if self.router_mode == "global":
            branch_routing_weights = (
                routing_bin_weights.permute(0, 2, 1).unsqueeze(2)
            )
        else:
            branch_routing_weights = routing_bin_weights.permute(0, 3, 1, 2)
        fused_spectrum = (branch_spectra * branch_routing_weights).sum(dim=1)
        fused_time = torch.fft.irfft(
            fused_spectrum,
            n=time_len,
            dim=-1,
            norm="ortho",
        ).permute(0, 2, 1)

        branch_times = [
            torch.fft.irfft(
                branch_spectra[:, branch_index],
                n=time_len,
                dim=-1,
                norm="ortho",
            ).permute(0, 2, 1)
            for branch_index in range(self.num_branches)
        ]

        reconstruction = (
            self.revin(fused_time, mode="denorm")
            if self.use_revin
            else fused_time
        )
        output = {
            "recon": reconstruction,
            "recon_norm": fused_time,
            "target_norm": target_norm,
            "target_spectrum": target_spectrum,
            "routing_weights": routing_weights,
            "routing_logits": routing_logits,
            "routing_bin_weights": routing_bin_weights,
            "input_mask": input_mask,
            "branch_spectra": branch_spectra,
            "fused_spectrum": fused_spectrum,
            "branch_times": branch_times,
            "branch_latents": branch_latents,
            "xf_real": target_spectrum.real,
            "xf_imag": target_spectrum.imag,
            "xf_rec_real": branch_spectra.permute(1, 0, 2, 3).real,
            "xf_rec_imag": branch_spectra.permute(1, 0, 2, 3).imag,
        }
        _assert_finite("routing_weights", routing_weights)
        _assert_finite("fused_spectrum", fused_spectrum)
        _assert_finite("reconstruction", reconstruction)

        if resolved_compute_scores:
            branch_errors = torch.stack(
                [
                    (target_norm - branch_time).square().mean(dim=-1)
                    for branch_time in branch_times
                ],
                dim=-1,
            )
            routing_sample_weights = self._routing_sample_weights(
                routing_weights,
                routing_bin_weights,
            )
            reconstruction_error = (target_norm - fused_time).square()
            time_score = reconstruction_error.mean(dim=-1)
            freq_score = self._multi_resolution_stft_residual(
                target_norm, fused_time
            )
            disagreement = branch_errors.var(dim=-1, unbiased=False)

            target_gradient = torch.diff(
                target_norm, dim=1, prepend=target_norm[:, :1]
            )
            gradient_errors = torch.stack(
                [
                    (
                        target_gradient
                        - torch.diff(
                            branch_time,
                            dim=1,
                            prepend=branch_time[:, :1],
                        )
                    )
                    .square()
                    .mean(dim=-1)
                    for branch_time in branch_times
                ],
                dim=-1,
            )
            gradient_score = (
                gradient_errors * routing_sample_weights[:, None, :]
            ).sum(dim=-1)
            anomaly_score = (
                time_score
                + self.score_freq_weight * freq_score
                + self.disagreement_weight * disagreement
            )
            if self.use_grad_score:
                anomaly_score = (
                    anomaly_score + self.grad_weight * gradient_score
                )
            output.update(
                {
                    "routing_sample_weights": routing_sample_weights,
                    "branch_errors": branch_errors,
                    "time_score": time_score,
                    "recon_err_t": time_score,
                    "recon_err_per_var": reconstruction_error,
                    "freq_score": freq_score,
                    "gradient_score": gradient_score,
                    "gradient_errors": gradient_errors,
                    "disagreement": disagreement,
                    "anom_score": anomaly_score,
                }
            )
            _assert_finite("anomaly_score", anomaly_score)
        return output

    def forward_masked(
        self,
        x: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        compute_scores: bool = True,
        allow_shorter_time_length: bool = False,
    ) -> dict:
        """Run an explicitly masked pass for stochastic imputation scoring."""
        return self.forward(
            x,
            apply_mask=True,
            generator=generator,
            seed=seed,
            compute_scores=compute_scores,
            allow_shorter_time_length=allow_shorter_time_length,
        )

    def _diversity_loss(
        self, branch_latents: Sequence[torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        if self.diversity_mode == "none" or len(branch_latents) < 2:
            return reference.new_zeros(())
        batch_size = branch_latents[0].shape[0]
        flattened = [
            latent.reshape(batch_size, -1) for latent in branch_latents
        ]
        max_width = max(latent.shape[-1] for latent in flattened)
        padded = [
            F.pad(latent, (0, max_width - latent.shape[-1]))
            for latent in flattened
        ]
        normalized = F.normalize(torch.stack(padded, dim=1), dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        pair_mask = torch.triu(
            torch.ones(
                self.num_branches,
                self.num_branches,
                dtype=torch.bool,
                device=reference.device,
            ),
            diagonal=1,
        )
        pair_similarity = similarity[:, pair_mask]
        if self.diversity_mode == "cosine":
            return pair_similarity.abs().mean()
        temperature = self.contrast_temp.to(dtype=reference.dtype).clamp_min(
            1e-4
        )
        return torch.logsumexp(
            pair_similarity.abs() / temperature, dim=-1
        ).mean() * temperature

    def compute_loss(
        self,
        x: torch.Tensor,
        forward_output: Mapping[str, Any],
        *,
        auxiliary_loss_scale: Real = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute assignment-aware time and frequency reconstruction losses."""
        if (
            isinstance(auxiliary_loss_scale, bool)
            or not isinstance(auxiliary_loss_scale, Real)
            or not torch.isfinite(
                torch.tensor(float(auxiliary_loss_scale))
            )
            or not 0.0 <= float(auxiliary_loss_scale) <= 1.0
        ):
            raise ValueError(
                "auxiliary_loss_scale must be a finite number in [0, 1]"
            )
        resolved_auxiliary_scale = float(auxiliary_loss_scale)
        self._validate_input(
            x,
            check_finite=False,
            allow_shorter_time_length=x.shape[1] < self.time_len,
        )
        if not isinstance(forward_output, Mapping):
            raise TypeError("forward_output must be a mapping")

        target_norm = forward_output.get("target_norm")
        if not isinstance(target_norm, torch.Tensor):
            target_norm = self.revin(x, mode="norm") if self.use_revin else x
        branch_times = forward_output.get("branch_times")
        if (
            not isinstance(branch_times, (list, tuple))
            or len(branch_times) != self.num_branches
        ):
            raise ValueError(
                "forward_output branch_times must contain one tensor per branch"
            )
        time_losses_per_sample = torch.stack(
            [
                (branch_time - target_norm).square().mean(dim=(1, 2))
                for branch_time in branch_times
            ],
            dim=-1,
        )

        routing_weights = forward_output.get("routing_weights")
        if not isinstance(routing_weights, torch.Tensor):
            raise ValueError("forward_output is missing routing_weights")
        runtime_freq_bins = x.shape[1] // 2 + 1
        runtime_patch_rows, _runtime_patch_groups = self._build_patch_layout(
            runtime_freq_bins
        )
        expected_patch_routing_shape = (
            (
                x.shape[0],
                x.shape[2],
                len(runtime_patch_rows),
                self.num_branches,
            )
            if self.router_mode == "variable"
            else (
                x.shape[0],
                len(runtime_patch_rows),
                self.num_branches,
            )
        )
        if tuple(routing_weights.shape) != expected_patch_routing_shape:
            raise ValueError(
                "routing_weights must have shape "
                f"{expected_patch_routing_shape}, got "
                f"{tuple(routing_weights.shape)}"
            )
        routing_bin_weights = forward_output.get("routing_bin_weights")
        if not isinstance(routing_bin_weights, torch.Tensor):
            routing_bin_weights = self._project_patch_weights_to_bins(
                routing_weights
            )
        expected_routing_shape = (
            (
                x.shape[0],
                x.shape[2],
                runtime_freq_bins,
                self.num_branches,
            )
            if self.router_mode == "variable"
            else (
                x.shape[0],
                runtime_freq_bins,
                self.num_branches,
            )
        )
        if tuple(routing_bin_weights.shape) != expected_routing_shape:
            raise ValueError(
                "routing_bin_weights must have shape "
                f"{expected_routing_shape}, got "
                f"{tuple(routing_bin_weights.shape)}"
            )
        routing_sample_weights = self._routing_sample_weights(
            routing_weights,
            routing_bin_weights,
        )
        time_loss = (
            time_losses_per_sample * routing_sample_weights
        ).sum(dim=-1).mean()

        target_spectrum = forward_output.get("target_spectrum")
        if not isinstance(target_spectrum, torch.Tensor):
            target_spectrum = torch.fft.rfft(
                target_norm.permute(0, 2, 1),
                dim=-1,
                norm="ortho",
            )
        branch_spectra = forward_output.get("branch_spectra")
        if not isinstance(branch_spectra, torch.Tensor):
            raise ValueError("forward_output is missing branch_spectra")
        expected_spectrum_shape = (
            x.shape[0],
            self.num_branches,
            self.num_vars,
            runtime_freq_bins,
        )
        if tuple(branch_spectra.shape) != expected_spectrum_shape:
            raise ValueError(
                f"branch_spectra must have shape {expected_spectrum_shape}, "
                f"got {tuple(branch_spectra.shape)}"
            )
        frequency_losses_per_bin = (
            (branch_spectra - target_spectrum[:, None, :, :])
            .abs()
            .square()
            .permute(0, 2, 3, 1)
        )
        frequency_routing_weights = (
            routing_bin_weights
            if self.router_mode == "variable"
            else routing_bin_weights.unsqueeze(1)
        )
        frequency_loss = (
            frequency_losses_per_bin * frequency_routing_weights
        ).sum(dim=-1).mean()

        diversity_loss = self._diversity_loss(
            forward_output.get("branch_latents", []), target_norm
        )
        routing_mean = routing_sample_weights.mean(dim=0)
        load_balance_loss = (
            (routing_mean * routing_mean.clamp_min(1e-8).log()).sum()
            + routing_mean.new_tensor(float(self.num_branches)).log()
        )
        total = (
            time_loss
            + resolved_auxiliary_scale
            * (
                self.lambda_freq * frequency_loss
                + self.diversity_weight * diversity_loss
                + self.load_balance_weight * load_balance_loss
            )
        )
        _assert_finite("loss", total)

        branch_mean_losses = time_losses_per_sample.mean(dim=0)
        losses = {
            "total": float(total.detach()),
            "recon": float(time_loss.detach()),
            "freq": float(frequency_loss.detach()),
            "diversity": float(diversity_loss.detach()),
            "load_balance": float(load_balance_loss.detach()),
            "auxiliary_scale": resolved_auxiliary_scale,
            **{
                f"branch_{index}": float(loss.detach())
                for index, loss in enumerate(branch_mean_losses)
            },
        }
        return total, losses

    @torch.no_grad()
    def detect(
        self, x: torch.Tensor, threshold: Optional[float] = None
    ) -> dict:
        """Deterministic unmasked inference."""
        self.eval()
        output = self.forward(
            x, apply_mask=False, compute_scores=True
        )
        result = {"anom_score": output["anom_score"]}
        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, Real)
                or not torch.isfinite(torch.tensor(float(threshold)))
            ):
                raise ValueError("threshold must be a finite number")
            result["anom_label"] = (
                output["anom_score"] > float(threshold)
            ).to(dtype=output["anom_score"].dtype)
        _assert_finite("detection_output", result)
        return result

    def get_config(self) -> dict:
        """Return the complete constructor configuration for strict restore."""
        return {
            "time_len": self.time_len,
            "num_vars": self.num_vars,
            "patch_len": self.patch_len,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "enc_hidden": self.enc_hidden,
            "bottleneck_dims": list(self.bottleneck_dims),
            "cross_var_heads": self.cross_var_heads,
            "use_revin": self.use_revin,
            "input_mask_ratio": self.input_mask_ratio,
            "mask_mode": self.mask_mode,
            "lambda_freq": self.lambda_freq,
            "router_hidden": self.router_hidden,
            "router_mode": self.router_mode,
            "expert_specialization": self.expert_specialization,
            "diversity_weight": self.diversity_weight,
            "load_balance_weight": self.load_balance_weight,
            "grad_weight": self.grad_weight,
            "score_freq_weight": self.score_freq_weight,
            "disagreement_weight": self.disagreement_weight,
            "diversity_mode": self.diversity_mode,
            "use_contrastive_div": self.use_contrastive_div,
            "use_grad_score": self.use_grad_score,
            "use_adaptive_patches": self.use_adaptive_patches,
            "use_learnable_temp": self.use_learnable_temp,
            "stft_window_sizes": list(self.stft_window_sizes),
            "stft_hop_ratio": self.stft_hop_ratio,
            "frequency_score_aggregation": self.frequency_score_aggregation,
            "architecture": "m2b_moe_v3",
        }
