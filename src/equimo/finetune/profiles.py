"""Declared fine-tuning method profiles."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from .config import MethodProfile, TargetSpec
from .peft.adapters import (
    AdaptFormerConfig,
    ConvPassConfig,
    OrthogonalAdapterConfig,
    RepAdapterConfig,
)
from .peft.lora import (
    EVAInitializerConfig,
    FourierFTConfig,
    LoRAFAConfig,
    LoftQConfig,
    RandLoRAConfig,
)


def adaptformer_chen2022_profile(
    config: AdaptFormerConfig | None = None,
) -> MethodProfile:
    """Return the AdaptFormer paper-profile declaration."""

    config = AdaptFormerConfig.paper_chen2022() if config is None else config
    return _profile(
        profile_id="adaptformer.chen2022.paper",
        method="adaptformer",
        fidelity="paper_exact",
        reference_ids=("chen2022_adaptformer",),
        config=config,
        known_deviations=(),
    )


def lora_fa_zhang2026_profile(config: LoRAFAConfig | None = None) -> MethodProfile:
    """Return the corrected LoRA-FA v3 profile declaration."""

    config = LoRAFAConfig() if config is None else config
    corrected = config.gradient_mode == "corrected_v3" and config.custom_vjp
    return _profile(
        profile_id="lora_fa.zhang2026.corrected_v3",
        method="lora_fa",
        fidelity="reference_implementation" if corrected else "experimental",
        reference_ids=("zhang2026_lora_fa_v3",),
        config=config,
        known_deviations=()
        if corrected
        else ("gradient_mode is a historical ablation, not corrected LoRA-FA v3.",),
    )


def eva_initializer_profile(config: EVAInitializerConfig) -> MethodProfile:
    """Return the EVA initializer profile declaration."""

    return _profile(
        profile_id="eva.initializer.reference",
        method="eva",
        fidelity="reference_implementation",
        reference_ids=("meng2024_eva",),
        config=config,
        target_spec={},
        required_artifacts=("activation_svd",),
        known_deviations=(
            "The initializer consumes immutable activation artifacts supplied by the caller.",
        ),
    )


def loftq_initializer_profile(config: LoftQConfig) -> MethodProfile:
    """Return the LoftQ initializer profile declaration."""

    return _profile(
        profile_id="loftq.initializer.reference",
        method="loftq",
        fidelity="reference_implementation",
        reference_ids=("li2023_loftq",),
        config=config,
        target_spec={},
        required_artifacts=("quantized_base", "quantization_residual"),
    )


def fourierft_gao2024_profile(config: FourierFTConfig) -> MethodProfile:
    """Return the FourierFT profile declaration."""

    deviations = ()
    if config.frequency_selection == "random" and config.seed is None:
        deviations = ("Random frequency selection requires an explicit seed for reproducibility.",)
    return _profile(
        profile_id="fourierft.gao2024.reference",
        method="fourierft",
        fidelity="reference_implementation",
        reference_ids=("gao2024_fourierft",),
        config=config,
        known_deviations=deviations,
    )


def oft_qiu2023_profile(
    config: OrthogonalAdapterConfig | None = None,
) -> MethodProfile:
    """Return the OFT Cayley-transform profile declaration."""

    config = (
        OrthogonalAdapterConfig(parameterization="cayley")
        if config is None
        else config
    )
    return _profile(
        profile_id="oft.qiu2023.cayley",
        method="oft",
        fidelity="reference_implementation",
        reference_ids=("qiu2023_oft",),
        config=config,
    )


def boft_liu2024_profile(config: OrthogonalAdapterConfig) -> MethodProfile:
    """Return the BOFT butterfly-Cayley profile declaration."""

    deviations = ()
    if config.parameterization != "butterfly_cayley":
        deviations = ("Config parameterization is not BOFT butterfly_cayley.",)
    if config.block_size is None:
        deviations = (*deviations, "BOFT requires an explicit block_size.")
    return _profile(
        profile_id="boft.liu2024.butterfly_cayley",
        method="boft",
        fidelity="reference_implementation"
        if not deviations
        else "experimental",
        reference_ids=("liu2024_boft",),
        config=config,
        known_deviations=deviations,
    )


def convpass_profile(config: ConvPassConfig | None = None) -> MethodProfile:
    """Return the ConvPass vision-native adapter profile declaration."""

    config = ConvPassConfig() if config is None else config
    required_artifacts = () if config.patch_grid is not None else ("patch_grid_metadata",)
    return _profile(
        profile_id="convpass.vision.reference",
        method="convpass",
        fidelity="model_family_recipe",
        reference_ids=("convpass",),
        config=config,
        required_artifacts=required_artifacts,
    )


def repadapter_profile(config: RepAdapterConfig | None = None) -> MethodProfile:
    """Return the RepAdapter structural-reparameterization profile."""

    config = RepAdapterConfig() if config is None else config
    return _profile(
        profile_id="repadapter.luo2023.structural",
        method="repadapter",
        fidelity="reference_implementation" if config.mergeable else "experimental",
        reference_ids=("luo2023_repadapter",),
        config=config,
        known_deviations=()
        if config.mergeable
        else ("Non-mergeable RepAdapter variants are not the structural profile.",),
    )


def randlora_profile(config: RandLoRAConfig | None = None) -> MethodProfile:
    """Return the RandLoRA frozen-random-basis profile."""

    config = RandLoRAConfig() if config is None else config
    deviations = ()
    if config.seed is None:
        deviations = ("RandLoRA requires an explicit seed for reproducible bases.",)
    return _profile(
        profile_id="randlora.albert2025.reference",
        method="randlora",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("albert2025_randlora",),
        config=config,
        known_deviations=deviations,
    )


def available_profile_ids() -> tuple[str, ...]:
    """Return the built-in profile identifiers exposed by Equimo."""

    return (
        "adaptformer.chen2022.paper",
        "boft.liu2024.butterfly_cayley",
        "convpass.vision.reference",
        "eva.initializer.reference",
        "fourierft.gao2024.reference",
        "loftq.initializer.reference",
        "lora_fa.zhang2026.corrected_v3",
        "oft.qiu2023.cayley",
        "randlora.albert2025.reference",
        "repadapter.luo2023.structural",
    )


def _profile(
    *,
    profile_id: str,
    method: str,
    fidelity: str,
    reference_ids: tuple[str, ...],
    config: Any,
    target_spec: dict[str, Any] | None = None,
    known_deviations: tuple[str, ...] = (),
    required_artifacts: tuple[str, ...] = (),
) -> MethodProfile:
    config_dict = _config_dict(config)
    return MethodProfile(
        id=profile_id,
        method=method,
        fidelity=fidelity,
        reference_ids=reference_ids,
        config=config_dict,
        target_spec=_target_spec_dict(config.target)
        if target_spec is None and hasattr(config, "target")
        else ({} if target_spec is None else target_spec),
        known_deviations=known_deviations,
        required_artifacts=required_artifacts,
    )


def _config_dict(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        return dict(value) if isinstance(value, dict) else {"value": value}
    result: dict[str, Any] = {}
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, TargetSpec):
            result[field.name] = _target_spec_dict(item)
        elif is_dataclass(item):
            result[field.name] = _config_dict(item)
        else:
            result[field.name] = item
    return result


def _target_spec_dict(target: TargetSpec) -> dict[str, Any]:
    return {
        "tags_all": target.tags_all,
        "tags_any": target.tags_any,
        "include": target.include,
        "exclude": target.exclude,
        "min_depth": target.min_depth,
        "max_depth": target.max_depth,
        "target_kind": target.target_kind,
        "allow_empty": target.allow_empty,
        "predicate": None if target.predicate is None else "<callable>",
    }


__all__ = (
    "adaptformer_chen2022_profile",
    "available_profile_ids",
    "boft_liu2024_profile",
    "convpass_profile",
    "eva_initializer_profile",
    "fourierft_gao2024_profile",
    "loftq_initializer_profile",
    "lora_fa_zhang2026_profile",
    "oft_qiu2023_profile",
    "randlora_profile",
    "repadapter_profile",
)
