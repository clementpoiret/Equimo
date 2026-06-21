"""Declared fine-tuning method profiles."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from .config import MethodProfile, TargetSpec
from .peft.adapters import (
    AdapterConfig,
    AdapterFusionConfig,
    AdaptFormerConfig,
    OrthogonalAdapterConfig,
)
from .peft.dora import DoRAConfig
from .peft.ia3 import IA3Config
from .peft.lora import (
    EVAInitializerConfig,
    FourierFTConfig,
    LoRAConfig,
    LoRAFAConfig,
    PiSSAConfig,
    RandLoRAConfig,
    RsLoRAConfig,
)
from .peft.prefix import PrefixConfig
from .peft.prompts import (
    PTuningV2Config,
    SoftPromptConfig,
    VPTDeepConfig,
    VPTShallowConfig,
)
from .peft.scale_shift import ScaleShiftConfig
from .peft.vera import VeRAConfig


def adaptformer_chen2022_profile(
    config: AdaptFormerConfig | None = None,
) -> MethodProfile:
    """Return the AdaptFormer paper-profile declaration."""

    config = AdaptFormerConfig.paper_chen2022() if config is None else config
    deviations = _adaptformer_paper_deviations(config)
    return _profile(
        profile_id="adaptformer.chen2022.paper",
        method="adaptformer",
        fidelity="paper_exact" if not deviations else "experimental",
        reference_ids=("chen2022_adaptformer", "shoufachen_adaptformer"),
        config=config,
        known_deviations=deviations,
    )


def adapter_houlsby2019_profile(config: AdapterConfig | None = None) -> MethodProfile:
    """Return the Houlsby bottleneck-adapter profile declaration."""

    config = AdapterConfig(placement="both") if config is None else config
    deviations = (
        "Equimo uses Kaiming down-projection and zero up-projection for "
        "identity-safe initialization; google-research adapter-bert uses small "
        "truncated-normal weights for both adapter projections.",
    )
    if config.placement != "both":
        deviations = (
            *deviations,
            "Houlsby adapters are inserted after both attention and feed-forward sublayers.",
        )
    if config.activation != "gelu":
        deviations = (
            *deviations,
            "google-research adapter-bert uses GELU inside feed-forward adapters.",
        )
    if config.pre_norm:
        deviations = (
            *deviations,
            "The google-research adapter-bert feed-forward adapter does not add "
            "an internal adapter LayerNorm.",
        )
    if config.train_base:
        deviations = (
            *deviations,
            "Houlsby adapter tuning freezes the pretrained base.",
        )
    return _profile(
        profile_id="adapter.houlsby2019.bottleneck",
        method="adapter",
        fidelity="safe_default"
        if config.placement == "both" and not config.train_base
        else "experimental",
        reference_ids=(
            "houlsby2019_adapter",
            "google_research_adapter_bert",
            "adapterhub_adapters",
        ),
        config=config,
        known_deviations=deviations,
    )


def adapterfusion_pfeiffer2021_profile(
    config: AdapterFusionConfig | None = None,
) -> MethodProfile:
    """Return the AdapterFusion attention-composition profile declaration."""

    config = AdapterFusionConfig() if config is None else config
    deviations = ()
    if config.fusion != "attention":
        deviations = (*deviations, "AdapterFusion requires attention fusion.")
    if not config.freeze_task_adapters:
        deviations = (
            *deviations,
            "AdapterFusion trains the fusion parameters in a second stage while "
            "keeping task adapters frozen.",
        )
    if "after_mlp" not in config.placement:
        deviations = (
            *deviations,
            "The Pfeiffer/AdapterHub setup composes named bottleneck adapters "
            "at feed-forward/output adapter sites.",
        )
    return _profile(
        profile_id="adapterfusion.pfeiffer2021.attention",
        method="adapter_fusion",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("pfeiffer2021_adapterfusion", "adapterhub_adapters"),
        config=config,
        target_spec={},
        required_artifacts=("trained_adapter_bank",),
        known_deviations=deviations,
    )


def vpt_deep_jia2022_profile(config: VPTDeepConfig | None = None) -> MethodProfile:
    """Return the VPT-Deep profile declaration."""

    config = VPTDeepConfig() if config is None else config
    deviations = _vpt_common_deviations(config)
    if config.depth != "deep":
        deviations = (*deviations, "VPT-Deep requires depth='deep'.")
    return _profile(
        profile_id="vpt.jia2022.deep",
        method="vpt",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("jia2022_vpt", "kmnp_vpt"),
        config=config,
        target_spec={},
        known_deviations=deviations,
    )


def vpt_shallow_jia2022_profile(
    config: VPTShallowConfig | None = None,
) -> MethodProfile:
    """Return the VPT-Shallow profile declaration."""

    config = VPTShallowConfig() if config is None else config
    deviations = _vpt_common_deviations(config)
    if config.depth != "shallow":
        deviations = (*deviations, "VPT-Shallow requires depth='shallow'.")
    return _profile(
        profile_id="vpt.jia2022.shallow",
        method="vpt",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("jia2022_vpt", "kmnp_vpt"),
        config=config,
        target_spec={},
        known_deviations=deviations,
    )


def soft_prompt_lester2021_profile(
    config: SoftPromptConfig | None = None,
) -> MethodProfile:
    """Return the Lester et al. input soft-prompt profile declaration."""

    config = SoftPromptConfig() if config is None else config
    deviations = ()
    if config.depth != "shallow":
        deviations = (
            *deviations,
            "Soft prompt tuning prepends prompts only at model input.",
        )
    if config.prepend_to != "input":
        deviations = (*deviations, "Soft prompt tuning requires prepend_to='input'.")
    if config.init != "from_embedding_if_available":
        deviations = (
            *deviations,
            "Equimo's Lester-profile default initializes from token embeddings when available.",
        )
    return _profile(
        profile_id="soft_prompt.lester2021.input",
        method="soft_prompt",
        fidelity="safe_default" if not deviations else "experimental",
        reference_ids=("lester2021_prompt_tuning", "google_research_prompt_tuning"),
        config=config,
        target_spec={},
        known_deviations=deviations,
    )


def ptuning_v2_liu2022_profile(
    config: PTuningV2Config | None = None,
) -> MethodProfile:
    """Return the P-Tuning v2 deep-prompt profile declaration."""

    config = PTuningV2Config() if config is None else config
    deviations = ()
    if config.depth != "all":
        deviations = (
            *deviations,
            "P-Tuning v2 profile inserts prompts at every layer.",
        )
    if config.share_across_layers:
        deviations = (
            *deviations,
            "The P-Tuning v2 profile uses layer-specific prompt representations.",
        )
    if config.reparameterizer != "none":
        deviations = (
            *deviations,
            "MLP prompt reparameterization is declared but not implemented locally.",
        )
    return _profile(
        profile_id="ptuning_v2.liu2022.deep_prompts",
        method="ptuning_v2",
        fidelity="safe_default" if not deviations else "experimental",
        reference_ids=("liu2022_ptuning_v2", "thudm_ptuning_v2"),
        config=config,
        target_spec={},
        known_deviations=deviations,
    )


def prefix_tuning_li2021_profile(
    config: PrefixConfig | None = None,
) -> MethodProfile:
    """Return the Li and Liang attention-prefix profile declaration."""

    config = PrefixConfig() if config is None else config
    deviations = ()
    if config.depth != "deep":
        deviations = (
            *deviations,
            "Prefix tuning profile uses layer-specific deep prefixes.",
        )
    if frozenset(config.target) != frozenset(("attention.k", "attention.v")):
        deviations = (
            *deviations,
            "Prefix tuning prepends trainable states only to attention K/V tensors.",
        )
    if not config.prefix_projection:
        deviations = (
            *deviations,
            "Li and Liang use a prefix projection/reparameterization during training.",
        )
    if config.direct_kv:
        deviations = (
            *deviations,
            "Direct serialized K/V prefix tensors are declared but not implemented locally.",
        )
    return _profile(
        profile_id="prefix_tuning.li2021.attention_kv",
        method="prefix_tuning",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("li2021_prefix_tuning", "xiangli1999_prefix_tuning"),
        config=config,
        target_spec={},
        known_deviations=deviations,
    )


def ia3_liu2022_profile(config: IA3Config | None = None) -> MethodProfile:
    """Return the Liu et al. IA3 activation-scaling profile declaration."""

    config = IA3Config() if config is None else config
    deviations = _ia3_deviations(config)
    return _profile(
        profile_id="ia3.liu2022.activation_scaling",
        method="ia3",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("liu2022_ia3", "r_three_t_few"),
        config=config,
        known_deviations=deviations,
    )


def ssf_lian2022_vit_profile(
    config: ScaleShiftConfig | None = None,
) -> MethodProfile:
    """Return the Lian et al. SSF ViT profile declaration."""

    config = ScaleShiftConfig.paper_lian2022_vit() if config is None else config
    deviations = _ssf_vit_deviations(config)
    return _profile(
        profile_id="ssf.lian2022.vit",
        method="scale_shift",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("lian2022_ssf", "dongzelian_ssf"),
        config=config,
        known_deviations=deviations,
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


def lora_equimo_default_profile(config: LoRAConfig | None = None) -> MethodProfile:
    """Return Equimo's safe-default LoRA profile declaration."""

    config = LoRAConfig() if config is None else config
    deviations = (
        "Equimo default targets fused attention.qkv and attention.proj, not the "
        "paper's common Q/V-only target set.",
    )
    return _profile(
        profile_id="lora.equimo_default",
        method="lora",
        fidelity="safe_default",
        reference_ids=("hu2021_lora",),
        config=config,
        known_deviations=deviations,
    )


def lora_hu2021_qv_profile(config: LoRAConfig | None = None) -> MethodProfile:
    """Return the Hu et al. Q/V projection-segment LoRA profile declaration."""

    if config is None:
        config = LoRAConfig(
            target=TargetSpec(
                tags_any=("attention.q", "attention.v"),
                target_kind="projection_segment",
            )
        )
    target = config.target
    selected = {tag.rsplit(".", maxsplit=1)[-1] for tag in target.tags_any}
    deviations: tuple[str, ...] = ()
    if target.target_kind != "projection_segment" or selected != {"q", "v"}:
        deviations = (
            "Hu et al. Q/V profile requires projection_segment target tags "
            "attention.q and attention.v.",
        )
    return _profile(
        profile_id="lora.hu2021.qv",
        method="lora",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("hu2021_lora",),
        config=config,
        known_deviations=deviations,
    )


def rslora_kalajdzievski2023_profile(
    config: RsLoRAConfig | None = None,
) -> MethodProfile:
    """Return the rank-stabilized LoRA profile declaration."""

    config = RsLoRAConfig() if config is None else config
    deviations = (
        ()
        if config.scaling == "alpha_over_sqrt_r"
        else ("rsLoRA requires alpha_over_sqrt_r scaling.",)
    )
    return _profile(
        profile_id="rslora.kalajdzievski2023",
        method="rslora",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("kalajdzievski2023_rslora", "hf_peft_use_rslora"),
        config=config,
        known_deviations=deviations,
    )


def pissa_meng2024_profile(config: PiSSAConfig | None = None) -> MethodProfile:
    """Return the PiSSA exact-SVD initialization profile declaration."""

    config = PiSSAConfig(svd="full", niter=0) if config is None else config
    deviations: tuple[str, ...] = ()
    if config.init != "pissa":
        deviations = (*deviations, "PiSSA profile requires init='pissa'.")
    if config.residual_handling != "freeze_residual":
        deviations = (
            *deviations,
            "PiSSA requires freeze_residual so W_residual + scaling * B0 @ A0 "
            "preserves the initial function.",
        )
    if config.svd not in {"full", "exact"}:
        deviations = (
            *deviations,
            "The reference profile uses dense exact SVD; randomized/iterative "
            "PiSSA is not implemented.",
        )
    if config.niter != 0:
        deviations = (
            *deviations,
            "niter is reserved for future iterative PiSSA profiles and is not "
            "used by Equimo's dense exact SVD initializer.",
        )
    return _profile(
        profile_id="pissa.meng2024.exact_svd",
        method="pissa",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("meng2024_pissa", "graphpku_pissa"),
        config=config,
        required_artifacts=("principal_svd_or_product",),
        known_deviations=deviations,
    )


def vera_kopiczko2024_profile(config: VeRAConfig | None = None) -> MethodProfile:
    """Return Equimo's shape-compatible VeRA profile declaration."""

    config = VeRAConfig() if config is None else config
    deviations = (
        "Equimo shares frozen random bases across shape-compatible modules; "
        "the paper/HF PEFT profile uses one shared basis pool across target layers.",
    )
    if not config.shared:
        deviations = (
            *deviations,
            "shared=False disables VeRA's shared random-basis contract.",
        )
    if config.trainable_output_scale_init != 0.0:
        deviations = (
            *deviations,
            "Identity-safe VeRA initialization requires output scale b=0.",
        )
    return _profile(
        profile_id="vera.kopiczko2024.shape_compatible",
        method="vera",
        fidelity="safe_default" if config.shared else "experimental",
        reference_ids=("kopiczko2024_vera", "hf_peft_vera"),
        config=config,
        known_deviations=deviations,
    )


def dora_liu2024_profile(config: DoRAConfig | None = None) -> MethodProfile:
    """Return the differentiable DoRA paper-equation profile declaration."""

    config = DoRAConfig() if config is None else config
    deviations = _dora_common_deviations(config)
    if config.norm_gradient != "full":
        deviations = (
            *deviations,
            "The differentiable DoRA paper-equation profile requires "
            "norm_gradient='full'.",
        )
    return _profile(
        profile_id="dora.liu2024.paper_equation",
        method="dora",
        fidelity="paper_exact" if not deviations else "experimental",
        reference_ids=("liu2024_dora",),
        config=config,
        known_deviations=deviations,
    )


def dora_nvlabs_reference_profile(config: DoRAConfig | None = None) -> MethodProfile:
    """Return the NVLabs/HF PEFT DoRA stop-gradient reference profile."""

    config = DoRAConfig(norm_gradient="detached") if config is None else config
    deviations = _dora_common_deviations(config)
    if config.norm_gradient != "detached":
        deviations = (
            *deviations,
            "The NVLabs/HF PEFT reference treats the direction norm as "
            "a stop-gradient constant.",
        )
    if config.norm_impl != "dense":
        deviations = (
            *deviations,
            "The NVLabs/HF PEFT reference uses dense norm materialization; "
            "use the factored 2026 profile for factored norms.",
        )
    return _profile(
        profile_id="dora.nvlabs_reference",
        method="dora",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("liu2024_dora", "nvlabs_dora", "hf_peft_dora"),
        config=config,
        known_deviations=deviations,
    )


def dora_factored_2026_profile(config: DoRAConfig | None = None) -> MethodProfile:
    """Return the Scaling DoRA factored-norm profile declaration."""

    config = (
        DoRAConfig(norm_gradient="detached", norm_impl="factored")
        if config is None
        else config
    )
    deviations = _dora_common_deviations(config)
    if config.norm_impl != "factored":
        deviations = (
            *deviations,
            "Scaling DoRA requires norm_impl='factored'.",
        )
    if config.norm_gradient != "detached":
        deviations = (
            *deviations,
            "Scaling DoRA follows the detached norm denominator used by "
            "the public DoRA reference path.",
        )
    deviations = (
        *deviations,
        "Equimo implements the factored row-norm algebra but not the "
        "paper's chunked fp32/Triton fused runtime.",
    )
    return _profile(
        profile_id="dora.factored_2026.eager",
        method="dora",
        fidelity="experimental",
        reference_ids=("zelenin2026_scaling_dora", "sockeye44_dorafactors"),
        config=config,
        known_deviations=deviations,
    )


def eva_initializer_profile(config: EVAInitializerConfig) -> MethodProfile:
    """Return the EVA initializer profile declaration."""

    return _profile(
        profile_id="eva.initializer.reference",
        method="eva",
        fidelity="reference_implementation",
        reference_ids=("paischer2024_eva", "ml_jku_eva"),
        config=config,
        target_spec={},
        required_artifacts=("activation_svd",),
        known_deviations=(
            "The initializer consumes immutable activation artifacts supplied by the caller.",
        ),
    )


def fourierft_gao2024_profile(config: FourierFTConfig) -> MethodProfile:
    """Return the FourierFT profile declaration."""

    deviations = ()
    if config.frequency_selection == "random" and config.seed is None:
        deviations = (
            "Random frequency selection requires an explicit seed for reproducibility.",
        )
    return _profile(
        profile_id="fourierft.gao2024.reference",
        method="fourierft",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("gao2024_fourierft", "chaos96_fourierft", "hf_peft_fourierft"),
        config=config,
        known_deviations=deviations,
    )


def oft_qiu2023_profile(
    config: OrthogonalAdapterConfig | None = None,
) -> MethodProfile:
    """Return the OFT Cayley-transform profile declaration."""

    config = (
        OrthogonalAdapterConfig(parameterization="cayley") if config is None else config
    )
    deviations = ()
    if config.parameterization != "cayley":
        deviations = (
            "OFT requires parameterization='cayley'; use the BOFT profile "
            "for butterfly_cayley.",
        )
    if config.train_base:
        deviations = (*deviations, "OFT freezes the pretrained base weights.")
    return _profile(
        profile_id="oft.qiu2023.cayley",
        method="oft",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("qiu2023_oft", "zqiu24_oft", "hf_peft_oft"),
        config=config,
        known_deviations=deviations,
    )


def boft_liu2024_profile(config: OrthogonalAdapterConfig) -> MethodProfile:
    """Return the BOFT butterfly-Cayley profile declaration."""

    deviations = ()
    if config.parameterization != "butterfly_cayley":
        deviations = ("Config parameterization is not BOFT butterfly_cayley.",)
    if config.block_size is None:
        deviations = (*deviations, "BOFT requires an explicit block_size.")
    if config.train_base:
        deviations = (*deviations, "BOFT freezes the pretrained base weights.")
    return _profile(
        profile_id="boft.liu2024.butterfly_cayley",
        method="boft",
        fidelity="reference_implementation" if not deviations else "experimental",
        reference_ids=("liu2024_boft", "wy1iu_butterfly_oft", "hf_peft_boft"),
        config=config,
        known_deviations=deviations,
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
        reference_ids=("albert2025_randlora", "paulalbert31_randlora"),
        config=config,
        known_deviations=deviations,
    )


def available_profile_ids() -> tuple[str, ...]:
    """Return the built-in profile identifiers exposed by Equimo."""

    return (
        "adapter.houlsby2019.bottleneck",
        "adapterfusion.pfeiffer2021.attention",
        "adaptformer.chen2022.paper",
        "boft.liu2024.butterfly_cayley",
        "dora.factored_2026.eager",
        "dora.liu2024.paper_equation",
        "dora.nvlabs_reference",
        "eva.initializer.reference",
        "fourierft.gao2024.reference",
        "ia3.liu2022.activation_scaling",
        "lora.equimo_default",
        "lora.hu2021.qv",
        "lora_fa.zhang2026.corrected_v3",
        "oft.qiu2023.cayley",
        "pissa.meng2024.exact_svd",
        "prefix_tuning.li2021.attention_kv",
        "ptuning_v2.liu2022.deep_prompts",
        "randlora.albert2025.reference",
        "rslora.kalajdzievski2023",
        "soft_prompt.lester2021.input",
        "ssf.lian2022.vit",
        "vera.kopiczko2024.shape_compatible",
        "vpt.jia2022.deep",
        "vpt.jia2022.shallow",
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


def _dora_common_deviations(config: DoRAConfig) -> tuple[str, ...]:
    deviations: tuple[str, ...] = ()
    if config.scaling != "alpha_over_r":
        deviations = (
            *deviations,
            "DoRA reference profiles use ordinary LoRA alpha_over_r scaling.",
        )
    if config.train_base:
        deviations = (
            *deviations,
            "DoRA freezes the pretrained base weights and trains magnitude plus LoRA factors.",
        )
    return deviations


def _adaptformer_paper_deviations(config: AdaptFormerConfig) -> tuple[str, ...]:
    deviations: tuple[str, ...] = ()
    if config.bottleneck != 64:
        deviations = (
            *deviations,
            "Chen et al. use bottleneck=64 in the paper profile.",
        )
    if config.placement != "parallel_mlp":
        deviations = (
            *deviations,
            "AdaptFormer paper profile requires placement='parallel_mlp'.",
        )
    if config.activation != "relu":
        deviations = (
            *deviations,
            "AdaptFormer paper/profile implementation uses ReLU in the adapter branch.",
        )
    if config.up_init != "zeros":
        deviations = (
            *deviations,
            "AdaptFormer reference initializes the up-projection to zero.",
        )
    if config.scale_init != 0.1:
        deviations = (
            *deviations,
            "AdaptFormer paper profile uses fixed scale_init=0.1.",
        )
    if config.scale_trainable:
        deviations = (
            *deviations,
            "The paper profile uses a fixed scalar multiplier, not a trainable scale.",
        )
    return deviations


def _vpt_common_deviations(config: VPTDeepConfig | VPTShallowConfig) -> tuple[str, ...]:
    deviations: tuple[str, ...] = ()
    if config.init != "normal":
        deviations = (*deviations, "VPT uses normal prompt initialization.")
    if config.init_std != 0.02:
        deviations = (*deviations, "VPT profile uses init_std=0.02.")
    if config.prepend_to != "after_cls":
        deviations = (*deviations, "VPT inserts prompt tokens after the class token.")
    if not config.exclude_prompt_tokens_from_pool:
        deviations = (*deviations, "VPT pooling excludes prompt tokens by default.")
    return deviations


def _ia3_deviations(config: IA3Config) -> tuple[str, ...]:
    deviations: tuple[str, ...] = ()
    required = {"attention.k", "attention.v", "mlp.hidden"}
    tags = set(config.target.tags_any) | set(config.target.tags_all)
    if not required <= tags:
        deviations = (
            *deviations,
            "IA3 scales attention key/value outputs and the MLP intermediate activation.",
        )
    if config.init != 1.0:
        deviations = (*deviations, "IA3 activation scales initialize to one.")
    if config.axis != "feature":
        deviations = (*deviations, "IA3 scales the feature/channel axis.")
    if not config.mergeable:
        deviations = (
            *deviations,
            "IA3 reference insertion points are mergeable when adjacent weights are available.",
        )
    return deviations


def _ssf_vit_deviations(config: ScaleShiftConfig) -> tuple[str, ...]:
    deviations: tuple[str, ...] = ()
    required = {
        "embedding.patch",
        "attention.qkv",
        "attention.proj",
        "mlp.hidden",
        "mlp.fc2",
        "norm",
    }
    tags = set(config.target.tags_any) | set(config.target.tags_all)
    if tags != required:
        deviations = (
            *deviations,
            "SSF ViT profile uses exact patch, norm, attention, and MLP insertion sites; "
            "broad tags such as 'attention' or 'mlp' are insufficient.",
        )
    if config.init != "normal" or config.scale_init != 1.0 or config.shift_init != 0.0:
        deviations = (
            *deviations,
            "SSF paper profile initializes scale around mean 1 and shift around mean 0.",
        )
    if config.init_std != 0.02:
        deviations = (
            *deviations,
            "SSF reference code uses normal initialization std=0.02.",
        )
    if config.axis != "feature":
        deviations = (
            *deviations,
            "The ViT SSF profile applies scale/shift on the feature axis.",
        )
    if not config.mergeable:
        deviations = (
            *deviations,
            "SSF paper profile is designed for inference-time reparameterization.",
        )
    return deviations


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
    "adapter_houlsby2019_profile",
    "adapterfusion_pfeiffer2021_profile",
    "adaptformer_chen2022_profile",
    "available_profile_ids",
    "boft_liu2024_profile",
    "dora_factored_2026_profile",
    "dora_liu2024_profile",
    "dora_nvlabs_reference_profile",
    "eva_initializer_profile",
    "fourierft_gao2024_profile",
    "ia3_liu2022_profile",
    "lora_equimo_default_profile",
    "lora_fa_zhang2026_profile",
    "lora_hu2021_qv_profile",
    "oft_qiu2023_profile",
    "pissa_meng2024_profile",
    "prefix_tuning_li2021_profile",
    "ptuning_v2_liu2022_profile",
    "randlora_profile",
    "rslora_kalajdzievski2023_profile",
    "soft_prompt_lester2021_profile",
    "ssf_lian2022_vit_profile",
    "vera_kopiczko2024_profile",
    "vpt_deep_jia2022_profile",
    "vpt_shallow_jia2022_profile",
)
