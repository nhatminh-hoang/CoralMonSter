"""Utility loaders for released CoralScapes baseline checkpoints."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import math

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from segmentation_models_pytorch import DeepLabV3Plus, UnetPlusPlus
from torch import nn
from transformers import (
    Dinov2Config,
    Dinov2Model,
    Dinov2PreTrainedModel,
    DPTConfig,
    DPTForSemanticSegmentation,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)
from transformers.modeling_outputs import SemanticSegmenterOutput


def _id_mappings(num_classes: int) -> Dict[int, str]:
    return {i: str(i) for i in range(num_classes)}


def _load_segformer(
    backbone: str,
    checkpoint_path: str,
    num_classes: int,
    use_lora: bool = False,
    lora_rank: int = 128,
    lora_alpha: int = 32,
    modules_to_save: Optional[list[str]] = None,
) -> nn.Module:
    config = SegformerConfig.from_pretrained(
        backbone,
        num_labels=num_classes,
        semantic_loss_ignore_index=0,
        id2label=_id_mappings(num_classes),
    )
    model = SegformerForSemanticSegmentation(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=modules_to_save or ["decode_head"],
        )
        model = get_peft_model(model, lora_cfg)
    model.load_state_dict(state_dict, strict=False)
    return model


def _load_smp(model_name: str, checkpoint_path: str, num_classes: int) -> nn.Module:
    if model_name == "deeplabv3+resnet50":
        model = DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif model_name == "unet++resnet50":
        model = UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported SMP model '{model_name}'")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    return model


class DPTDinov2ForSemanticSegmentation(nn.Module):
    """Wrapper that pairs a DINOv2 backbone with the DPT segmentation head."""

    def __init__(self, num_labels: int, backbone: str = "facebook/dinov2-base") -> None:
        super().__init__()
        self.num_labels = num_labels
        dinov2_config = Dinov2Config.from_pretrained(backbone, reshape_hidden_states=True)
        self.backbone = Dinov2Model.from_pretrained(backbone, reshape_hidden_states=True)
        if backbone.endswith("base"):
            indices = (2, 5, 8, 11)
        elif backbone.endswith("giant"):
            indices = (9, 19, 29, 39)
        else:
            raise ValueError(f"Unsupported DINOv2 backbone '{backbone}'")
        self.indices = indices
        dpt_config = DPTConfig(
            num_labels=num_labels,
            ignore_index=0,
            semantic_loss_ignore_index=0,
            is_hybrid=False,
            backbone_out_indices=indices,
            backbone_config=dinov2_config,
        )
        dpt = DPTForSemanticSegmentation(dpt_config)
        self.neck = dpt.neck
        self.head = dpt.head

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> SemanticSegmenterOutput:
        features = self.backbone(pixel_values, output_hidden_states=True)
        pyramid = [features.hidden_states[i] for i in self.indices]
        h, w = pixel_values.shape[-2:]
        if h != w:
            if h % 14 != 0 or w % 14 != 0:
                raise ValueError("Height and width must be divisible by patch size (14).")
            pyramid = self.neck(pyramid, patch_height=h // 14, patch_width=w // 14)
        else:
            pyramid = self.neck(pyramid)
        logits = self.head(pyramid)
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        loss = None
        if labels is not None:
            upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(upsampled, labels.long(), ignore_index=0)
        return SemanticSegmenterOutput(loss=loss, logits=logits)


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, num_labels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classifier = nn.Linear(in_channels, num_labels)

    def forward(self, embeddings: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        batch, tokens, channels = embeddings.shape
        if tokens != token_h * token_w:
            raise ValueError("Unexpected number of tokens for Dinov2 linear probe")
        logits = self.classifier(embeddings)
        logits = logits.permute(0, 2, 1)
        return logits.reshape(batch, logits.shape[1], token_h, token_w)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    """Linear probe on top of frozen DINOv2 patch tokens."""

    def __init__(self, config: Dinov2Config, token_w: int = 74, token_h: int = 37) -> None:
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, config.num_labels)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None, **_) -> SemanticSegmenterOutput:
        outputs = self.dinov2(pixel_values, output_hidden_states=False, output_attentions=False)
        patches = outputs.last_hidden_state[:, 1:, :]
        tokens = patches.shape[1]
        token_dim = int(math.sqrt(tokens))
        if token_dim * token_dim != tokens:
            raise ValueError(f"Cannot reshape {tokens} tokens into a square feature map for Dinov2 linear probe")
        logits = self.classifier(patches, token_dim, token_dim)
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        loss = None
        if labels is not None:
            upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(upsampled, labels.long(), ignore_index=0)
        return SemanticSegmenterOutput(loss=loss, logits=logits)


def _load_dpt(
    backbone: str,
    checkpoint_path: str,
    num_classes: int,
    use_lora: bool = False,
    lora_rank: int = 128,
    lora_alpha: int = 32,
    modules_to_save: Optional[list[str]] = None,
) -> nn.Module:
    model = DPTDinov2ForSemanticSegmentation(num_labels=num_classes, backbone=backbone)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=modules_to_save or ["head"],
        )
        model = get_peft_model(model, lora_cfg)
    model.load_state_dict(state_dict, strict=False)
    return model


def _load_linear_dino(checkpoint_path: str, num_classes: int, **_: object) -> nn.Module:
    dinov2_config = Dinov2Config.from_pretrained(
        "facebook/dinov2-base",
        num_labels=num_classes,
        image_size=518,
        patch_size=14,
    )
    model = Dinov2ForSemanticSegmentation(dinov2_config)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    weight_key = "classifier.classifier.weight"
    if weight_key in state_dict and state_dict[weight_key].dim() == 4:
        state_dict[weight_key] = state_dict[weight_key].squeeze(-1).squeeze(-1)
    model.load_state_dict(state_dict, strict=False)
    return model


@dataclass
class BaselineSpec:
    loader: Callable[..., nn.Module]
    kwargs: Dict[str, object]


BASELINE_SPECS: Dict[str, BaselineSpec] = {
    "segformer_mit_b2": BaselineSpec(loader=_load_segformer, kwargs={"backbone": "nvidia/mit-b2"}),
    "segformer_mit_b2_lora": BaselineSpec(
        loader=_load_segformer,
        kwargs={"backbone": "nvidia/mit-b2", "use_lora": True, "modules_to_save": ["decode_head"]},
    ),
    "segformer_mit_b5": BaselineSpec(loader=_load_segformer, kwargs={"backbone": "nvidia/mit-b5"}),
    "segformer_mit_b5_lora": BaselineSpec(
        loader=_load_segformer,
        kwargs={"backbone": "nvidia/mit-b5", "use_lora": True, "modules_to_save": ["decode_head"]},
    ),
    "segformer_mit_b5_more_aug": BaselineSpec(loader=_load_segformer, kwargs={"backbone": "nvidia/mit-b5"}),
    "deeplabv3plus_resnet50": BaselineSpec(loader=_load_smp, kwargs={"model_name": "deeplabv3+resnet50"}),
    "unetplusplus_resnet50": BaselineSpec(loader=_load_smp, kwargs={"model_name": "unet++resnet50"}),
    "dpt_dinov2_base": BaselineSpec(loader=_load_dpt, kwargs={"backbone": "facebook/dinov2-base"}),
    "dpt_dinov2_base_lora": BaselineSpec(
        loader=_load_dpt,
        kwargs={"backbone": "facebook/dinov2-base", "use_lora": True, "modules_to_save": ["head"]},
    ),
    "dpt_dinov2_giant": BaselineSpec(loader=_load_dpt, kwargs={"backbone": "facebook/dinov2-giant"}),
    "dpt_dinov2_giant_lora": BaselineSpec(
        loader=_load_dpt,
        kwargs={"backbone": "facebook/dinov2-giant", "use_lora": True, "modules_to_save": ["head"]},
    ),
    "linear_dinov2_base": BaselineSpec(loader=_load_linear_dino, kwargs={}),
}


def load_coralscapes_baseline(model_key: str, checkpoint_path: str, num_classes: int) -> nn.Module:
    spec = BASELINE_SPECS.get(model_key)
    if spec is None:
        raise ValueError(f"Unknown baseline model '{model_key}'")
    model = spec.loader(checkpoint_path=checkpoint_path, num_classes=num_classes, **spec.kwargs)
    return model


def _initialize_hf_backbone(model_key: str, cache_dir: Optional[str], local_files_only: bool) -> None:
    spec = BASELINE_SPECS.get(model_key)
    if spec is None:
        raise ValueError(f"Unknown baseline model '{model_key}'")

    hf_kwargs = {"local_files_only": local_files_only}
    if cache_dir:
        hf_kwargs["cache_dir"] = cache_dir

    if spec.loader is _load_segformer:
        backbone = spec.kwargs["backbone"]
        SegformerConfig.from_pretrained(backbone, **hf_kwargs)
        print(f"Cached SegFormer backbone '{backbone}'.")
    elif spec.loader is _load_dpt:
        backbone = spec.kwargs["backbone"]
        Dinov2Config.from_pretrained(backbone, **hf_kwargs)
        Dinov2Model.from_pretrained(backbone, **hf_kwargs)
        print(f"Cached DINOv2 backbone '{backbone}' for DPT wrapper.")
    elif spec.loader is _load_linear_dino:
        backbone = "facebook/dinov2-base"
        Dinov2Config.from_pretrained(backbone, **hf_kwargs)
        Dinov2Model.from_pretrained(backbone, **hf_kwargs)
        print("Cached DINOv2 base backbone for linear probe baseline.")
    else:
        print(f"Baseline '{model_key}' does not require Hugging Face downloads.")


def initialize_hf_backbones(model_keys: List[str], cache_dir: Optional[str] = None, local_files_only: bool = False) -> None:
    for key in model_keys:
        _initialize_hf_backbone(key, cache_dir, local_files_only)


__all__ = ["load_coralscapes_baseline", "BASELINE_SPECS", "initialize_hf_backbones"]


def _parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Pre-download Hugging Face assets for CoralScapes baselines.")
    parser.add_argument(
        "--model_keys",
        nargs="+",
        default=list(BASELINE_SPECS.keys()),
        choices=sorted(BASELINE_SPECS.keys()),
        help="Subset of baseline keys to initialize.",
    )
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory override.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Set to verify that requested assets already exist offline.",
    )
    return parser


if __name__ == "__main__":
    args = _parse_args().parse_args()
    initialize_hf_backbones(args.model_keys, cache_dir=args.cache_dir, local_files_only=args.local_files_only)