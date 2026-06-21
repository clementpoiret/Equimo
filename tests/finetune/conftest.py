"""Pytest fixtures for fine-tuning tests."""

from __future__ import annotations

import jax.random as jr
import pytest

from fixtures import (
    TinyASTLikeEncoder,
    TinyConvNeXtLike,
    TinyLinearMLP,
    TinyTextEncoder,
    TinyVisionTransformer,
)


@pytest.fixture
def finetune_key():
    return jr.PRNGKey(0)


@pytest.fixture
def tiny_ast_like_encoder(finetune_key):
    return TinyASTLikeEncoder(key=finetune_key)


@pytest.fixture
def tiny_convnext_like(finetune_key):
    return TinyConvNeXtLike(key=finetune_key)


@pytest.fixture
def tiny_linear_mlp(finetune_key):
    return TinyLinearMLP(key=finetune_key)


@pytest.fixture
def tiny_text_encoder(finetune_key):
    return TinyTextEncoder(key=finetune_key)


@pytest.fixture
def tiny_vision_transformer(finetune_key):
    return TinyVisionTransformer(key=finetune_key)
