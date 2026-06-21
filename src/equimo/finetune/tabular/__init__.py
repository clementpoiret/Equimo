"""Tabular fine-tuning recipes."""

from . import recipes
from .recipes import freeze_all, head_only, inspect_trainables

__all__ = (
    "freeze_all",
    "head_only",
    "inspect_trainables",
    "recipes",
)
