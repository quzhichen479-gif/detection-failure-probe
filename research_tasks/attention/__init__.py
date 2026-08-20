"""Attention-module research candidates for YOLO11 floating-object experiments."""

from .boltzmann import BoltzmannSparseAttention
from .bra import BiLevelRoutingAttention
from .lsk import LSKAttention
from .rala import RALAAttention
from .registry import ATTENTION_TASKS, AttentionTask, build_attention
from .shsa import SHSAAttention

__all__ = [
    "ATTENTION_TASKS",
    "AttentionTask",
    "BiLevelRoutingAttention",
    "BoltzmannSparseAttention",
    "LSKAttention",
    "RALAAttention",
    "SHSAAttention",
    "build_attention",
]
