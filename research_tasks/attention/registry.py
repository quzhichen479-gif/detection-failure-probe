from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from .boltzmann import BoltzmannSparseAttention
from .bra import BiLevelRoutingAttention
from .lsk import LSKAttention
from .rala import RALAAttention
from .shsa import SHSAAttention


@dataclass(frozen=True)
class AttentionTask:
    key: str
    module: type[nn.Module]
    priority: str
    recommended_stage: str
    hypothesis: str
    risk: str


ATTENTION_TASKS: dict[str, AttentionTask] = {
    "rala": AttentionTask(
        "rala",
        RALAAttention,
        "P0",
        "P3 neck",
        "global context with linear-cost rank repair",
        "medium",
    ),
    "lsk": AttentionTask(
        "lsk",
        LSKAttention,
        "P0",
        "P3 neck",
        "adaptive receptive-field selection for tiny objects",
        "low",
    ),
    "shsa": AttentionTask(
        "shsa",
        SHSAAttention,
        "P0",
        "P3 neck / C2PSA control",
        "partial-channel global attention",
        "low",
    ),
    "boltzmann": AttentionTask(
        "boltzmann",
        BoltzmannSparseAttention,
        "P1",
        "P3 neck",
        "uncertainty-aware sparse focus on tiny regions",
        "high",
    ),
    "bra": AttentionTask(
        "bra",
        BiLevelRoutingAttention,
        "P1",
        "P3/P4 neck",
        "content-aware coarse-to-fine sparse routing",
        "medium-high",
    ),
}


def build_attention(name: str, channels: int, **kwargs: object) -> nn.Module:
    try:
        task = ATTENTION_TASKS[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(ATTENTION_TASKS))
        raise KeyError(f"unknown attention={name!r}; valid: {valid}") from exc
    factory: Callable[..., nn.Module] = task.module
    return factory(channels=channels, **kwargs)
