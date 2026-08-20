# Attention module improvement tasks

This directory is the **attention-module task class** for YOLO11 floating-object research. It is intentionally isolated from `src/failure_probe`: the production package remains lightweight and does not gain a PyTorch dependency.

The implementations here are clean research adapters written for this repository. They preserve the central mechanism of each paper but are **not verbatim copies of official repositories** and should not be described as reproduction-grade implementations until validated against the authors' code.

## Task queue

| Task | Module | Priority | First insertion | Hypothesis | Main failure mode |
|---|---|---:|---|---|---|
| ATT-P0-01 | RALA | P0 | P3 neck, after C3k2 | high-resolution tiny features need global water context without quadratic attention; rank augmentation should reduce linear-attention collapse | global context is not the bottleneck; approximation differs from RAVLT |
| ATT-P0-02 | LSK | P0 | P3 neck, after C3k2 | tiny objects need content-dependent receptive fields to distinguish debris from ripples/reflections | large receptive fields smooth weak tiny features |
| ATT-P0-03 | SHSA | P0 | P3 neck; optional C2PSA replacement control | a small channel subset may carry enough global context, avoiding redundant multi-head attention | P3 quadratic token attention is still too expensive |
| ATT-P1-01 | Boltzmann sparse attention | P1 | P3 neck | uncertain tiny-object locations benefit from broad-to-focused sparse sampling | selector misses weak targets; stochasticity destabilizes training |
| ATT-P1-02 | BRA / BiFormer-style | P1 | P3/P4 neck | coarse content routing can reject large irrelevant water regions before fine attention | routing error suppresses tiny targets; reference loop is slow |

## Why these five

The task class deliberately excludes another CBAM/ECA/CA/EMA/SimAM sweep. The project already has negative evidence for several generic attention/feature-fusion variants. These candidates correspond to materially different hypotheses: rank repair, selective large context, partial-channel global attention, uncertainty-aware sparse attention, and content-aware bi-level routing.

## Common interface

Every adapter accepts and returns an NCHW tensor with the same shape:

```python
from research_tasks.attention import build_attention

module = build_attention("rala", channels=256)
y = module(x)  # [B, 256, H, W] -> [B, 256, H, W]
```

This contract is chosen so a YOLO wrapper can place a module after a neck `C3k2` block without changing Detect head channels.

## Recommended experiment order

1. `RALA@P3`
2. `LSK@P3`
3. `SHSA@P3`
4. stop the attention track if all three are negative under the matched protocol
5. only then run Boltzmann/BRA as higher-risk sparse-attention experiments

Do **not** stack the modules in the first pass. Each experiment should answer one mechanism question.

### Minimal matched controls

- YOLO11 baseline with identical training budget and insertion identity control.
- Same model scale, image size, augmentations, seed policy, max_det and checkpoint rule.
- Report AP50:95, `<8 px` recall, per-river/worst-river, empty-image FP/image, params, FLOPs and measured P50/P95 latency.
- Final promising candidate should be repeated across seeds; a single successful seed is not sufficient.

### Attention-specific diagnostics

Before promoting a candidate, collect at least one mechanism diagnostic:

- **RALA:** numerical rank / singular-value concentration of P3 token features before and after the block.
- **LSK:** selection-weight distribution versus object size and river/background type.
- **SHSA:** latency and gain versus `attn_ratio` (`0.125/0.25/0.5`).
- **Boltzmann:** sampled-region recall for `<8 px` GT and entropy versus temperature.
- **BRA:** routed-region recall for tiny GT and performance versus `topk`.

If a sparse selector fails to include tiny GT neighborhoods, do not interpret downstream AP loss as evidence that attention itself is ineffective; it is a routing failure.

## YOLO11 insertion sketch

Preferred first-pass location:

```text
P3 backbone -> upsample/concat -> C3k2 -> [ATTENTION] -> Detect(P3)
```

Keep the native deep C2PSA unchanged in the first experiment. A second-stage SHSA control may replace C2PSA only after `SHSA@P3` is understood.

Ultralytics integration should be done in the actual training repository by registering one wrapper at a time. This repository intentionally contains framework-neutral NCHW blocks rather than editing a vendored Ultralytics package.

## Smoke test

PyTorch is an optional research dependency and is not added to this repository's production `pyproject.toml`.

```bash
python -m research_tasks.attention.smoke_test
```

## Sources / provenance

- Fan, Huang, He. **Breaking the Low-Rank Dilemma of Linear Attention**, CVPR 2025. Official project: `qhfan/RALA`.
- Li et al. **Large Selective Kernel Network for Remote Sensing Object Detection**, ICCV 2023. Tiny remote-sensing objects and context selection motivate LSK.
- Yun, Ro. **SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design**, CVPR 2024.
- Zhao et al. **Boltzmann Attention Sampling for Image Analysis with Small Objects**, CVPR 2025.
- Zhu et al. **BiFormer: Vision Transformer with Bi-Level Routing Attention**, CVPR 2023.

The LSK authors' released repository is CC BY-NC 4.0; no code from it is copied here. Treat all adapters as independent research implementations and cite the original papers when used experimentally.
