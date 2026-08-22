# Codex task — FloW Phase 5 large-kernel context

Repository: https://github.com/quzhichen479-gif/detection-failure-probe  
Branch: `research/flow-large-kernel-phase5`

Read this directory completely before editing the local training tree:

```text
research/flow_large_kernel_phase5/
```

## Task

Port the three frozen Phase-5 modules into the local Ultralytics **8.4.113** runtime used by the FloW seed79 experiments, create three YOLO11n model YAMLs, pass smoke/parity/export tests, then launch M1-M3 as independent runs under the existing frozen B0 protocol.

## Required local integration

1. Copy/adapt `large_kernel_context.py` into an isolated local module such as:

```text
ultralytics/nn/modules/flow_large_kernel.py
```

2. Export the classes through the normal `ultralytics.nn.modules` import path.

3. Do **not** insert the blocks inline in the shared neck. Create a minimal `FlowLargeKernelDetect(Detect)` that:

- mirrors the exact local 8.4.113 `Detect.__init__` signature rather than assuming upstream `main`;
- calls `super().__init__()`;
- wraps only `self.cv3[0]` with `ContextBeforeTower(build_p3_context(variant, ch[0]), self.cv3[0])`;
- leaves `cv2[0]`, all P4/P5 towers, DFL, inference decode and `Detect.forward()` unchanged;
- does not copy/override `Detect.forward()` unless the local source makes that unavoidable (document if unavoidable).

4. Register `FlowLargeKernelDetect` in `tasks.py` exactly wherever stock `Detect`-family heads receive their input-channel list.

5. Copy stock `yolo11n.yaml` into exactly three files. Change only the final Detect head class/variant:

```text
yolo11n-flow-unireplk-p3cls.yaml   -> variant=unirep
yolo11n-flow-striplkc-p3cls.yaml   -> variant=strip
yolo11n-flow-cepconv-p3cls.yaml    -> variant=cepconv
```

Do not change backbone/neck width, depth, Detect inputs or training hyperparameters.

## Frozen variants

```text
M1: UniRepLKControl(kernel_size=17, gamma_init=0)
M2: StripLKC(strip_kernel=17, gamma_init=0)
M3: CEPConvLKC(kernel_size=17, center_kernel=5, gamma_init=0)
```

No search, no combinations.

## Required tests before training

- run the package tests from this repository;
- add local Ultralytics tests for build, shape, empty-GT, GT backward, AMP finite;
- copy stock Detect weights into each custom head at gamma=0 and prove full raw output parity within numerical tolerance;
- explicitly prove P3 box output and all P4/P5 outputs are unchanged;
- for M1 and M3 deep-copy the model, call deploy materialization, verify PyTorch parity, then ONNX export/parity;
- run TensorRT FP16 smoke if that is part of the frozen project deployment evaluator;
- record Params/GFLOPs/P50/P95 before starting full training.

## Training

Use the existing frozen FloW protocol and split lock. B0 is not rerun. Train M1/M2/M3 independently from scratch with seed79. Use the existing evaluator and existing run/report conventions. The 75-epoch decision is a continuation/stop decision inside the same run; do not restart passed models.

Formal Test is used only after a model completes the pre-registered Val selection rule. Never tune based on Test.

## Output

Commit local implementation/config/test changes and write one report containing:

```text
commit SHA
exact changed files
build/export/parity status
M1/M2/M3 train status
Val/Test metrics when eligible
AP_small / Recall_small
Params/GFLOPs/P50/P95
negative results and stop reasons
```

Do not add GPRA/RS/P2/attention/frequency/downsampling or any other second mechanism in this phase.
