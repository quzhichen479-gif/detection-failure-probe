# Round-4 Codex Repository + Training Execution Addendum

> This file is an **authoritative execution addendum** for Round-4 FreqFusion + DBRA.
> It supplements `17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md` and overrides any ambiguous wording about repository location or whether formal training should actually be launched.

## 1. Repository boundary — explicit

### Specification / research repository

Codex must read the Round-4 design and reference assets from the connected GitHub repository:

```text
https://github.com/quzhichen479-gif/detection-failure-probe
```

Required subtree:

```text
research_tracks/attention_cls_branch/
```

This repository is the **research/specification store**. Do **not** implement the detector under:

```text
detection-failure-probe/src/
```

### Actual detector implementation / training repository

The real implementation must be applied to the **current YOLO11 engineering repository/workspace already used for the accepted baseline and DBRA experiments**.

The connected GitHub account currently exposes only `quzhichen479-gif/detection-failure-probe`; there is no separately exposed YOLO11 GitHub repository through this connector. Therefore Codex must use the YOLO11 repository/worktree that is already mounted/open in its execution environment and contains the existing accepted DBRA implementation, training scripts, configs, checkpoints, and experiment artifacts.

Before editing, Codex must positively identify that worktree by verifying all of the following:

```text
[ ] it is the same YOLO11 project used for the accepted DBRA P3-mid experiment
[ ] Ultralytics version is 8.4.113
[ ] the accepted DBRA P3-mid implementation/YAML exists
[ ] the previous fixed training/evaluation artifacts exist
[ ] the baseline/DBRA training command or resolved args can be recovered
```

If multiple YOLO11 worktrees exist, choose the one that contains the accepted DBRA artifacts; do not create a new detector repository just for Round-4.

Record the resolved absolute worktree path, git commit, branch and dirty status in:

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

## 2. Model to implement — exactly one

```text
R4-FD1 = YOLO11n
       + fixed accepted DBRA @ P3-Cls-Mid
       + detection-profile FreqFusion only at final P4 -> P3 top-down fusion
```

FreqFusion is a two-input fusion operator:

```text
backbone P3 (HR) ---------\
                           -> FreqFusionConcat -> C3k2 -> P3' -> DBRA cls-mid
fused P4 (LR) ------------/
```

Do not add another module, loss, IoU, P2, attention, optimizer change, augmentation change, resolution change, or DBRA retuning in this round.

## 3. Source and primary FreqFusion profile

Pin:

```text
repo:   https://github.com/Linwei-Chen/FreqFusion
commit: 3fb0c70637a3c194fb74294d3ce4681958b26241
file:   FreqFusion.py
blob:   b8fa94d418c3094a8d6653712b65037f70daccec
```

Primary detector profile:

```text
compress_ratio          = 8
compressed_channels     = (C_hr + C_lr) // 8
lowpass_kernel          = 5
highpass_kernel         = 3
feature_resample        = True
feature_resample_group  = 4
semi_conv               = True
use_high_pass           = True
use_low_pass            = True
comp_feat_upsample      = True
hr_residual             = True
hamming_window          = True
feature_resample_norm   = True
```

No first-pass parameter sweep.

## 4. Training protocol lock — recover, do not guess

The baseline and DBRA parent have already been trained. **Do not retrain them.**

Codex must recover the exact formal training settings from the accepted prior experiment artifacts before launching R4-FD1. Prefer the actual saved resolved args/config/command from the accepted DBRA run; if that run intentionally inherited the frozen baseline recipe, verify the baseline recipe lock as well.

The Round-4 formal training command must match the accepted parent protocol for all comparison-critical fields, including at least:

```text
dataset YAML / split
model scale = YOLO11n
imgsz
batch
seed
optimizer
initial/final LR and scheduler settings
momentum / weight decay
warmup
epoch count / patience policy
AMP
workers / cache policy
augmentation settings
mosaic/mixup/copy-paste/close_mosaic and related augmentation fields
pretrained initialization policy
device policy, except when hardware availability requires a documented equivalent device
validation settings
Ultralytics version
```

Allowed differences are only those required by this candidate:

```text
model YAML / architecture
run name
output/project directory
new FreqFusion parameters
```

Do not infer missing values from Ultralytics defaults when the accepted run artifacts can resolve them.

Before formal training, generate a machine-readable diff showing:

```text
accepted DBRA resolved args
vs
R4-FD1 resolved args
```

and assert that every comparison-critical field is identical.

## 5. No polling / no rescue policy

This Round-4 request is **not** a hyperparameter search.

Codex must NOT perform:

```text
parameter polling / grid search / random search
kernel-size sweep
compress-ratio sweep
feature_resample_group sweep
multiple seeds in this first formal run
repeated restarts because early metrics look weak
mid-training parameter edits
"try another setting" rescue runs
retraining the same candidate after seeing validation/test results
```

There is exactly one pre-registered headline configuration for the first formal run.

Also do not continuously poll the training process just to keep the session occupied. After successfully launching the formal training job, record the launch command, PID/process identifier if available, run directory, log path, start time, and resolved args. A single immediate launch-health check is sufficient to confirm the process started and did not fail at initialization.

## 6. Mandatory implementation gates before formal training

Complete all gates first:

```text
[ ] source/provenance/license check
[ ] module import
[ ] parser support
[ ] YAML parse
[ ] model build
[ ] HR/LR order and 2:1 spatial checks
[ ] output channels correct
[ ] Detect strides remain 8/16/32
[ ] accepted DBRA class/config/site unchanged
[ ] parent -> R4 state-dict remap/transfer audit
[ ] FP32 forward/loss/backward finite
[ ] AMP finite if formal protocol uses AMP
[ ] ALPF/AHPF/resampler gradients finite
[ ] DBRA gradients finite
[ ] 1-epoch smoke train
[ ] smoke val
[ ] smoke predict
[ ] Params/GFLOPs/VRAM/latency profile
[ ] resolved training-args diff is clean
```

Generate:

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

before the formal run.

## 7. Then launch formal training — mandatory

Once all gates pass, Codex must **immediately launch the single formal R4-FD1 training run** using the exact frozen parent training parameters.

Do not stop after implementation/report generation and merely tell the user which command to run.

The required workflow is:

```text
implement
-> audit
-> smoke test
-> write integration report
-> resolve exact prior training args
-> assert protocol diff
-> launch one formal R4-FD1 training run
-> perform one immediate launch-health check
-> record launch metadata
-> return control without continuous polling
```

The formal run is not allowed to change parameters based on smoke-test metrics.

## 8. Required launch record

After launch, write/update:

```text
ROUND4_FREQFUSION_DBRA_TRAIN_LAUNCH.md
```

It must include:

```text
YOLO worktree absolute path
git commit / branch / dirty status
model YAML
exact training command
resolved args snapshot
source run/config used to recover the frozen protocol
candidate run name
output directory
log path
PID/process identifier if available
launch timestamp
one immediate health-check result
statement: no parameter sweep, no rescue run, no repeated launch
```

Do not claim training is completed until it actually completes in a later interaction/check.

## 9. Exact Codex startup prompt

Use the following prompt verbatim or equivalently:

```text
Use GitHub repository https://github.com/quzhichen479-gif/detection-failure-probe as the Round-4 research/specification source. Read research_tracks/attention_cls_branch/round4_freqfusion_dbra/README.md first, then read the required files it lists, including 16_ROUND4_FREQFUSION_DBRA_DESIGN.md, reference_code/freqfusion_yolo_adapter.py, reference_code/test_freqfusion_yolo_adapter.py, 17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md, and 18_CODEX_ROUND4_REPO_AND_TRAINING_EXECUTION.md. Do NOT implement detector code under detection-failure-probe/src. Apply the implementation to the current YOLO11 engineering repository/worktree that already contains the accepted YOLO11n DBRA P3-mid experiment and its artifacts; verify that it is the correct worktree and record its absolute path, git commit/branch/dirty status. Implement exactly one candidate: fixed accepted DBRA P3-Cls-Mid + detection-profile FreqFusion only at the final P4->P3 top-down fusion, using FreqFusionConcat([backbone_P3, fused_P4]). Keep the accepted DBRA implementation/config/site unchanged. Do not modify loss, TAL, DFL, P2, imgsz, augmentation, optimizer, scheduler, epoch count, seed policy, dataset split, or any other comparison-critical training setting. Recover the exact training parameters from the accepted prior DBRA/baseline run artifacts; do not guess from defaults. Complete source/provenance checks, parser/YAML integration, explicit state-dict transfer/remap audit, unit/shape/gradient tests, one-epoch smoke train, smoke val/predict, and cost profiling, then write ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md. After all gates pass, do NOT stop at implementation: immediately launch exactly one formal R4-FD1 training run using the same frozen parameters as the accepted prior model, with differences only for model YAML, run name/output directory, and the new FreqFusion architecture. Do not perform parameter polling, grid/random search, multiple rescue runs, repeated restarts, or retuning. After launch, perform only one immediate health check, write ROUND4_FREQFUSION_DBRA_TRAIN_LAUNCH.md with the exact command/resolved args/PID or process id/run directory/log path/start time, and return without continuously polling the training process.
```
