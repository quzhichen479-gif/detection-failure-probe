# Baseline Reuse and Candidate Training Protocol

## 1. Existing baseline is frozen and must not be retrained

The YOLO11n baseline for this research line already exists. It is the comparison reference.

Codex MUST NOT:

- retrain the baseline;
- launch a second baseline run for convenience;
- replace the baseline with a newly reproduced checkpoint;
- average a new baseline into the existing result;
- silently change the baseline recipe and call it matched.

Use the existing baseline checkpoint, metrics, run metadata, and evaluator outputs.

## 2. Recover the real baseline recipe before candidate training

Before any attention candidate is trained, locate the actual baseline training artifacts in the local project. Prefer primary artifacts such as:

- baseline run `args.yaml` or equivalent saved argument file;
- original training command or script;
- model YAML and data YAML;
- run logs / results CSV;
- checkpoint metadata;
- evaluator configuration and checkpoint-selection record.

Do not reconstruct parameters from memory or from generic Ultralytics defaults when primary artifacts exist.

Create:

`research_tracks/attention_cls_branch/implementation/BASELINE_RECIPE_LOCK.md`

The lock file must record the resolved value and source artifact for every comparison-critical field.

## 3. Fields that must match the preceding baseline

Candidate training must preserve, at minimum:

- dataset identity and split;
- `imgsz`;
- epoch budget;
- patience / early-stopping behavior;
- batch size;
- random seed;
- deterministic/reproducibility flags;
- pretrained initialization policy and starting weights policy;
- optimizer type;
- initial/final learning-rate settings;
- scheduler;
- momentum or optimizer betas;
- weight decay;
- warmup settings;
- loss-side hyperparameters not explicitly part of the candidate;
- all data augmentation switches and probabilities;
- mosaic / mixup / copy-paste or equivalent settings;
- workers and cache policy where relevant to reproducibility;
- AMP setting;
- freeze settings;
- validation settings and cadence;
- checkpoint save policy;
- best-checkpoint selection rule;
- evaluator implementation and metric protocol.

The candidate experiment is allowed to differ only in:

1. model structure / selected attention module and its documented parameters;
2. experiment/run name;
3. output directory.

A device change is allowed only if unavoidable and must be explicitly reported because it may affect timing and numerical reproducibility.

## 4. Pre-launch configuration diff gate

Before launching a candidate, generate a baseline-vs-candidate configuration diff.

The launch must be blocked if any non-allowed field differs.

Write the diff for each candidate under:

`research_tracks/attention_cls_branch/implementation/protocol_diffs/`

For example:

- `caa_p3_vs_baseline.md`
- `lsk_p3_vs_baseline.md`
- `bra_p3_vs_baseline.md`

Each report must end with either:

- `PROTOCOL_MATCH: PASS`
- `PROTOCOL_MATCH: FAIL`

Only `PASS` may proceed to training.

## 5. Ambiguity rule

If the actual preceding-baseline value for a comparison-critical parameter cannot be recovered unambiguously, do not guess and do not substitute the current Ultralytics default.

Stop candidate training and report:

- the unresolved field;
- artifacts searched;
- conflicting values if any;
- what evidence is required to resolve it.

Engineering implementation/tests may continue, but training must not start until the protocol is resolved.

## 6. Evaluation rule

After candidate training, compare it against the already-existing baseline using the same checkpoint-selection and evaluation protocol. Do not create a fresh baseline merely to pair with the candidate.

The independent test set must not be used for iterative module selection or tuning unless a later explicit project decision changes that boundary.
