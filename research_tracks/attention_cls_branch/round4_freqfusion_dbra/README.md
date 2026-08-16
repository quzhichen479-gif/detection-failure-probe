# Round-4 FreqFusion + DBRA — Codex README

> **后续实现直接从这里启动。**  
> Research/spec repository: **`https://github.com/quzhichen479-gif/detection-failure-probe`**  
> Implementation target: **Codex 当前已打开/挂载、且包含已验证 DBRA 实验的 YOLO11 工程 worktree**  
> Target: YOLO11n / Ultralytics 8.4.113 / PoTATO  
> Head parent: **fixed accepted DBRA @ P3-Cls-Mid**  
> New neck change: **FreqFusion only at P4 -> P3**  
> Loss: **本轮完全不改，留到后续单独设计。**

---

## 0. 仓库边界必须先确认

Round-4 的设计资料从这个已连接 GitHub 仓库读取：

```text
https://github.com/quzhichen479-gif/detection-failure-probe
```

具体目录：

```text
research_tracks/attention_cls_branch/
```

**不要**把 detector 实现写进：

```text
detection-failure-probe/src/
```

真正的模型代码、YAML、训练脚本和训练输出必须写入 Codex 当前环境中**此前已经用于 YOLO11n baseline / DBRA P3-mid 实验的 YOLO11 工程 worktree**。

当前 GitHub 连接只暴露 `quzhichen479-gif/detection-failure-probe`，没有单独暴露一个 YOLO11 GitHub repository。因此 Codex 不要新建第二个 YOLO 仓库；应先在本地/当前工作区定位那个已有 DBRA checkpoint、YAML、训练记录和 evaluator 产物的 YOLO11 worktree。

开始修改前必须记录：

```text
YOLO worktree absolute path
git branch
git commit
git dirty status
Ultralytics version = 8.4.113
accepted DBRA P3-mid implementation/YAML path
accepted DBRA/baseline resolved training args source
```

---

## 1. 本轮只实现什么

```text
YOLO11n
  + FreqFusion(P4 -> P3)
  + 已验证 DBRA(P3 classification-mid)
```

两个组件职责必须分开：

```text
FreqFusion -> 处理 P4 语义特征与高分辨率 P3 特征的跨尺度重建/对齐
DBRA       -> 在最终 P3 分类分支内做内容依赖的上下文路由
```

禁止同时加入：

```text
新 loss / IoU / DFL / TAL
P2
GRN / Slide / FocalMod / 其他 attention
imgsz 改动
augmentation / optimizer / LR / epoch 改动
DBRA 参数或位置改动
```

---

## 2. 必读文件

按顺序读取：

```text
research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md
research_tracks/attention_cls_branch/07_DBRA.md
research_tracks/attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md
research_tracks/attention_cls_branch/16_ROUND4_FREQFUSION_DBRA_DESIGN.md
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/reference_code/test_freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md
research_tracks/attention_cls_branch/18_CODEX_ROUND4_REPO_AND_TRAINING_EXECUTION.md
```

随后必须读取真实 YOLO 工程中**已经跑通并产生当前 DBRA 结果的实现、YAML、训练 args 和历史运行记录**。不要从本 README 猜 DBRA API 或训练参数。

`18_CODEX_ROUND4_REPO_AND_TRAINING_EXECUTION.md` 对仓库定位和正式训练启动具有最高执行优先级；如果旧文档存在“实现完成后再由用户手动训练”之类的歧义，以 `18` 为准。

---

## 3. 最容易实现错的地方：FreqFusion 不是单输入上采样器

错误：

```text
P4 -> FreqFusion -> Concat(P3)
```

正确：

```text
backbone P3 (HR) ---------\
                           -> FreqFusionConcat -> C3k2 -> P3' -> DBRA cls-mid
fused P4 (LR) ------------/
```

调用：

```python
_, hr_refined, lr_reconstructed = freqfusion(
    hr_feat=backbone_p3,
    lr_feat=fused_p4,
)
out = torch.cat((hr_refined, lr_reconstructed), dim=1)
```

所以 `FreqFusionConcat` 一次性替换原 YOLO11 最后一次 top-down 的：

```text
nearest x2 + Concat(backbone P3)
```

输入顺序固定：

```text
[HR, LR] = [backbone P3, fused P4]
```

---

## 4. 上游源码固定

```text
repo:   https://github.com/Linwei-Chen/FreqFusion
commit: 3fb0c70637a3c194fb74294d3ce4681958b26241
file:   FreqFusion.py
blob:   b8fa94d418c3094a8d6653712b65037f70daccec
```

复制上游代码前先确认 license/redistribution 条款并记录 provenance。

如果使用 pinned clean source 的非-MMCV CARAFE fallback，需要删除其中 tensor-shape 的 debug `print(...)`，并在 `SOURCE.md` 记录为 semantic-neutral patch。

---

## 5. 主配置使用官方 detection profile

```text
compress_ratio=8
compressed_channels=(C_hr+C_lr)//8
lowpass_kernel=5
highpass_kernel=3
feature_resample=True
feature_resample_group=4
semi_conv=True
use_high_pass=True
use_low_pass=True
comp_feat_upsample=True
hr_residual=True
hamming_window=True
feature_resample_norm=True
```

`feature_resample=False` 只作为**主模型有效之后的机制消融**，不是首版默认。本轮不做任何 FreqFusion 参数轮询或搜索。

---

## 6. 参考实现

项目自写 adapter：

```text
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
```

核心类：

```python
FreqFusionConcat
FreqFusionConcatDebug
```

参考测试：

```text
research_tracks/attention_cls_branch/reference_code/test_freqfusion_yolo_adapter.py
```

建议移植到真实 YOLO 工程：

```text
ultralytics/nn/modules/freqfusion_yolo.py
```

---

## 7. parser 接入

在 `parse_model()` 增加显式多输入分支：

```python
elif m is FreqFusionConcat:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("FreqFusionConcat requires [hr_source, lr_source]")
    hr_c, lr_c = (ch[x] for x in f)
    args = [hr_c, lr_c, *args]
    c2 = hr_c + lr_c
```

不要把它当普通单输入 base module。

---

## 8. YAML 接入

必须从真实 accepted DBRA parent YAML 修改。

stock-like 语义示例：

```yaml
# fused P4 已存在
- [[4, 13], 1, FreqFusionConcat, []]   # HR=P3, LR=P4
- [-1, 2, C3k2, [256, False]]         # new P3
...
- [[15, 18, 21], 1, AttnDetect, <EXACT_EXISTING_DBRA_ARGS>]
```

节点减少会导致后续 index 变化。真实 index 必须通过实际 parent graph 重算，不要照抄示例数字。

---

## 9. 权重迁移是强制门禁

因为 `Upsample + Concat` 两节点合成一个，后续 `model.<index>` 可能整体偏移。

必须生成：

```text
implementation/ROUND4_PARENT_CONFIG_DIFF.md
```

并逐项审计：

```text
parent DBRA keys
R4 keys
仅因 index shift 改名的语义同一模块
显式 remap 后的加载结果
DBRA 权重是否完整迁移
FreqFusion 新参数列表
```

不能只看 `Transferred X/Y items`。

---

## 10. 正式训练参数必须与前置模型一致

baseline 和 DBRA parent 都已经存在，**不得重新训练或重新评价它们来启动本轮**。

Codex 必须从此前 accepted DBRA/baseline 的真实 artifacts 中恢复完整 training args，优先读取保存的 resolved args / train command / config，而不是使用当前 Ultralytics 默认值猜测。

除以下内容允许变化外：

```text
model YAML / architecture
run name
output/project directory
FreqFusion 新结构参数
```

其他所有 comparison-critical training 参数必须和此前固定协议一致，包括至少：

```text
dataset/split
imgsz
batch
seed
optimizer
LR / scheduler
momentum / weight_decay
warmup
epochs / patience policy
AMP
workers/cache
全部 augmentation 参数
pretrained initialization policy
validation settings
Ultralytics version
```

正式训练前必须输出 accepted parent args 与 R4 args 的结构化 diff，并确认除允许差异外全部相同。

---

## 11. 长训练前门禁

```text
[ ] 上游来源/授权确认
[ ] import / YAML parse / model build
[ ] HR/LR 顺序正确
[ ] H_hr = 2*H_lr, W_hr = 2*W_lr
[ ] 输出通道 = C_hr + C_lr
[ ] Detect stride 仍为 8/16/32
[ ] DBRA class/config/site 与 parent 完全一致
[ ] parent -> R4 权重迁移审计
[ ] FP32 forward/loss/backward finite
[ ] AMP finite（若正式协议启用）
[ ] ALPF/AHPF/resampler 都有 finite gradient
[ ] DBRA gradient finite
[ ] 1-epoch smoke train
[ ] smoke val / predict
[ ] Params/GFLOPs/VRAM/latency
[ ] parent-vs-R4 resolved training args diff 仅包含允许差异
```

全部通过后先生成：

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

**然后必须立即拉起一次正式 A2/R4-FD1 训练，不是停下来只给训练命令。**

---

## 12. 单次正式训练：必须启动，但不做轮询

正式实验只有：

```text
A0 YOLO11n baseline                                  reuse
A1 YOLO11n + DBRA P3-mid                            reuse
A2 YOLO11n + DBRA + detection-profile FreqFusion   launch exactly once
```

门禁通过后，Codex 必须：

```text
1. 使用前置 accepted 模型完全相同的冻结训练参数；
2. 启动且只启动一次 A2 正式训练；
3. 做一次立即的 launch-health check，确认不是启动即报错；
4. 记录 exact command / resolved args / PID或process id / run dir / log path / start time；
5. 写入 ROUND4_FREQFUSION_DBRA_TRAIN_LAUNCH.md；
6. 不持续轮询训练进度，返回控制权。
```

禁止：

```text
参数轮询
网格/随机搜索
多 seed 首轮并发
因为前几轮指标不好就重启
救参
重复训练同一配置
边训练边改参数
```

如果未来 A2 validation 正向，再单独注册 `feature_resample=False` 的 attribution control；本次不要自动启动它。

---

## 13. Codex 一句话启动提示词

```text
使用 GitHub 仓库 https://github.com/quzhichen479-gif/detection-failure-probe 作为 Round-4 设计/规范来源。先读取 research_tracks/attention_cls_branch/round4_freqfusion_dbra/README.md，再严格按其中“必读文件”顺序读取全部资料，特别是 16_ROUND4_FREQFUSION_DBRA_DESIGN.md、reference_code/freqfusion_yolo_adapter.py、reference_code/test_freqfusion_yolo_adapter.py、17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md 和 18_CODEX_ROUND4_REPO_AND_TRAINING_EXECUTION.md。不要把 detector 代码实现到 detection-failure-probe/src；实际修改 Codex 当前已经打开/挂载、且包含此前 accepted YOLO11n DBRA P3-mid 实验及其 checkpoint/YAML/训练 artifacts 的 YOLO11 工程 worktree，先确认并记录其绝对路径、git commit/branch/dirty status。实现唯一候选：固定 accepted DBRA P3-Cls-Mid + detection-profile FreqFusion 仅用于最后的 P4->P3 top-down fusion；FreqFusion 必须作为双输入 FreqFusionConcat([backbone_P3, fused_P4]) 一次性替代原第二组 Upsample+Concat，并保留 YOLO concat 语义。上游固定为 Linwei-Chen/FreqFusion commit 3fb0c70637a3c194fb74294d3ce4681958b26241；主配置使用 compress_ratio=8、lowpass=5、highpass=3、feature_resample=True、feature_resample_group=4、semi_conv=True、高低通均启用。不要修改 DBRA、loss、TAL、DFL、P2、imgsz、augmentation、optimizer、scheduler、epoch、seed policy、数据划分或任何比较关键训练设置。必须从此前 accepted DBRA/baseline 的真实运行 artifacts 中恢复精确训练参数，不要用默认值猜测，并在正式训练前做 resolved-args diff。完成 source/provenance、module/parser/YAML、显式 state-dict remap/逐键迁移审计、unit/shape/gradient tests、1-epoch smoke train、smoke val/predict 和成本测试，生成 ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md。所有门禁通过后不要停在“实现完成”：立即使用与前置 accepted 模型完全相同的冻结训练参数启动且只启动一次正式 R4-FD1/A2 训练，差异只允许 model YAML、run name/output path 和 FreqFusion 架构本身。禁止参数轮询、网格/随机搜索、救参、重复重启或重复训练。启动后只做一次立即健康检查，写 ROUND4_FREQFUSION_DBRA_TRAIN_LAUNCH.md，记录 exact command、resolved args、PID/process id、run dir、log path、start time，然后不要持续轮询训练进度，直接返回。
```
