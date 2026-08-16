# Round-4 FreqFusion + DBRA — Codex README

> **后续实现直接从这里启动。**  
> Target: YOLO11n / Ultralytics 8.4.113 / PoTATO  
> Head parent: **fixed DBRA @ P3-Cls-Mid**  
> New neck change: **FreqFusion only at P4 -> P3**  
> Loss: **本轮完全不改，留到后续单独设计。**

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
```

随后必须读取真实 YOLO 工程中**已经跑通并产生当前 DBRA 结果的实现和 YAML**。不要从本 README 猜 DBRA API。

---

## 3. 最容易实现错的地方

### FreqFusion 不是单输入上采样器

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

## 5. 主配置不是 clean-demo 默认，而是官方 detection profile

官方 Faster R-CNN/COCO FreqFusion 配置启用了 resampling。因此本轮主模型固定：

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

`feature_resample=False` 只作为**主模型有效之后的机制消融**，不是首版默认。

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

不要把 detector 实现塞进本仓库 `detection-failure-probe/src`。

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

必须从真实 DBRA parent YAML 修改。

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

## 10. 长训练前门禁

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
```

全部通过后先生成：

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

再允许长训练。

---

## 11. 正式实验只需要这一张初始消融

```text
A0 YOLO11n baseline                                  reuse
A1 YOLO11n + DBRA P3-mid                            reuse
A2 YOLO11n + DBRA + detection-profile FreqFusion   train
```

如果 A2 validation 正向，再做：

```text
A3-core: A2 but feature_resample=False
```

用来判断增益来自 ALPF+AHPF 本身，还是 local-similarity-guided resampling 也有贡献。

损失函数放到下一阶段，绝对不要和 A2 同时改。

---

## 12. Codex 一句话启动提示词

```text
读取 research_tracks/attention_cls_branch/round4_freqfusion_dbra/README.md，并严格按其中“必读文件”顺序读取全部 Round-4 资料，然后检查真实 YOLO11 工程中已验证的 DBRA P3-mid 实现/YAML。实现唯一候选：固定 DBRA P3-Cls-Mid + FreqFusion 仅用于最后的 P4->P3 top-down fusion。FreqFusion 必须作为双输入 FreqFusionConcat([backbone_P3, fused_P4]) 一次性替代原第二组 Upsample+Concat，并保留 YOLO 的 concat 融合语义。上游固定为 Linwei-Chen/FreqFusion commit 3fb0c70637a3c194fb74294d3ce4681958b26241；复制前核对授权。主配置使用官方 object-detection profile：compress_ratio=8、lowpass=5、highpass=3、feature_resample=True、feature_resample_group=4、semi_conv=True、高低通均启用。不要修改 DBRA、loss、TAL、DFL、imgsz、augmentation、optimizer、训练周期、数据划分或 baseline。完成 module/parser/YAML 接入后，必须处理节点 index 改变带来的 state-dict 权重迁移，完成逐键审计、shape/gradient/unit tests、smoke train/val/predict 和性能开销测试，并先生成 ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md。所有门禁通过后，才按冻结协议训练 A2，并只使用 validation 做选择。
```
