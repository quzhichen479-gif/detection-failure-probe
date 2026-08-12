# Detection Failure Probe — Implementation Report

Date: 2026-08-12
Local version: `0.1.0a0`
Git state: local repository initialized on branch `main`; no commit and no remote repository created.

## 1. 实际实现了什么

### YOLO Dataset Audit

- 安全读取 YOLO dataset YAML、图片目录/单图/文本列表和五列检测标签。
- 类别框数量、覆盖图片数、原始分辨率下的 box 宽/高/面积/长宽比统计。
- COCO-style small/medium/large 面积桶和小目标比例。
- 检测 malformed fields、非数值、NaN/Infinity、非法类别、非正尺寸、越界坐标、越出图片的框。
- 检测 missing labels、orphan labels（对应 configured images 缺失）、empty-label images、不可读图片。
- 检测同类 exact duplicate 和 IoU 阈值可配置的 near-duplicate annotations。
- 标记小于 2 px 的 tiny boxes 和至少 10:1 的 extreme-aspect boxes。
- 生成包含逐图、逐框、逐 issue 记录的 `audit.json`，可直接供研究脚本使用。

### Prediction Failure Analysis

- 接受明确、非 pickle 的 JSON prediction contract：`xywh`/`xyxy`、pixel/normalized、嵌套或 flat list。
- 置信度优先、class-aware、greedy IoU matching。
- 计算 TP、FP、FN、precision、recall 和 per-class 指标。
- 将 unmatched predictions 细分为：
  - `localization_error`
  - `classification_error`
  - `duplicate_detection`
  - `background_false_positive`
- 按 confidence bucket 和 object scale 汇总失败。
- 保留逐图 GT/prediction 状态、匹配 GT index 和 IoU，供 Review UI 叠框与筛选。
- 明确指标口径：具体失败类型仍属于 FP；unmatched GT 仍属于 FN。

### Resolution Survival Analysis

- 对多个 square input resolution 进行 aspect-preserving resize 几何计算。
- 输出每个目标在各分辨率下的 pixel width、height、area 和 minimum-side scale bucket。
- 输出 min side ≥ 1/2/4/8/16 px 的 count 与 ratio。
- 在 JSON、HTML 和 README 中均明确：这是几何诊断，不能解释为模型性能预测。

### Local Review UI

- 标准库 `ThreadingHTTPServer`，无 Node/前端构建依赖。
- Canvas 显示原图、GT、prediction box 和失败类型颜色。
- 支持 FP、FN、localization、classification、duplicate detection、suspicious annotation 筛选。
- 支持 class 和 minimum confidence 筛选。
- 展示 duplicate/suspicious audit flags。
- 支持 reviewer note 和 `reviewed` / `confirmed_suspicious` / `confirmed_duplicate` 标记。
- Notes 原子写入当前 run 的 `reviewer_notes.json`。

### CLI 与 Python API

已实现并实际运行：

```text
failure-probe audit dataset.yaml
failure-probe analyze --dataset dataset.yaml --predictions predictions.json
failure-probe review runs/<run>
failure-probe report runs/<run>
```

核心逻辑位于 Python API，不在 CLI 中堆叠：

- `load_dataset`
- `audit_dataset`
- `analyze_predictions`
- `resolution_survival`
- `generate_report`
- `run_audit` / `run_analysis`

### OSS 工程基础

- `pyproject.toml`、editable install、console script。
- pytest 测试、ruff 配置、GitHub Actions（Python 3.10/3.12）。
- README、MIT LICENSE、CONTRIBUTING、SECURITY、`.gitignore`、`.gitattributes`。
- local Git repository 已初始化为 `main` 分支。
- 不含 API key、token、遥测、上传客户端、真实用户数据、虚构 benchmark 或虚构社区指标。

## 2. 项目目录结构

```text
.
├── .github/
│   └── workflows/ci.yml
├── demo/
│   ├── dataset/
│   │   ├── dataset.yaml
│   │   ├── images/
│   │   │   ├── scene_01.png
│   │   │   ├── scene_02.png
│   │   │   ├── scene_03.png
│   │   │   └── scene_04.png
│   │   └── labels/
│   │       ├── orphan.txt
│   │       ├── scene_01.txt
│   │       ├── scene_02.txt
│   │       └── scene_03.txt
│   └── predictions.json
├── scripts/
│   └── generate_demo.py
├── src/failure_probe/
│   ├── __init__.py
│   ├── analysis.py
│   ├── audit.py
│   ├── cli.py
│   ├── dataset.py
│   ├── errors.py
│   ├── geometry.py
│   ├── models.py
│   ├── paths.py
│   ├── report.py
│   ├── resolution.py
│   ├── review.py
│   └── workflow.py
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_audit.py
│   ├── test_geometry.py
│   └── test_paths_and_workflow.py
├── .gitattributes
├── .gitignore
├── CONTRIBUTING.md
├── IMPLEMENTATION_REPORT.md
├── LICENSE
├── README.md
├── SECURITY.md
└── pyproject.toml
```

`runs/` 是实际验证时生成的本地输出，已被 `.gitignore` 排除，不应提交 reviewer notes 或用户数据。

## 3. 实际执行并成功的命令

以下命令均在当前 Windows 工作目录中实际成功执行。由于机器 PATH 最前方原有一个失效的 Python 3.12 `pip.exe`/`pytest.exe`，验收前先把当前 Python 3.13 的 Scripts 目录临时置于 PATH 首位；未修改系统环境变量。之后下列字面命令成功：

```powershell
pip install -e .
pytest
ruff check .
```

其他成功命令：

```powershell
python scripts/generate_demo.py
failure-probe audit demo/dataset/dataset.yaml --run-name demo_audit
failure-probe analyze --dataset demo/dataset/dataset.yaml `
  --predictions demo/predictions.json --run-name release_candidate
failure-probe report runs/release_candidate
python -m pip check
python -m compileall -q src scripts tests
failure-probe --version
failure-probe --help
```

Review 服务还进行了真实 HTTP smoke test：短暂启动 `failure-probe review`，随后实际请求 UI、`/api/data`、一张白名单 demo PNG 和 `/api/notes`，四项均返回 HTTP 200，note 返回 `saved: true`；测试后服务 PID 已关闭。

## 4. 测试结果

最终测试结果：

```text
collected 13 items
12 passed, 1 skipped in 0.24s
```

唯一 skip 是 Windows 当前账户没有创建 directory symlink 的权限，因此 symlink-run rejection 用例在本机跳过；测试在具备 symlink 权限的 Linux GitHub Actions runner 上会正常执行。

最终 lint：

```text
All checks passed!
```

最终依赖检查：

```text
No broken requirements found.
```

最终 synthetic demo 实际结果：

```text
Images: 4
Valid boxes: 4
Invalid boxes: 1
Audit issues: 6
TP: 2
FP: 5
FN: 2
Failure types exercised:
  background_false_positive
  classification_error
  duplicate_detection
  localization_error
Resolutions: 320, 640, 1280
```

Demo 还真实包含并检出了 missing label、orphan label、empty label、exact duplicate annotation、invalid box 和 suspicious extreme-aspect box。

## 5. 已知限制

- 只支持 YOLO 五列 axis-aligned detection labels；不支持 segmentation、keypoints、rotated boxes。
- 不直接导入 COCO JSON、Ultralytics result objects 或其他框架对象；当前需要转换成文档化 JSON。
- 只检测 annotation duplicate/near-duplicate，不做图片内容 perceptual deduplication。
- Greedy matching 是可解释诊断，不是 COCO evaluator/TIDE 的完整替代；没有 crowd、ignore region、area-range AP、mAP。
- 未配置的预测 class id 会计入总 FP，但不会出现在 configured per-class 表中。
- Resolution Survival 假定 aspect-preserving resize 到 square canvas，padding 不改变目标尺寸；不模拟 stride、augmentation、feature pyramid 或具体模型。
- 当前会把 audit 和 review 数据一次性载入内存；超大数据集还需要 streaming/indexing/progress 支持。
- HTML report 只展示前 100 条 findings，完整结果在 JSON 中。
- 未实现数据集自动修复，这是刻意的 MVP 安全/可审计边界。
- 当前只在本机 Windows/Python 3.13 实际运行；Python 3.10/3.12 由 CI matrix 定义，但远程 Actions 尚未发生，因为仓库尚未上线。

## 6. 当前真实安全攻击面

已做的安全控制：

- YAML 使用 `yaml.safe_load`，不执行 dataset 中的 Python 或 shell 内容。
- Prediction 只接受 JSON；拒绝 pickle、checkpoint、URL、archive 和非标准 NaN/Infinity。
- YAML、image list、label、prediction JSON 均有限制大小。
- 所有 dataset-referenced paths 在任何图片/标签内容读取前解析并限制在 YAML 所在根目录内；拒绝 `..`、外部绝对路径和 symlink escape。
- Run name 使用 allowlist；新 run 目录独占创建，已有目录不会被复用或静默覆盖。
- Report/note 写入前验证 run marker，拒绝 run/artifact symlink，并使用同目录临时文件 + `os.replace` 原子写入。
- Review 只允许 `127.0.0.1`/`localhost`/`::1`，使用进程级随机 token、图片白名单、CSP、安全响应头、64 KB note request 上限、2,000 字符 note 上限和 100 MB 单图片响应上限。
- 包内没有外传数据的网络客户端、遥测或 analytics。

仍然存在的真实攻击面：

- Pillow 图片解码器和浏览器图片解码器仍处理用户提供的图片；Pillow decompression-bomb exception 已捕获，但第三方解码器漏洞仍依赖及时升级 Pillow/浏览器。
- 大量文件、深目录、复杂但小于限制的 YAML aliases、接近 100 MB 的 JSON、许多 annotations 可能造成 CPU/内存/磁盘 DoS；本工具不提供 OS 级资源隔离。
- Review 启动期间，同机恶意进程、浏览器扩展和拥有文件权限的其他用户属于本机信任边界；随机 token 不是对本机管理员/恶意软件的防御。
- Reviewer notes、JSON 和 HTML 是本地明文，任何拥有目录读取权限的人都能读取。
- 原子写入可避免部分损坏和普通覆盖风险，但无法消除拥有同等文件权限的本地进程故意制造的所有 TOCTOU race。
- 用户显式指定的 `--runs-dir` 可以位于任意用户有权写入的位置；工具只在其下创建新的安全命名子目录。这是明确用户授权的输出边界，不是 sandbox。

对真正敌对的数据集，仍应在低权限容器/VM 中运行，并限制 CPU、内存、文件数量和磁盘。

## 7. 是否已经适合公开 GitHub

**适合作为诚实标注的 alpha/MVP 公开。** 当前已经具备可安装包、清晰输入 contract、完整 demo、核心测试、lint、CI 配置、隐私/安全文档和贡献指南；陌生用户可按 README 完成 install → run demo → inspect result。

但它尚未是正式 `v0.1.0` release：远程仓库没有创建，GitHub Actions 没有真实运行，尚无 commit/tag/release，也没有经过外部数据集兼容性验证。公开时应保留当前 `0.1.0a0`/Alpha 定位，不应声称 production-ready。

## 8. 推荐的 3 个 repo name

1. `detection-failure-probe` — 最清晰，和当前包名、CLI 概念一致，推荐。
2. `failure-probe` — 更短、更易输入，但搜索语义略宽。
3. `detprobe` — 最简洁、适合传播，但首次看到时含义不如首选明确。

名称的 GitHub 可用性尚未在线核验。

## 9. 推荐 GitHub description

> Local-first YOLO dataset auditing, object-detection failure analysis, resolution diagnostics, and visual review — CPU-only and privacy-preserving.

推荐 topics：`object-detection`, `yolo`, `dataset-audit`, `failure-analysis`, `computer-vision`, `local-first`。

推荐仓库地址：

<https://github.com/quzhichen479-gif/detection-failure-probe>

这只是推荐地址；本次实现没有在线创建该仓库。

## 10. 从当前状态到 v0.1.0 还差什么

发布前最低清单：

1. 在 GitHub 创建空仓库，审阅后完成首次 commit/push；不要提交 `runs/`。
2. 观察 GitHub Actions 在 Python 3.10 和 3.12 上真实通过，尤其确认 Linux symlink 用例执行而非 skip。
3. 增加至少一次 macOS 或额外 Linux smoke test，确认 report/review 路径行为一致。
4. 把 Review server 构造从 blocking API 中拆出可注入 server fixture，将当前人工 HTTP smoke test变成 CI 自动测试。
5. 用 2–3 个可公开的小型真实 YOLO 数据集验证路径布局和 prediction conversion 文档；只记录可复现兼容性，不虚构 benchmark。
6. 为大文件/多文件场景做基础 profiling，并决定 v0.1 的合理对象数、JSON 大小和 UI 数据量边界。
7. 固定 `audit.json` / `analysis.json` / `resolution.json` 的 v1 schema contract，补 schema migration policy。
8. 增加 wheel/sdist build-and-install CI smoke test，决定是否发布 PyPI；发布时将版本从 `0.1.0a0` 提升到 `0.1.0`。
9. 在 GitHub 启用 private vulnerability reporting，并确认 SECURITY.md 中的实际联系路径可用。
10. 创建签名/可追溯的 `v0.1.0` tag 和 release notes；不添加虚构 stars、用户、下载量、contributors、issues 或 benchmarks。
