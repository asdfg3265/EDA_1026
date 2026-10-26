# PCBParse —— 原理图图片自动解析（2026 创芯·EDA 精英挑战赛 · 赛题六）

本仓库是我们参赛项目的完整快照（2026-10-26）：代码、平台提交包、自建数据集、技术报告和每轮的运行指南都在这里。新加入的同学按本文从头读一遍，就能知道这个项目在做什么、怎么跑起来、目前做到哪一步、接下来还要做什么。

---

## 1. 赛题在做什么

**输入**：一张 PCB 原理图的 PNG 图片（大多是嘉立创 EDA 导出的，也可能来自 Altium、KiCad 等工具，可能带图框和功能框）。

**输出**：一个 JSON 文件，描述图上的电路：
- `components`：每个元件的位号（key，如 `R1`、`U3`）、类型（`r` / `c` / `box` / `gnd` …，共 34 类）、型号 `Name`、取值 `value`、外框 `bbox`；
- `pins`：每个元件的引脚号、引脚名、引脚端点坐标；
- `nets`：网络连接（哪些引脚连在一起）和导线线段。

坐标以图片**左下角为原点**，单位是像素。

**评分**由四项 F1 组成：元件、引脚、网络连接、导线线段（详见 `docs/` 中的评测要点）。公开题 13 道，隐藏题是公开题的两倍；**最终成绩以最后一次提交为准**。另有“数据集附加分”：自建数据集按质量和数量加分。

## 2. 我们的方案（一句话版）

**纯计算机视觉 + 轻量 OCR，不调用大模型**（Token 消耗为 0，单图约 3–6 秒，只用 CPU）：
1. **预处理**：去掉图框、标题栏、虚线功能框；把其他工具的配色重画成嘉立创配色；把画成导线绿色的地符号改回符号色。
2. **按颜色分层**：红色是符号和引脚，绿色是导线，蓝色和黑色是文字，红点是连接点。
3. **导线拓扑**：识别导线、T 形连接和带点十字交叉，按连接点拆分成线段。
4. **符号分割**：找出引脚短桩，把笔画分组成元件，再按形状判型（电阻框、电容板、二极管三角、地符号横杠、网络端口旗形、芯片矩形……）。
5. **文字**：先用 PP-OCRv5 识别模型（onnxruntime）读出文字；再用颜色（位号是藏青，取值是亮蓝）、位置和元件库校正；最后用全局最优匹配把位号、取值、型号分给元件。
6. **组装输出**：引脚编号和引脚名、网络、导线线段，写成赛题 JSON。

## 3. 仓库结构

| 路径 | 内容 |
|---|---|
| `README.md` | 本文件 |
| `code/PCBParse/` | **完整源代码**（解压自 `PCBParse_v20.zip`）。结构见第 5 节 |
| `submission/submission_v16.zip` | **当前平台提交包**（Dockerfile + run.sh + agent.py + 模型 + 代码） |
| `dataset/` | 自建数据集（附加分用），切成 12 个 90 MB 分片，另附 `SHA256SUMS` |
| `check/check_v9_overlay.zip` | 数据集抽查叠图（标注画在原图上，人工目视检查用） |
| `docs/` | 技术报告、方案进展、评测与 Q&A 要点、各轮运行指南 |
| `open_GT-1020.zip` | 官方公开的 13 道题及标准答案（10-20 版） |

## 4. 快速上手（新手照着做）

### 4.1 环境
```bash
conda create -n eda python=3.11 -y && conda activate eda
cd code/PCBParse
pip install -r requirements.txt     # numpy scipy opencv-python-headless onnxruntime pillow
```

### 4.2 跑一张或一批图
```bash
python run.py <输入图片目录> <输出目录>
# 输出：<输出目录>/<case>/<case>.json；进程数可用 PCB_WORKERS=4 指定
```

### 4.3 在公开 13 题上打分
```bash
unzip -q ../../open_GT-1020.zip -d /tmp/ogt         # 得到 /tmp/ogt/open_GT/N/N.png + N.json
python run.py /tmp/ogt/open_GT /tmp/pred
PCB_SCORE_RULE=platform python -m pcbparse.scorer_official /tmp/ogt/open_GT /tmp/pred
# 应得到 MEAN 99.3845（平台现行规则）；去掉 PCB_SCORE_RULE=platform 为 10-22 书面规则 99.79
```

### 4.4 按平台方式自测提交包
```bash
mkdir -p /tmp/sub && cd /tmp/sub && unzip -q <仓库>/submission/submission_v16.zip
mkdir -p /tmp/in && cp /tmp/ogt/open_GT/*/*.png /tmp/in/
python agent.py -input /tmp/in -output /tmp/out     # 结果在 /tmp/out/final_results/{case}_Pred.json
```

### 4.5 还原数据集
```bash
cd dataset
cat team_20260107530_result_dataset.zip.part* > team_20260107530_result_dataset.zip
sha256sum -c SHA256SUMS        # 必须显示 OK
unzip -l team_20260107530_result_dataset.zip | tail -1
```
包内：`dataset/case_XXXX.png` + `case_XXXX.json`（4706 例）、`sources.csv`（每例的来源工程、作者、许可证）、`dataset_description.docx`（数据说明）。

## 5. 代码导读（`code/PCBParse/`）

| 路径 | 作用 |
|---|---|
| `run.py` | 批量推理入口 |
| `agent.py`、`submission/` | 平台入口（`run.sh -input … -output …`），含超时、安全模式重试 |
| `pcbparse/pipeline.py` | 主流程：预处理 → 分层 → 拓扑 → 分割 → 文字 → 组装 → 输出 JSON |
| `pcbparse/frame.py` | 图框、标题栏、虚线功能框去除 |
| `pcbparse/style.py` | 其他工具配色（Altium、KiCad…）重画为嘉立创配色 |
| `pcbparse/greengnd.py` | 画成导线绿色的地符号改回符号色 |
| `pcbparse/colors.py`、`prims.py`、`geom.py` | 墨色分类、笔画提取、连接点、尺度估计 |
| `pcbparse/wiregraph.py` | 导线拓扑与线段 |
| `pcbparse/sym.py` | 引脚检测、符号分组 |
| `pcbparse/assemble.py` | 核心：判型、位号/取值/型号分配、电源地、网络端口、IC 引脚（最大的文件） |
| `pcbparse/text.py`、`textfix.py`、`textclass.py`、`ocr.py` | 文字检测、OCR、常见误读修正、文字分类 |
| `pcbparse/library.py` | 元件库匹配（`data/*.json`，含官方 cpntLibrary） |
| `pcbparse/scorer_official.py` | 本地评分（与平台逐题一致） |
| `easypro/` | 数据引擎：把嘉立创专业版工程源文件渲染成图片，并计算标准答案 |
| `tools/` | 数据集工具：抓取、构建、重算标注、校验、叠图、打包 |
| `docs/` | 各轮运行指南（每一版改了什么、怎么提交） |

**每一项改进都有环境变量开关**（形如 `PCB_XXX=0/1`，在对应文件顶部），出问题时可以单独关掉一项来对比。

## 6. 目前进展（2026-10-26）

### 6.1 分数
| 数据集 | 当前版本 v16（平台现行规则） | 10-22 书面规则 |
|---|---|---|
| 公开 13 题 | **99.38** | 99.79 |
| 开发集 99 例（调参用） | 90.14 | 92.27 |
| 留出集 184 例（从不用于调参） | 89.02 | 91.13 |
| 带图框的开发集 / 留出集 | 88.15 / 87.67 | 90.11 / 89.31 |
| 合成 Altium / KiCad 配色（各 8 例） | — | 82.69 / 85.52 |

- 开发集和留出集是我们自己从开源工程渲染、自动标注的测试集，比公开题难得多（图更大、器件更多、版式更杂），用来衡量隐藏题上的泛化能力。
- 公开题剩下约 0.6 分，主要来自第 12 题 VIN_2 / VIN_4 的编号互换。这个编号按图元创建顺序给出，图上看不出来；我们不针对公开题写规则。

### 6.2 版本历史（摘要，详见 `docs/方案进展与结果*.md`）
| 版本 | 主要内容 |
|---|---|
| v8–v11 | 基础流程；网络端口类型；叠放器件拆分；型号全局分配 |
| v12 | 其他工具配色归一化（Altium / KiCad） |
| v13 | 各类地符号、BGA 引脚号、文字与导线重叠等系统性修正 |
| v14 | 图框、标题栏、虚线功能框去除 |
| v15 | 文字颜色线索（位号与取值）、电源电压名、端口整列分配、叠放电阻拆分、光耦引脚名、按标注规则判型等 |
| v16 | 绿色地符号、十六进制器件 ID、标成 X 的 LED |

### 6.3 已知不足（下一步方向）
- 位号根本没印在图上的元件（标准答案有，图上看不到），无法识别。
- 被导线、箭头压住的文字，OCR 会读错。
- 非嘉立创风格（Altium、KiCad）明显偏低，比嘉立创低约 7–10 分。
- 少数特殊画法：两排排针叠画、数码管、只认出一半引脚的开关等。

## 7. 时间节点与待办

- [ ] **数据集附加分：11/02 前上传**（本仓库 `dataset/` 即第 12 轮重算后的数据包）。
- [ ] **作品截止：11 月 7 日 17:00**，时区待向组委会确认；按较早的时间准备，至少提前 2 小时提交最终版。
- [ ] 最终提交：最后一次提交的结果即最终成绩，只交经过完整回归的版本。
- [ ] 技术报告：秘书处模板发来后，把 `docs/技术报告_v15.docx` 套进模板。

## 8. 每次改代码的规矩（重要）

1. **公开 13 题的输出必须逐字节不变**，或者分数不降，才能提交。
2. 每一版都在开发集、留出集、带图框集、多风格集上全量回归，看逐例的升降，而不只看平均分。
3. **不针对公开题或单个案例写规则**；不记忆、不硬编码标准答案。平台规定严禁绕过评测机制，违者取消资格。
4. **不上传任何账号凭据**：`cookie.txt`、浏览器 HAR 文件、GitHub token 都不能放进仓库。

## 9. 数据集说明（附加分）

- **来源**：嘉立创开源硬件平台（oshwhub.com）上公开源文件的工程。许可证禁止演绎的工程已排除；每例的出处、作者和许可证都在 `sources.csv`，使用时请保留署名并遵守原许可证（含 NC / SA 条款）。
- **图片**：由 `easypro/render.py` 从工程源文件直接渲染，不是截图。
- **标注**：由 `easypro/gt.py` 从同一份源文件计算得到，不是手工标注。转换规则逐项对照官方公开答案确定。
- **质量检查**：全部通过 `tools/validate_dataset.py` 自动校验，并抽样用 `tools/overlay_dataset.py` 生成叠图目视检查。
- **重算与打包**：
  ```bash
  python tools/regen_gt.py --raw <原始工程目录> --dataset <数据集目录>
  python tools/package_dataset.py <数据集目录> --team <官方队伍编号> --out <输出目录>
  ```
