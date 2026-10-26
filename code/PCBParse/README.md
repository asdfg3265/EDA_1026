# PCBParse —— AI 驱动的 PCB 原理图自动解析（EDA 精英挑战赛 赛题六）

把一张嘉立创 EDA / EasyEDA 风格的原理图 PNG 解析成赛题规定的 JSON（components / pins / nets）。
全流程为**确定性计算机视觉 + 轻量 OCR（PP-OCRv5 识别模型，onnxruntime 推理）+ 元器件库校正**，
**不调用任何大模型 API，Token 消耗为 0**（Score_Token 区间满分），单图 2–6 秒（CPU），40 张图串行 < 5 分钟。

## 1. 快速开始

```bash
pip install -r requirements.txt            # numpy scipy opencv-python-headless onnxruntime pillow
python run.py <输入图片目录> <输出目录>     # 递归查找 png/jpg，输出 <输出目录>/<case>/<case>.json
```

* 并行：默认使用 `min(cpu/2, 16)` 个进程，可用 `PCB_WORKERS=8` 指定。
* GPU 不是必需的；如安装了 `onnxruntime-gpu` 也可正常运行（OCR 模型很小，CPU 已足够快）。
* 元器件库：`data/BoxICLibrary.json`、`data/cpntLibrary.json`（赛题官方库）。官方库更新后直接覆盖这两个文件，
  或用环境变量 `PCB_LIBS=/path/a.json:/path/b.json` 追加。
* Docker：`docker build -t pcbparse . && docker run --rm -v $IN:/data/input -v $OUT:/data/output pcbparse`

## 2. 本地评测

```bash
python -m pcbparse.scorer <GT目录> <预测目录> [-v]
```
`pcbparse/scorer.py` 按赛题指南 + Q&A 复现了 ComponentF1 / PinF1 / NetHypergraphF1 / NetLineF1 四项指标
（∅ 匿名电源/地按坐标匹配、同名 GND/VCC 合并、非 box 元件引脚号不计分、线段匈牙利匹配 5px）。
官方评测代码发布后请以官方为准。

在官方公开的 13 个 open_GT 案例上：平台现行规则 **99.38**，10-22 书面规则 99.79（`PCB_SCORE_RULE=platform python -m pcbparse.scorer_official <GT> <预测>`；本节下面的 `pcbparse.scorer` 是早期的近似实现）。当前进展与各数据集分数见仓库根目录的 README 和 `docs/` 中最新一轮运行指南。

## 3. 目录

| 路径 | 说明 |
|---|---|
| `run.py` | 批量推理入口 |
| `pcbparse/colors.py` | 基于"背景差分向量方向"的墨色分类（红=器件/引脚、绿=导线、蓝/黑=文字、灰=未连接标记） |
| `pcbparse/geom.py` | 水平/竖直笔画提取（形态学开运算 + 连通域 + 共线缝隙桥接） |
| `pcbparse/prims.py` | 尺度估计、导线、连接点（红色实心圆）、No-Connect “X”、灰点 |
| `pcbparse/wiregraph.py` | 导线拓扑：T 接/L 角/带点十字连通，不带点十字不连通；按连接点拆分线段（edges） |
| `pcbparse/sym.py` | 符号分割：引脚（直线短桩）检测、引脚–引脚直连（contact）拆分、锚定分组、单位长度 u 估计 |
| `pcbparse/text.py`, `ocr.py` | 文字连通域成行（横/竖）、去除穿字导线、PP-OCRv5 识别 + 约束解码 |
| `pcbparse/assemble.py` | 位号/值/型号/网络标签/电源地 关联，类型判定，IC 引脚号与引脚名，bbox 规则 |
| `pcbparse/library.py` | 元器件库模糊匹配：纠正 IC 型号与引脚名、按引脚名反推引脚号 |
| `pcbparse/pipeline.py` | 端到端 + JSON 输出（左下角坐标系，整数像素） |
| `pcbparse/scorer.py` | 本地评分器 |
| `tools/preannotate.py` | 数据贡献：自动预标注 + 叠加图 |
| `tools/annotator.html` | 数据贡献：浏览器标注校对工具（离线单文件） |
| `tools/validate_dataset.py` | 数据贡献：按《数据提交说明》自动检查 |
| `tools/package_dataset.py` | 数据贡献：打包 `team_<id>_result_dataset.zip` |

## 4. 数据贡献（附加分 10 分）流程

1. 从嘉立创开源广场（oshwhub）等开源渠道导出原理图 PNG（无图框、彩色、不与 200 个训练集重复），放入 `raw/`；
2. `python tools/preannotate.py raw/ dataset/` 自动生成标注 JSON 和 `review/*_overlay.png`；
3. 用浏览器打开 `tools/annotator.html`，载入 PNG + JSON 逐张校对（拖动引脚点/线段端点、改位号/型号、Shift 拖动重画 bbox），导出覆盖原 JSON；
4. `python tools/validate_dataset.py dataset/` 直到全部 OK（≥20 例，每例 ≥1 个 IC、≥5 元件、≥5 网络、IC 有效引脚数达标）；
5. `python tools/package_dataset.py dataset/ --team <队伍标识>` 生成提交包（含 dataset_description.docx）。

注意：抽检要求批次网络 F1 ≥ 95%、单例 ≥ 85%，**必须人工逐张核对**后再提交；11 月 2 日 23:00 前提交的数据有机会进入最终测试集。
