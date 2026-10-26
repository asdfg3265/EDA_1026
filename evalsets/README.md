# 评测集（开发集、留出集、带图框集、多风格集）

这些都是本队用数据引擎自建的测试集，用来衡量隐藏题上的泛化能力。全部来自 oshwhub.com 上公开源文件的嘉立创专业版工程：图片由 `easypro/render.py` 从源文件渲染，标准答案由 `easypro/gt.py` 从同一份源文件计算，格式与赛题 JSON 相同（左下角原点、像素坐标）。

| 目录 | 例数 | 是什么 | 用途 |
|---|---|---|---|
| `dev_99/` | 99 | 开发集：73 个工程里抽的原理图页 | 调参、逐例排查失分 |
| `holdout_184/` | 184 | 留出集：133 个工程，和开发集基本不重叠（只有 3 个工程两边都有） | **从不用于调参**，只用来确认改进是通用的 |
| `framed_dev_50/` | 50 | 开发集中 50 页，按 A4/A3 加图框和标题栏重新渲染（答案不变） | 检验图框去除 |
| `framed_holdout_70/` | 70 | 留出集中 70 页，同上 | 同上 |
| `style_altium_8/`、`style_kicad_8/` | 各 8 | 嘉立创图重新上色成 Altium、KiCad 的默认配色 | 检验其他工具风格 |

每个目录里是成对的 `case_XXXX.png` 和 `case_XXXX.json`；`cases.txt` 列出每例来自哪个工程、哪一页。

标注版本：开发集、留出集为第 9 版标注（g9）。与 10-26 上传的数据集同一套生成规则，只有一处差别：数据集已用第 12 轮的 gt.py 重算（未标类型的 Q 管按引脚名区分三极管和 MOSFET），开发集因此有 4 个 Q 不同。

## 打分
```bash
cd code/PCBParse
python run.py ../../evalsets/dev_99 /tmp/pred_dev
PCB_SCORE_RULE=platform python -m pcbparse.scorer_official ../../evalsets/dev_99 /tmp/pred_dev   # 平台现行规则
python -m pcbparse.scorer_official ../../evalsets/dev_99 /tmp/pred_dev                            # 10-22 书面规则
```

当前 v16 成绩（平台规则 / 10-22 规则）：开发集 90.14 / 92.27，留出集 89.02 / 91.13，带图框开发 88.15 / 90.11，带图框留出 87.67 / 89.31，Altium / KiCad（10-22 规则）82.69 / 85.52。

注意：这些案例与上传的附加分数据集来自同一批工程，**不要把它们再单独提交**作为数据贡献。
