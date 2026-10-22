# 论文图表

本目录由`scripts/generate_paper_figures.py`从冻结JSON/CSV确定性生成，不访问模型权重、数据集或test-dev。每组图同时提供300 DPI PNG和矢量PDF：

- `bdd-size-ap-comparison`：BDD100K原生YOLO26n、官方P2和P2通道注意力的尺寸AP。
- `visdrone-training-curves`：VisDrone原生YOLO26n三个随机种子的100轮val mAP50-95曲线。
- `visdrone-multiseed-size-errorbars`：三个随机种子的尺寸AP/AR均值及样本标准差。
- `visdrone-val-test-size-gap`：冻结seed0在val与唯一一次test-dev评估间的尺寸AP差距。
- `visdrone-official-class-gap`：官方兼容口径下val与test-dev的分类别AP差距。

`manifest.json`记录生成脚本所读输入及全部图文件的SHA256。复现命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-paper \
PYTHONPATH=upstream:. \
/home/<user>/miniconda3/envs/yolo26-sdf/bin/python \
scripts/generate_paper_figures.py --output results/paper_figures
```
