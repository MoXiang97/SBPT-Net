# SBPT-Net 开源代码

本仓库对应论文：

**Superline-Based Synthetic-to-Real Domain Generalization for Gauge Line Segmentation in Shoe Upper Point Clouds**

当前版本包含与论文现有网络结构一致的预处理、训练、评估和推理代码。数据、实验目录、预测结果和模型权重不直接提交到 Git 仓库。为兼容已有脚本，Python 包名仍为 `s2rspc`。

## 当前方法

1. 使用局部颜色对比保留高召回率候选点。
2. 使用尺度归一化图构造 superlines；边排序采用论文公式（2）中的符号不变正交切向残差。
3. 从局部支持区域选择观测代表点，并保存重叠的 token 支持成员。
4. PointMLP 以中心化 token 坐标和平均 RGB（6D）预测初始 logit。
5. 中心 token 的 6 个结构描述子和 6 个归一化属性组成 12D 输入，由结构编码器预测结构 logit。
6. 按论文中的固定规则融合两个 logit。回投时，对覆盖同一候选点的 token 概率取平均，再按原始点索引散射到完整点云。

当前网络不包含上下文 token 聚合模块。

## 安装与检查

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
pytest
python tools/verify_release.py
```

无需数据的基础检查：

```bash
sbptnet-toy-demo --config configs/sbptnet_sts2r.yaml --output toy_output
```

该输出只用于软件检查，不是论文实验结果。

## 数据与训练

原始点云使用七列文本格式：

```text
X Y Z R G B Label
```

STS2R 数据集 v1 发布于 [Zenodo](https://doi.org/10.5281/zenodo.19528228)，合成数据生成代码见 [MoXiang97/STS2R-code](https://github.com/MoXiang97/STS2R-code)。

```bash
sbptnet-preprocess --config configs/sbptnet_sts2r.yaml --input-root data/raw
sbptnet-train --config configs/sbptnet_sts2r.yaml --token-root data/tokens --output runs/sbptnet_seed42
```

训练脚本只读取合成训练集与合成验证集：先训练 PointMLP，再冻结主干并训练 12D 结构编码器；checkpoint 和阈值均由合成验证集选择。模型结构的开发阶段曾参考较早的真实集评估结果，因此真实扫描不应被表述为与模型开发完全独立的未触碰测试集。

详细说明见：

- [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)
- [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
- [docs/REPRODUCTION.md](docs/REPRODUCTION.md)

仓库不包含完整实验结果归档或暂定论文结果表；正式结果应与冻结 checkpoint、运行清单和完成的验证记录一起发布。
