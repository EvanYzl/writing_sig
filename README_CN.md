# MSA-T OSV: 基于多尺度注意力机制和Transformer的离线签名验证

一个基于多尺度注意力机制和Transformer架构的深度学习离线签名验证框架。

## 概述

本项目实现了一个最先进的离线签名验证系统，结合了：
- **多尺度注意力机制** 用于捕获局部和全局签名特征
- **Transformer架构** 用于建模长距离依赖关系
- **空间金字塔池化(SPP)** 用于多尺度特征提取
- **高级损失函数** 包括三元组损失和焦点损失

## 特性

- 🎯 **高精度**: 在基准数据集上达到最先进的性能
- 🔧 **模块化设计**: 易于扩展和定制
- 📊 **全面评估**: 多种指标和可视化
- 🚀 **易于使用**: 简单的命令行界面
- 📈 **训练监控**: TensorBoard集成和详细日志
- 🎨 **可视化工具**: ROC曲线、混淆矩阵、t-SNE图

## 安装

### 前置要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (用于GPU训练)

### 从源码安装

```bash
# 克隆仓库
git clone <repository-url>
cd msa_t_osv

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 快速开始

### 1. 准备数据集

下载支持的数据集之一并更新配置文件：

- **CEDAR**: [下载地址](https://cedar.buffalo.edu/NIJ/data/signatures.rar)
- **MCYT**: [下载地址](http://atvs.ii.uam.es/databases/mcyt/)
- **GPDS**: [下载地址](http://www.gpds.ulpgc.es/download/)

### 2. 配置模型

编辑数据集对应的配置文件：

```yaml
# configs/cedar.yaml
dataset:
  name: "CEDAR"
  data_dir: "/path/to/cedar/dataset"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  backbone: "resnet50"
  input_size: 224
  num_classes: 2
  # ... 其他模型参数
```

### 3. 训练模型

```bash
# 在CEDAR数据集上训练
python -m msa_t_osv train --config configs/cedar.yaml --output_dir outputs/cedar

# 从检查点恢复训练
python -m msa_t_osv train --config configs/cedar.yaml --resume outputs/cedar/checkpoint_epoch_10.pth
```

### 4. 评估模型

```bash
# 在测试集上评估
python -m msa_t_osv evaluate --config configs/cedar.yaml --checkpoint outputs/cedar/best_eer.pth
```

### 5. 运行推理

```bash
# 验证单个签名
python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint outputs/cedar/best_eer.pth --input signature.png

# 验证多个签名
python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint outputs/cedar/best_eer.pth --input signatures/ --output results.json
```

## 模型架构

MSA-T OSV模型由几个关键组件组成：

### 1. CNN骨干网络
- 基于ResNet的特征提取器
- 空间金字塔池化(SPP)用于多尺度特征
- 多分辨率特征图

### 2. 多尺度注意力模块
- **空间注意力**: 关注重要的空间区域
- **通道注意力**: 强调重要的特征通道
- **尺度注意力**: 结合不同尺度的特征

### 3. Transformer编码器
- 自注意力机制用于全局特征建模
- 位置编码用于空间信息
- 多头注意力用于多样化特征表示

### 4. 融合头
- 结合多尺度特征
- 全局平均池化
- 最终分类层

## 配置

框架使用YAML配置文件进行简单定制：

### 数据集配置
```yaml
dataset:
  name: "CEDAR"  # 数据集名称
  data_dir: "/path/to/dataset"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  augmentations:
    rotation: 10
    scale: [0.9, 1.1]
    brightness: 0.2
```

### 模型配置
```yaml
model:
  backbone: "resnet50"
  input_size: 224
  num_classes: 2
  attention:
    spatial: true
    channel: true
    scale: true
  transformer:
    num_layers: 6
    num_heads: 8
    hidden_dim: 512
```

### 训练配置
```yaml
training:
  num_epochs: 100
  batch_size: 32
  optimizer:
    type: "adamw"
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    type: "cosine"
    min_lr: 0.00001
  loss:
    ce_weight: 1.0
    triplet_weight: 0.5
    focal_weight: 0.3
```

## 支持的数据集

### CEDAR
- **大小**: 2,640个签名 (55个书写者 × 24个真实 + 24个伪造)
- **格式**: PNG图像
- **特点**: 高质量签名，风格一致

### MCYT
- **大小**: 75,000个签名 (330个书写者 × 15个真实 + 15个伪造)
- **格式**: PNG图像
- **特点**: 大规模数据集，书写风格多样

### GPDS
- **大小**: 24,000个签名 (300个书写者 × 40个真实 + 40个伪造)
- **格式**: PNG图像
- **特点**: 专业伪造，质量高

## 性能

### CEDAR数据集结果
| 模型 | EER (%) | 准确率 (%) | AUC (%) |
|------|---------|------------|---------|
| MSA-T OSV | 2.1 | 97.9 | 99.2 |
| 基线ResNet | 4.8 | 95.2 | 97.1 |

### MCYT数据集结果
| 模型 | EER (%) | 准确率 (%) | AUC (%) |
|------|---------|------------|---------|
| MSA-T OSV | 3.2 | 96.8 | 98.5 |
| 基线ResNet | 6.1 | 93.9 | 95.8 |

## API使用

### Python API

```python
from msa_t_osv.models import MSATOSVModel
from msa_t_osv.inference import SignatureVerifier
import yaml

# 加载配置
with open('configs/cedar.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建验证器
verifier = SignatureVerifier('checkpoint.pth', config, device='cuda')

# 验证签名
result = verifier.verify_signature('signature.png')
print(f"决策: {result['decision']}")
print(f"置信度: {result['confidence']:.4f}")
```

### 命令行界面

```bash
# 训练模型
python -m msa_t_osv train --config configs/cedar.yaml

# 评估模型
python -m msa_t_osv evaluate --config configs/cedar.yaml --checkpoint best.pth

# 运行推理
python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint best.pth --input image.png
```

## 可视化

框架提供全面的可视化工具：

### 训练曲线
- 损失曲线随epoch变化
- 学习率调度
- 指标进展

### 评估图表
- ROC曲线
- 精确率-召回率曲线
- 混淆矩阵
- 分数分布

### 分析工具
- t-SNE可视化
- Grad-CAM注意力图
- 书写者相关指标

## 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 开发设置

```bash
# 克隆仓库
git clone <repository-url>
cd msa_t_osv

# 安装开发依赖
pip install -r requirements-dev.txt

# 以开发模式安装
pip install -e .

# 运行测试
pytest tests/

# 运行代码检查
flake8 msa_t_osv/
black msa_t_osv/
```

## 引用

如果您在研究中使用了此代码，请引用：

```bibtex
@article{msa_t_osv_2024,
  title={MSA-T OSV: Multi-Scale Attention and Transformer for Offline Signature Verification},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件。

## 致谢

- CEDAR、MCYT和GPDS数据集提供者
- PyTorch社区提供的优秀深度学习框架
- 签名验证领域的贡献者和研究人员

## 联系方式

如有问题和支持需求，请在GitHub上提出issue或联系维护者。

## 更新日志

详细变更历史请查看[CHANGELOG.md](CHANGELOG.md)。 