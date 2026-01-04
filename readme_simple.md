### 主要功能

1. **检测**：使用增强的 YOLOv8 架构检测空中视频中的小目标。
2. **预处理**：包括 CLAHE 和伽马校正等图像增强技术。
3. **域适应**：利用 CycleGAN 适应不同的照明条件。
4. **跟踪**：使用卡尔曼滤波进行多目标跟踪。
5. **行为分析**：使用 HMM 模型进行异常检测和威胁评估。
6. **可视化**：实时显示检测结果、轨迹、人群流动和威胁等级。
7. **指标**：提供检测、跟踪和行为分析的全面评估指标。

### 系统要求

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- NumPy, SciPy, scikit-learn
- 支持 CUDA 的 GPU（推荐）

### 安装步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/yourusername/aerial-crowd-analysis.git
   cd aerial-crowd-analysis
   ```

2. **创建并激活虚拟环境**：
   ```bash
   conda create -n aerial-crowd python=3.8
   conda activate aerial-crowd
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **设置 GPU 加速**（如果适用）：
   ```bash
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

5. **下载预训练模型**：
   ```bash
   python scripts/download_models.py --all
   ```

### 使用方法

#### 推理

在视频或图像序列上运行推理：

```bash
python main.py --mode inference --source path/to/video.mp4 --output results/output.mp4
```

- 使用 `--conf-thres` 和 `--iou-thres` 调整检测参数。
- 使用 `--use-domain-adaptation` 启用域适应。

#### 训练

训练模型：

- **检测模型**：
  ```bash
  python main.py --mode train --train-type detection
  ```

- **CycleGAN 模型**：
  ```bash
  python main.py --mode train --train-type cyclegan
  ```

- **HMM 模型**：
  ```bash
  python main.py --mode train --train-type hmm
  ```

#### 评估

评估系统性能：

```bash
python main.py --mode test --source datasets/test/ --output evaluation_results/
```

### 配置

系统使用 YAML 配置文件（`config.yaml`）管理参数。您可以通过复制和编辑此文件创建自定义配置：

```bash
cp config.yaml myconfig.yaml
```

### 高级选项

- **实时处理**：使用 `--source 0` 进行实时摄像头输入。
- **批量处理**：使用 `--mode batch-inference` 处理多个文件。
- **可视化**：通过命令行标志启用或禁用可视化功能。

### 常见问题

- **检测效果不佳**：调整阈值或使用更大的模型。
- **跟踪不稳定**：修改 `max_age` 和 `min_hits` 参数。
- **行为分析误报**：如有需要，重新训练 HMM 模型。

