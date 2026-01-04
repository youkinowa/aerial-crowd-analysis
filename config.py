import os
import yaml
import json
import argparse
from pathlib import Path

class Config:
    """空中人群分析系统的配置管理器"""
    
    def __init__(self, config_path=None):
        """
        初始化配置
        
        参数:
            config_path: 配置文件路径 (YAML 或 JSON)
        """
        # 默认配置
        self.config = {
            # 系统路径
            'paths': {
                'model_dir': 'models/weights',
                'data_dir': 'data',
                'output_dir': 'output',
                'log_dir': 'logs',
            },
            
            # 检测模型配置
            'detection': {
                'model': 'yolov8n.pt',  # 基础 YOLOv8 模型
                'backbone': 'enhanced',  # 'original' 或 'enhanced'
                'device': 'cuda:0',      # 'cuda:0', 'cpu'
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'img_size': 640,
                'max_detections': 300,
                'class_agnostic_nms': False,
                'augmented_inference': True,  # 使用 TTA
                'preprocessing': {
                    'enable_clahe': True,
                    'clahe_clip_limit': 2.0,
                    'clahe_tile_grid_size': 8,
                    'enable_gamma': True,
                    'gamma_value': 1.2
                }
            },
            
            # 域适应配置
            'domain_adaptation': {
                'enabled': True,
                'cyclegan_weights': 'models/weights/cyclegan_lighting.pt',
                'adaptation_type': 'lighting',  # 'lighting', 'weather' 等
                'input_size': 512,
                'batch_size': 1,
                'normalize_means': [0.5, 0.5, 0.5],
                'normalize_stds': [0.5, 0.5, 0.5]
            },
            
            # 跟踪配置
            'tracking': {
                'method': 'iou',         # 'iou', 'deep_sort' 等
                'max_age': 30,           # 最大跟踪帧数
                'min_hits': 3,           # 建立跟踪的最小命中数
                'iou_threshold': 0.3,    # 匹配的 IoU 阈值
                'use_appearance': False, # 使用外观特征
                'use_kalman': True       # 使用卡尔曼滤波
            },
            
            # HMM 行为分析配置
            'behavior': {
                'hmm_states': 5,         # HMM 状态数
                'hmm_init_type': 'kmeans', # 'random', 'kmeans'
                'min_trajectory_length': 10, # 分析的最小长度
                'features': [            # 要提取的特征
                    'velocity', 'acceleration', 'direction', 'curvature',
                    'stop_ratio', 'trajectory_length'
                ],
                'window_size': 10,       # 特征提取的窗口大小
                'anomaly_threshold': 0.7, # 对数似然阈值
                'threat_levels': {
                    'low': 0.3,          # 低威胁阈值
                    'medium': 0.6,       # 中威胁阈值
                    'high': 0.8          # 高威胁阈值
                }
            },
            
            # 可视化配置
            'visualization': {
                'show_boxes': True,      # 显示边界框
                'show_labels': True,     # 显示标签
                'show_scores': True,     # 显示置信度分数
                'show_tracks': True,     # 显示跟踪 ID
                'show_trajectories': True, # 显示轨迹
                'trajectory_length': 30, # 最大轨迹长度
                'density_map': True,     # 显示密度图
                'crowd_flow': True,      # 显示人群流动
                'threat_assessment': True, # 显示威胁评估
                'save_frames': False,    # 保存帧到磁盘
                'save_video': True,      # 保存视频到磁盘
                'font_scale': 0.5,       # 字体缩放
                'line_thickness': 2,     # 线条粗细
                'colors': {
                    'box': (0, 255, 0),  # 绿色框
                    'track_id': (255, 255, 255), # 白色跟踪 ID
                    'trajectory': (0, 255, 255), # 黄色轨迹
                    'alert_low': (0, 255, 0),    # 绿色警报
                    'alert_medium': (0, 165, 255), # 橙色警报
                    'alert_high': (0, 0, 255)    # 红色警报
                }
            },
            
            # 推理设置
            'inference': {
                'source': None,         # 视频或文件夹路径
                'output_path': None,    # 保存输出的路径
                'batch_size': 1,        # 推理的批量大小
                'save_results': True,   # 保存检测结果
                'fps_limit': 0,         # FPS 限制 (0 = 无限制)
                'real_time': False,     # 实时模式 (跳过帧)
                'frame_skip': 0,        # 跳过的帧数
                'start_frame': 0,       # 起始帧
                'end_frame': -1,        # 结束帧 (-1 = 所有帧)
                'show_fps': True,       # 显示 FPS 计数器
                'quiet': False          # 抑制输出
            },
            
            # 训练设置
            'training': {
                # 通用
                'seed': 42,
                'epochs': 100,
                'batch_size': 16,
                'num_workers': 4,
                'pin_memory': True,
                'mixed_precision': True,
                
                # 检测模型
                'detection': {
                    'learning_rate': 0.01,
                    'weight_decay': 0.0005,
                    'momentum': 0.937,
                    'warmup_epochs': 3,
                    'warmup_momentum': 0.8,
                    'warmup_bias_lr': 0.1,
                    'pretrained': True,
                    'resume': False,
                    'image_size': 640,
                    'augmentation': {
                        'hsv_h': 0.015,  # HSV 色调增强
                        'hsv_s': 0.7,    # HSV 饱和度增强
                        'hsv_v': 0.4,    # HSV 值增强
                        'degrees': 0.0,  # 旋转
                        'translate': 0.1, # 平移
                        'scale': 0.5,    # 缩放
                        'shear': 0.0,    # 剪切
                        'perspective': 0.0, # 透视
                        'flipud': 0.0,   # 上下翻转
                        'fliplr': 0.5,   # 左右翻转
                        'mosaic': 1.0,   # 马赛克增强
                        'mixup': 0.0,    # 混合增强
                        'copy_paste': 0.0 # 复制粘贴增强
                    }
                },
                
                # CycleGAN
                'cyclegan': {
                    'learning_rate': 0.0002,
                    'beta1': 0.5,        # Adam beta1
                    'beta2': 0.999,      # Adam beta2
                    'lambda_A': 10.0,    # 循环损失的权重 (A -> B -> A)
                    'lambda_B': 10.0,    # 循环损失的权重 (B -> A -> B)
                    'lambda_identity': 0.5, # 身份损失的权重
                    'pool_size': 50,     # 图像池大小
                    'epochs': 100,
                    'decay_epoch': 50,   # 开始线性学习率衰减的 epoch
                    'image_size': 256,
                    'input_nc': 3,       # 输入通道数
                    'output_nc': 3,      # 输出通道数
                    'gan_mode': 'lsgan', # GAN模式：'vanilla', 'lsgan', 'wgangp'
                    'netG': 'resnet_9blocks', # 生成器网络：'resnet_9blocks', 'resnet_6blocks', 'unet_256'
                    'netD': 'basic',     # 判别器网络：'basic', 'n_layers', 'pixel'
                    'n_layers_D': 3,     # 如果判别器网络是'n_layers'时使用的层数
                    'n_layers_D': 3,     # 如果判别器网络是'n_layers'时使用的层数
                    'norm': 'instance',  # 归一化方式：'batch', 'instance', 'none'
                    'init_type': 'normal', # 权重初始化方式：'normal', 'xavier', 'kaiming', 'orthogonal'
                    'init_gain': 0.02,   # 权重初始化的缩放因子，适用于normal、xavier和orthogonal初始化
                    'no_dropout': False  # 生成器是否不使用dropout
                },
                
                # HMM
                'hmm': {
                    'n_iter': 100,       # EM算法的迭代次数
                    'tol': 1e-6,         # 收敛阈值
                    'n_init': 3,         # 初始化次数
                    'algorithm': 'viterbi', # 算法选择：'viterbi', 'map'
                    'covariance_type': 'full', # 协方差类型：'spherical', 'diag', 'full', 'tied'
                    'validation_split': 0.2, # 验证集划分比例
                    'test_split': 0.1    # 测试集划分比例
                }
            }
        }
        
        # 如果提供了配置文件路径，则加载配置
        if config_path is not None:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        从文件加载配置
        
        参数:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        # 根据文件扩展名加载配置
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 使用加载的值更新配置
        self._update_config(self.config, loaded_config)
        
        # 创建必要的目录
        self._create_directories()
    
    def _update_config(self, config, update):
        """
        递归更新嵌套字典
        
        参数:
            config: 要更新的配置字典
            update: 包含更新值的字典
        """
        for key, value in update.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def _create_directories(self):
        """如果目录不存在，则创建必要的目录"""
        for path_key, path in self.config['paths'].items():
            os.makedirs(path, exist_ok=True)
    
    def save_config(self, config_path):
        """
        将配置保存到文件
        
        参数:
            config_path: 保存配置文件的路径
        """
        config_path = Path(config_path)
        
        # 如果父目录不存在，则创建
        os.makedirs(config_path.parent, exist_ok=True)
        
        # 根据文件扩展名保存
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def get(self, key, default=None):
        """
        通过键获取配置值
        
        参数:
            key: 配置键（可以使用点号表示嵌套，例如'detection.model'）
            default: 如果未找到键时的默认值
            
        返回:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """
        通过键设置配置值
        
        参数:
            key: 配置键（可以使用点号表示嵌套，例如'detection.model'）
            value: 要设置的值
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

def parse_args():
    """解析应用程序的命令行参数"""
    parser = argparse.ArgumentParser(description='空中人群分析系统')
    
    # 通用参数
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'inference'],
                        default='inference', help='操作模式')
    
    # 输入参数
    parser.add_argument('--source', type=str, help='输入视频或文件夹路径')
    parser.add_argument('--output', type=str, help='输出目录路径')
    
    # 模型参数
    parser.add_argument('--model', type=str, help='检测模型路径')
    parser.add_argument('--backbone', type=str, choices=['original', 'enhanced'],
                        help='骨干网络架构')
    parser.add_argument('--device', type=str, help='运行设备 (cuda:0, cpu)')
    
    # 检测参数
    parser.add_argument('--conf-thresh', type=float, help='置信度阈值')
    parser.add_argument('--iou-thresh', type=float, help='IoU阈值')
    parser.add_argument('--img-size', type=int, help='输入图像大小')
    
    # 预处理参数
    parser.add_argument('--enable-clahe', action='store_true', help='启用CLAHE预处理')
    parser.add_argument('--disable-clahe', action='store_false', dest='enable_clahe', help='禁用CLAHE预处理')
    parser.add_argument('--enable-gamma', action='store_true', help='启用gamma预处理')
    parser.add_argument('--disable-gamma', action='store_false', dest='enable_gamma', help='禁用gamma预处理')
    
    # 域适应参数
    parser.add_argument('--enable-adaptation', action='store_true', help='启用域适应')
    parser.add_argument('--disable-adaptation', action='store_false', dest='enable_adaptation', help='禁用域适应')
    
    # 跟踪参数
    parser.add_argument('--tracking-method', type=str, help='跟踪方法')
    parser.add_argument('--max-age', type=int, help='轨迹的最大年龄')
    parser.add_argument('--min-hits', type=int, help='建立轨迹的最小命中次数')
    
    # 行为分析参数
    parser.add_argument('--hmm-states', type=int, help='HMM状态数')
    parser.add_argument('--anomaly-thresh', type=float, help='异常检测阈值')
    
    # 可视化参数
    parser.add_argument('--show-boxes', action='store_true', help='显示边界框')
    parser.add_argument('--hide-boxes', action='store_false', dest='show_boxes', help='隐藏边界框')
    parser.add_argument('--show-tracks', action='store_true', help='显示轨迹ID')
    parser.add_argument('--hide-tracks', action='store_false', dest='show_tracks', help='隐藏轨迹ID')
    parser.add_argument('--show-trajectories', action='store_true', help='显示轨迹')
    parser.add_argument('--hide-trajectories', action='store_false', dest='show_trajectories', help='隐藏轨迹')
    parser.add_argument('--save-frames', action='store_true', help='保存帧')
    parser.add_argument('--save-video', action='store_true', help='保存视频')
    
    # 推理参数
    parser.add_argument('--batch-size', type=int, help='推理的批量大小')
    parser.add_argument('--fps-limit', type=int, help='FPS限制 (0 = 无限制)')
    parser.add_argument('--real-time', action='store_true', help='实时模式（跳过帧）')
    # 检测参数
    parser.add_argument('--conf-thresh', type=float, help='置信度阈值')
    parser.add_argument('--iou-thresh', type=float, help='IoU阈值')
    parser.add_argument('--img-size', type=int, help='输入图像大小')
    
    # 预处理参数
    parser.add_argument('--enable-clahe', action='store_true', help='启用CLAHE预处理')
    parser.add_argument('--disable-clahe', action='store_false', dest='enable_clahe', help='禁用CLAHE预处理')
    parser.add_argument('--enable-gamma', action='store_true', help='启用gamma预处理')
    parser.add_argument('--disable-gamma', action='store_false', dest='enable_gamma', help='禁用gamma预处理')
    
    # 域适应参数
    parser.add_argument('--enable-adaptation', action='store_true', help='启用域适应')
    parser.add_argument('--disable-adaptation', action='store_false', dest='enable_adaptation', help='禁用域适应')
    
    # 跟踪参数
    parser.add_argument('--tracking-method', type=str, help='跟踪方法')
    parser.add_argument('--max-age', type=int, help='轨迹的最大年龄')
    parser.add_argument('--min-hits', type=int, help='建立轨迹的最小命中次数')
    
    # 行为分析参数
    parser.add_argument('--hmm-states', type=int, help='HMM状态数')
    parser.add_argument('--anomaly-thresh', type=float, help='异常检测阈值')
    
    # 可视化参数
    parser.add_argument('--show-boxes', action='store_true', help='显示边界框')
    parser.add_argument('--hide-boxes', action='store_false', dest='show_boxes', help='隐藏边界框')
    parser.add_argument('--show-tracks', action='store_true', help='显示轨迹ID')
    parser.add_argument('--hide-tracks', action='store_false', dest='show_tracks', help='隐藏轨迹ID')
    parser.add_argument('--show-trajectories', action='store_true', help='显示轨迹')
    parser.add_argument('--hide-trajectories', action='store_false', dest='show_trajectories', help='隐藏轨迹')
    parser.add_argument('--save-frames', action='store_true', help='保存帧')
    parser.add_argument('--save-video', action='store_true', help='保存视频')
    
    # 推理参数
    parser.add_argument('--batch-size', type=int, help='推理的批量大小')
    parser.add_argument('--fps-limit', type=int, help='FPS限制 (0 = 无限制)')
    parser.add_argument('--real-time', action='store_true', help='实时模式（跳过帧）')
    parser.add_argument('--frame-skip', type=int, help='跳过的帧数')
    parser.add_argument('--start-frame', type=int, help='起始帧')
    parser.add_argument('--end-frame', type=int, help='结束帧 (-1 = 所有帧)')
    parser.add_argument('--quiet', action='store_true', help='抑制输出')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, help='训练epoch数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--train-batch-size', type=int, help='训练批量大小')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """
    从命令行参数更新配置
    
    参数:
        config: 配置对象
        args: 解析后的命令行参数
    """
    # 只更新非None的值
    arg_dict = vars(args)
    
    # 将参数名映射到配置键
    arg_to_config = {
        'source': 'inference.source',
        'output': 'inference.output_path',
        'model': 'detection.model',
        'backbone': 'detection.backbone',
        'device': 'detection.device',
        'conf_thresh': 'detection.conf_threshold',
        'iou_thresh': 'detection.iou_threshold',
        'img_size': 'detection.img_size',
        'enable_clahe': 'detection.preprocessing.enable_clahe',
        'enable_gamma': 'detection.preprocessing.enable_gamma',
        'enable_adaptation': 'domain_adaptation.enabled',
        'tracking_method': 'tracking.method',
        'max_age': 'tracking.max_age',
        'min_hits': 'tracking.min_hits',
        'hmm_states': 'behavior.hmm_states',
        'anomaly_thresh': 'behavior.anomaly_threshold',
        'show_boxes': 'visualization.show_boxes',
        'show_tracks': 'visualization.show_tracks',
        'show_trajectories': 'visualization.show_trajectories',
        'save_frames': 'visualization.save_frames',
        'save_video': 'visualization.save_video',
        'batch_size': 'inference.batch_size',
        'fps_limit': 'inference.fps_limit',
        'real_time': 'inference.real_time',
        'frame_skip': 'inference.frame_skip',
        'start_frame': 'inference.start_frame',
        'end_frame': 'inference.end_frame',
        'quiet': 'inference.quiet',
        'epochs': 'training.epochs',
        'lr': 'training.detection.learning_rate',
        'train_batch_size': 'training.batch_size',
        'resume': 'training.detection.resume',
        'seed': 'training.seed'
    }
    
    # 使用参数值更新配置
    for arg_name, config_key in arg_to_config.items():
        value = arg_dict.get(arg_name)
        if value is not None:
            config.set(config_key, value)
    
    # 更新模式
    if args.mode is not None:
        config.set('mode', args.mode) 