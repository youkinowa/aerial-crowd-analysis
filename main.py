#!/usr/bin/env python3
"""
空中人群分析系统 - 主入口

该系统结合了增强的YOLOv8目标检测和基于隐马尔可夫模型的人群行为分析，
用于无人机航拍视频分析。
"""

import os
import sys
import time
import logging
import torch
from pathlib import Path

# 导入项目组件
from config import Config, parse_args, update_config_from_args
from inference import AerialCrowdAnalysisPipeline
from train import DetectionModelTrainer, CycleGANTrainer, HMMTrainer
from utils.metrics import DetectionEvaluator, TrackingEvaluator, HMMEvaluator, PerformanceMonitor
from models.behavior.hmm_model import CrowdBehaviorHMM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/aerial_crowd_analysis.log', mode='a')
    ]
)

logger = logging.getLogger('AerialCrowdAnalysis')

def run_inference(config):
    """
    在视频或图像序列上运行推理
    
    参数:
        config: 配置对象
    """
    # 检查是否提供了源
    source = config.get('inference.source')
    if source is None:
        logger.error("未提供推理源。请使用--source参数或在配置中设置。")
        return False
    
    # 创建处理管道
    pipeline = AerialCrowdAnalysisPipeline(config)
    
    # 在源上运行
    pipeline.run(source)
    
    return True

def run_training(config):
    """
    运行模型训练
    
    参数:
        config: 配置对象
    """
    # 设置随机种子以保证可重复性
    seed = config.get('training.seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 设置模式特定的日志
    train_log_path = Path(config.get('paths.log_dir')) / 'training.log'
    train_log_handler = logging.FileHandler(train_log_path, mode='a')
    train_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(train_log_handler)
    
    logger.info(f"开始训练，随机种子为 {seed}")
    
    # 确定要训练的内容
    train_detection = True  # 默认训练检测模型
    train_cyclegan = True   # 默认训练CycleGAN
    train_hmm = True        # 默认训练HMM
    
    # TODO: 允许通过配置或参数选择特定模型进行训练
    
    # 训练检测模型
    if train_detection:
        logger.info("训练检测模型中...")
        detection_trainer = DetectionModelTrainer(config)
        detection_trainer.train()
        
        # 导出训练好的模型
        model_path = detection_trainer.export_model()
        logger.info(f"检测模型已保存至 {model_path}")
    
    # 训练CycleGAN用于域适应
    if train_cyclegan:
        logger.info("训练CycleGAN用于域适应...")
        cyclegan_trainer = CycleGANTrainer(config)
        cyclegan_trainer.train()
        
        # 导出训练好的模型
        model_path = cyclegan_trainer.export_model()
        logger.info(f"CycleGAN模型已保存至 {model_path}")
    
    # 训练HMM用于行为分析
    if train_hmm:
        logger.info("训练HMM用于行为分析...")
        hmm_trainer = HMMTrainer(config)
        hmm_trainer.train()
        
        # 导出训练好的模型
        model_path = hmm_trainer.export_model()
        logger.info(f"HMM模型已保存至 {model_path}")
    
    logger.info("训练成功完成")
    return True

def run_evaluation(config):
    """
    在测试数据集上运行评估
    
    参数:
        config: 配置对象
    """
    # 设置评估特定的日志
    eval_log_path = Path(config.get('paths.log_dir')) / 'evaluation.log'
    eval_log_handler = logging.FileHandler(eval_log_path, mode='a')
    eval_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(eval_log_handler)
    
    logger.info("开始评估")
    
    # 创建评估管道（类似于推理但带有真实值）
    pipeline = AerialCrowdAnalysisPipeline(config)
    
    # TODO: 实现带有真实值比较的评估管道
    # 应该：
    # 1. 加载带有真实值标注的测试数据集
    # 2. 在测试数据集上运行推理
    # 3. 将结果与真实值进行比较
    # 4. 计算检测、跟踪和行为分析的指标
    
    # 创建评估器
    detection_evaluator = DetectionEvaluator()
    tracking_evaluator = TrackingEvaluator()
    hmm_evaluator = HMMEvaluator()
    performance_monitor = PerformanceMonitor()
    
    # 运行推理和评估
    test_source = config.get('inference.source')  # 应指向测试数据集
    if test_source is None:
        logger.error("未提供测试源进行评估")
        return False
    
    # 在测试源上运行并收集指标
    pipeline.run_evaluation(
        test_source, 
        detection_evaluator=detection_evaluator,
        tracking_evaluator=tracking_evaluator,
        hmm_evaluator=hmm_evaluator,
        performance_monitor=performance_monitor
    )
    
    # 保存结果
    output_dir = Path(config.get('paths.output_dir')) / 'evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    detection_results = detection_evaluator.save_results(output_dir / 'detection_results.json')
    hmm_results = hmm_evaluator.save_results(output_dir / 'hmm_results.json')
    performance_results = performance_monitor.save_results(output_dir / 'performance_results.json')
    
    # 记录摘要
    logger.info("评估完成。结果：")
    logger.info(f"检测mAP: {detection_evaluator.stats['mean_iou']:.4f}")
    logger.info(f"检测F1: {detection_evaluator.stats['f1_score']:.4f}")
    logger.info(f"HMM状态准确率: {hmm_evaluator.stats['state_accuracy']:.4f}")
    logger.info(f"平均FPS: {performance_monitor.stats['avg_fps']:.2f}")
    
    return True

def main():
    """应用程序主入口"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = Config(args.config if hasattr(args, 'config') else None)
    
    # 从命令行参数更新配置
    update_config_from_args(config, args)
    
    # 创建所需目录
    for path in config.get('paths').values():
        os.makedirs(path, exist_ok=True)
    
    # 获取操作模式
    mode = args.mode if hasattr(args, 'mode') else 'inference'
    
    # 在指定模式下运行
    if mode == 'inference':
        success = run_inference(config)
    elif mode == 'train':
        success = run_training(config)
    elif mode == 'test':
        success = run_evaluation(config)
    else:
        logger.error(f"未知操作模式: {mode}")
        success = False
    
    # 返回退出码
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 