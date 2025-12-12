#!/usr/bin/env python3
"""
独立的metrics计算脚本
包含完整的4种评估指标：ROUGE1, ROUGEL, BLEU, BERTScore
"""
import os
import sys
import json
import argparse
import logging
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# 直接导入所需的包
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_next_metrics_version(output_dir, base_name="open_metrics"):
    """
    获取下一个可用的metrics文件版本号
    """
    # 首先检查是否存在 open_metrics.json（视为v0）
    open_metrics_path = os.path.join(output_dir, "open_metrics.json")
    if os.path.exists(open_metrics_path):
        # 如果open_metrics.json存在，将其重命名为open_metrics_v0.json
        new_path = os.path.join(output_dir, "open_metrics_v0.json")
        try:
            os.rename(open_metrics_path, new_path)
            logger.info(f"Renamed {open_metrics_path} to {new_path}")
        except Exception as e:
            logger.warning(f"Failed to rename {open_metrics_path} to {new_path}: {e}")
    
    # 查找所有已存在的版本文件
    existing_versions = []
    pattern = re.compile(rf"{base_name}_v(\d+)\.json")
    
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            match = pattern.match(filename)
            if match:
                version = int(match.group(1))
                existing_versions.append(version)
    
    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        # 检查是否存在open_metrics_v0.json
        v0_path = os.path.join(output_dir, "open_metrics_v0.json")
        if os.path.exists(v0_path):
            next_version = 1
        else:
            next_version = 0
    
    return f"{base_name}_v{next_version}.json"

def check_metrics_exist(output_dir, base_name="open_metrics"):
    """
    检查是否已经存在处理过的结果
    返回：是否存在，以及最新的metrics文件路径
    """
    if not os.path.exists(output_dir):
        return False, None
    
    # 首先检查是否存在 open_metrics.json（视为v0）
    open_metrics_path = os.path.join(output_dir, "open_metrics.json")
    if os.path.exists(open_metrics_path):
        return True, open_metrics_path
    
    # 查找所有已存在的版本文件
    existing_files = []
    pattern = re.compile(rf"{base_name}_v(\d+)\.json")
    
    for filename in os.listdir(output_dir):
        if pattern.match(filename):
            existing_files.append(filename)
    
    if not existing_files:
        return False, None
    
    # 找到版本号最大的文件
    latest_file = None
    latest_version = -1
    
    for filename in existing_files:
        match = pattern.match(filename)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version
                latest_file = filename
    
    if latest_file:
        latest_path = os.path.join(output_dir, latest_file)
        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)
            
            # 检查必要的字段是否存在
            required_fields = ['total_samples', 'accuracy', 'metrics_summary', 'five_metrics']
            for field in required_fields:
                if field not in existing_metrics:
                    logger.warning(f"Existing metrics file missing field: {field}")
                    return False, None
            
            # 检查metrics中的关键指标
            five_metrics = existing_metrics.get('five_metrics', {})
            required_metrics = ['rouge1', 'rougeL', 'bleu', 'bertscore']
            for metric in required_metrics:
                if metric not in five_metrics:
                    logger.warning(f"Existing metrics file missing metric: {metric}")
                    return False, None
            
            logger.info(f"Valid metrics file already exists: {latest_path}")
            return True, latest_path
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to read existing metrics file: {e}")
            return False, None
    
    return False, None

def initialize_evaluators():
    """初始化各种评估器"""
    evaluators = {}
    
    # ROUGE评估器
    try:
        evaluators['rouge'] = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    except Exception as e:
        logger.warning(f"Failed to initialize ROUGE: {e}")
    
    # BERTScore评估器
    try:
        evaluators['bertscore'] = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', rescale_with_baseline=True)
    except Exception as e:
        logger.warning(f"Failed to initialize BERTScore: {e}")
    
    return evaluators

def compute_rouge_metrics(reference, hypothesis, evaluators):
    """计算ROUGE1和ROUGEL指标"""
    if 'rouge' not in evaluators:
        return {'rouge1': 0.0, 'rougeL': 0.0}
    
    try:
        scores = evaluators['rouge'].score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {'rouge1': 0.0, 'rougeL': 0.0}

def compute_bleu_metrics(reference, hypothesis):
    """计算BLEU指标 - 改进版本"""
    try:
        ref_tokens = [reference.split()]
        hyp_tokens = hypothesis.split()
        
        if len(hyp_tokens) == 0 or len(ref_tokens[0]) == 0:
            return {'bleu': 0.0}
        
        # 根据文本长度选择合适的权重
        if len(hyp_tokens) <= 2:
            weights = (1.0, 0, 0, 0)  # 只使用1-gram
        elif len(hyp_tokens) <= 4:
            weights = (0.5, 0.5, 0, 0)  # 使用1-gram和2-gram
        else:
            weights = (0.25, 0.25, 0.25, 0.25)  # 标准4-gram
        
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu(ref_tokens, hyp_tokens, 
                                  weights=weights, 
                                  smoothing_function=smoothie)
        
        return {'bleu': min(bleu_score, 1.0)}  # 确保不超过1.0
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return {'bleu': 0.0}

def compute_bertscore_metrics(reference, hypothesis, evaluators):
    """计算BERTScore指标"""
    if 'bertscore' not in evaluators:
        return {'bertscore': 0.0}
    
    try:
        P, R, F1 = evaluators['bertscore'].score([hypothesis], [reference])
        return {'bertscore': F1.item()}  # 使用F1分数
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return {'bertscore': 0.0}

def simple_extract(text, tag):
    """简单的提取函数"""
    text = text.strip()
    
    # 尝试提取<tag>...</tag>格式
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    if start_tag in text and end_tag in text:
        start_idx = text.find(start_tag) + len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if start_idx >= 0 and end_idx >= 0:
            return text[start_idx:end_idx].strip()
    
    return text

def compute_exact_match(reference, hypothesis):
    """计算精确匹配"""
    return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0

def comprehensive_text_evaluation(reference, hypothesis, evaluators):
    """综合文本评估 - 4种指标"""
    metrics = {}
    
    # 基础指标
    metrics['exact_match'] = compute_exact_match(reference, hypothesis)
    
    # 4种要求的评估指标
    # 1-2. ROUGE指标
    rouge_metrics = compute_rouge_metrics(reference, hypothesis, evaluators)
    metrics.update(rouge_metrics)
    
    # 3. BLEU指标
    bleu_metrics = compute_bleu_metrics(reference, hypothesis)
    metrics.update(bleu_metrics)
    
    # 4. BERTScore指标
    bertscore_metrics = compute_bertscore_metrics(reference, hypothesis, evaluators)
    metrics.update(bertscore_metrics)
    
    return metrics

def compute_statistics(scores):
    """计算统计信息：最大值、最小值、平均值、标准差"""
    if not scores:
        return {
            'max': 0.0,
            'min': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'count': 0
        }
    
    scores_array = np.array(scores)
    return {
        'max': float(np.max(scores_array)),
        'min': float(np.min(scores_array)),
        'mean': float(np.mean(scores_array)),
        'std': float(np.std(scores_array)),
        'count': len(scores)
    }

def recompute_omnibrain_metrics(results_path, output_path):
    """
    重新计算OmniBrainBench的metrics（包含4种评估指标）
    """
    output_dir = os.path.dirname(output_path)
    
    # 检查是否已经存在处理结果，如果存在则生成新版本
    metrics_exist, existing_metrics_path = check_metrics_exist(output_dir)
    
    if metrics_exist:
        # 获取下一个版本的文件名
        new_filename = get_next_metrics_version(output_dir)
        output_path = os.path.join(output_dir, new_filename)
        logger.info(f"Metrics already exist, creating new version: {output_path}")
    
    logger.info(f"Loading results from: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        out_samples = json.load(f)
    
    logger.info(f"Total samples: {len(out_samples)}")
    
    # 初始化评估器
    evaluators = initialize_evaluators()
    
    # 初始化metrics结构
    metrics = {
        "total_samples": len(out_samples),
        "accuracy": 0.0,
        "metrics_summary": {
            "rouge1": {},
            "rougeL": {},
            "bleu": {},
            "bertscore": {}
        },
        "five_metrics": {
            "rouge1": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "bertscore": 0.0
        },
        "sample_details": []
    }
    
    # 用于收集所有样本的分数
    rouge1_scores = []
    rougeL_scores = []
    bleu_scores = []
    bertscore_scores = []
    exact_match_scores = []
    
    # 处理每个样本
    for i, sample in enumerate(tqdm(out_samples, desc="Computing 4 metrics")):
        response = sample.get("response", "")
        response = simple_extract(response, "answer")
        answer = sample.get("answer", "")
        
        # 综合文本评估（4种指标）
        text_metrics = comprehensive_text_evaluation(answer, response, evaluators)
        
        # 保存样本级别的metrics
        sample_detail = {
            "id": sample.get("id", i),
            "response": response,
            "answer": answer,
            "metrics": text_metrics
        }
        metrics["sample_details"].append(sample_detail)
        
        # 收集各指标分数
        if 'rouge1' in text_metrics:
            rouge1_scores.append(text_metrics['rouge1'])
        if 'rougeL' in text_metrics:
            rougeL_scores.append(text_metrics['rougeL'])
        if 'bleu' in text_metrics:
            bleu_scores.append(text_metrics['bleu'])
        if 'bertscore' in text_metrics:
            bertscore_scores.append(text_metrics['bertscore'])
        if 'exact_match' in text_metrics:
            exact_match_scores.append(text_metrics['exact_match'])
    
    # 计算总体准确率
    metrics["accuracy"] = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
    
    # 计算每个指标的统计信息
    metrics["metrics_summary"]["rouge1"] = compute_statistics(rouge1_scores)
    metrics["metrics_summary"]["rougeL"] = compute_statistics(rougeL_scores)
    metrics["metrics_summary"]["bleu"] = compute_statistics(bleu_scores)
    metrics["metrics_summary"]["bertscore"] = compute_statistics(bertscore_scores)
    
    # 计算five_metrics的均值
    metrics["five_metrics"]["rouge1"] = metrics["metrics_summary"]["rouge1"]["mean"]
    metrics["five_metrics"]["rougeL"] = metrics["metrics_summary"]["rougeL"]["mean"]
    metrics["five_metrics"]["bleu"] = metrics["metrics_summary"]["bleu"]["mean"]
    metrics["five_metrics"]["bertscore"] = metrics["metrics_summary"]["bertscore"]["mean"]
    
    # 新增：计算所有指标的综合均值（排除gpt_judge）
    valid_metrics = [v for k, v in metrics["five_metrics"].items() if k != "gpt_judge"]
    metrics["five_metrics"]["average_metric"] = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0
    
    # 保存metrics
    logger.info(f"Saving metrics to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    # 打印摘要
    logger.info(f"Total Accuracy: {metrics['accuracy']:.4f}")
    logger.info("4 Evaluation Metrics Mean Values:")
    for metric_name, metric_value in metrics["five_metrics"].items():
        logger.info(f"  - {metric_name}: {metric_value:.4f}")
    
    logger.info("Detailed Statistics:")
    for metric_name, stats in metrics["metrics_summary"].items():
        logger.info(f"  - {metric_name}:")
        logger.info(f"      Max: {stats['max']:.4f}")
        logger.info(f"      Min: {stats['min']:.4f}")
        logger.info(f"      Mean: {stats['mean']:.4f}")
        logger.info(f"      Std: {stats['std']:.4f}")
        logger.info(f"      Count: {stats['count']}")
    
    return metrics

def batch_recompute_metrics(base_dir, dataset_name="OmniBrainBench-Open"):
    """
    批量处理多个模型的结果
    """
    if not os.path.exists(base_dir):
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    # 只选择以 _update 结尾的目录
    model_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('_update')]
    
    logger.info(f"Found {len(model_dirs)} model directories")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for model_dir in tqdm(model_dirs, desc="Processing models"):
        # 修复：直接使用 model_dir，不再添加 _update
        results_path = os.path.join(base_dir, model_dir, dataset_name, "results.json")
        
        if not os.path.exists(results_path):
            logger.warning(f"Results file not found: {results_path}")
            skip_count += 1
            continue
        
        # 修复：输出路径也直接使用 model_dir
        output_dir = os.path.join(base_dir, model_dir, dataset_name)
        metrics_path = os.path.join(output_dir, "open_metrics.json")
        
        logger.info(f"Processing {model_dir}/{dataset_name}")
        
        try:
            result = recompute_omnibrain_metrics(
                results_path=results_path,
                output_path=metrics_path
            )
            if result is not None:
                success_count += 1
                logger.info(f"✓ Successfully processed {model_dir}")
            else:
                logger.info(f"✓ {model_dir} already processed but created new version")
                success_count += 1
        except Exception as e:
            logger.error(f"✗ Failed to process {model_dir}: {e}")
            error_count += 1
    
    logger.info(f"Batch processing completed: Success={success_count}, Skipped={skip_count}, Errors={error_count}")

def process_single_model(model_name, base_dir=None, dataset_name="OmniBrainBench-Open"):
    """
    处理单个模型的metrics计算
    """
    if base_dir is None:
        base_dir = r"./eval_results"
    
    # 修复：如果传入的 model_name 不包含 _update，则添加
    if not model_name.endswith('_update'):
        model_dir = f"{model_name}_update"
    else:
        model_dir = model_name
    
    # 构建文件路径
    results_path = os.path.join(base_dir, model_dir, dataset_name, "results.json")
    output_dir = os.path.join(base_dir, model_dir, dataset_name)
    metrics_path = os.path.join(output_dir, "open_metrics.json")
    
    # 检查文件是否存在
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return False
    
    logger.info(f"Processing single model: {model_dir}")
    logger.info(f"Input: {results_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        result = recompute_omnibrain_metrics(
            results_path=results_path,
            output_path=metrics_path
        )
        if result is not None:
            logger.info(f"✓ Successfully processed {model_dir}")
            return True
        else:
            logger.info(f"✓ {model_dir} already processed but created new version")
            return True
    except Exception as e:
        logger.error(f"✗ Failed to process {model_dir}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Comprehensive text evaluation with 4 metrics')
    parser.add_argument('--results_path', type=str, help='Path to results.json file')
    parser.add_argument('--output_path', type=str, help='Path to save metrics file')
    parser.add_argument('--batch', action='store_true', help='Batch process multiple models')
    parser.add_argument('--base_dir', type=str, default=r'./eval_results', 
                       help='Base directory for batch processing')
    parser.add_argument('--dataset_name', type=str, default='OmniBrainBench-Open', 
                       help='Dataset name for batch processing')
    parser.add_argument('--model_name', type=str, help='Single model name to process')
    
    args = parser.parse_args()
    
    if args.model_name:
        # 处理单个模型
        process_single_model(
            model_name=args.model_name,
            base_dir=args.base_dir,
            dataset_name=args.dataset_name
        )
    elif args.batch:
        # 批量处理
        batch_recompute_metrics(
            base_dir=args.base_dir,
            dataset_name=args.dataset_name
        )
    else:
        # 单文件处理
        if not args.results_path or not args.output_path:
            parser.error("--results_path and --output_path are required when not using --batch mode or --model_name")
        
        recompute_omnibrain_metrics(
            results_path=args.results_path,
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()
    ## single model test example
    # CUDA_VISIBLE_DEVICES=5,6,7,8 python recompute_open_metrics.py --model_name "Qwen2.5-VL-7B"
    ## multiple models batch test example
    # CUDA_VISIBLE_DEVICES=5,6,7,8 python recompute_open_metrics.py --batch