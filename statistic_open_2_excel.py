import pandas as pd
import os
import json
import openpyxl

base_folder = './eval_results'

# 严格按照指定的模型名称和顺序
STANDARD_MODEL_ORDER = [
    'Janus-Pro-7B',
    'InternVL3-8B',
    'InternVL3-9B',
    'InternVL3-14B',
    'InternVL3-38B',
    'Qwen2.5-VL-7B',
    'Qwen2.5-VL-32B',
    'Qwen3-VL-4B',
    'Qwen3-VL-8B',
    'Qwen3-VL-30B',
    'MedVLM-R1-2B',
    'MedGemma-4B',
    'Llava-Med-7B',
    'Lingshu-7B',
    'Lingshu-32B',
    'HuatuoGPT-V-7B',
    'HuatuoGPT-V-34B',
    'Deepseek-V3.1',
    'Grok-4',
    'GPT-4o',
    'GPT-5',
    'GPT-5-mini',
    'Claude-4.5-Sonnet',
    'Gemini-2.5-Pro'
]

# 模型名称映射规则
MODEL_NAME_MAPPING = {
    # Janus
    'janus-pro-7b': 'Janus-Pro-7B',
    'janus-pro': 'Janus-Pro-7B',
    
    # InternVL3
    'internvl3-8b': 'InternVL3-8B',
    'internvl3-9b': 'InternVL3-9B',
    'internvl3-14b': 'InternVL3-14B',
    'internvl3-38b': 'InternVL3-38B',
    
    # Qwen
    'qwen2.5-vl-7b': 'Qwen2.5-VL-7B',
    'qwen2.5-vl-32b': 'Qwen2.5-VL-32B',
    'qwen3-vl-4b': 'Qwen3-VL-4B',
    'qwen3-vl-8b': 'Qwen3-VL-8B',
    'qwen3-vl-30b': 'Qwen3-VL-30B',
    
    # Med models
    'medvlm-r1-2b': 'MedVLM-R1-2B',
    'medgemma-4b': 'MedGemma-4B',
    'llava-med-7b': 'Llava-Med-7B',
    
    # Lingshu
    'lingshu-7b': 'Lingshu-7B',
    'lingshu-32b': 'Lingshu-32B',
    
    # Huatuo
    'huatuo-7b': 'HuatuoGPT-V-7B',
    'huatuogpt-v-7b': 'HuatuoGPT-V-7B',
    'huatuo-34b': 'HuatuoGPT-V-34B',
    'huatuogpt-v-34b': 'HuatuoGPT-V-34B',
    
    # Deepseek
    'deepseek-v3.1': 'Deepseek-V3.1',
    'deepseek-v3': 'Deepseek-V3.1',
    
    # Grok
    'grok-4': 'Grok-4',
    
    # GPT
    'gpt-4o': 'GPT-4o',
    'gpt-5-2025-08-07': 'GPT-5',
    'gpt-5-mini-2025-08-07': 'GPT-5-mini',
    
    # Claude
    'claude-sonnet-4-5-20250929': 'Claude-4.5-Sonnet',
    'claude-4.5-sonnet': 'Claude-4.5-Sonnet',
    
    # Gemini
    'gemini-2.5-pro': 'Gemini-2.5-Pro'
}

def normalize_model_name(raw_name):
    """将原始模型名称标准化"""
    # 去掉_update后缀
    if raw_name.endswith('_update'):
        raw_name = raw_name[:-7]
    
    # 转换为小写进行匹配
    lower_name = raw_name.lower()
    
    # 查找匹配的标准化名称
    for pattern, standard_name in MODEL_NAME_MAPPING.items():
        if pattern in lower_name:
            return standard_name
    
    # 如果没有找到匹配，尝试模糊匹配
    for standard_name in STANDARD_MODEL_ORDER:
        std_lower = standard_name.lower()
        if std_lower in lower_name or lower_name in std_lower:
            return standard_name
    
    # 如果还是没有匹配，返回原始名称（首字母大写）
    return raw_name

def collect_all_data():
    """收集所有模型的数据"""
    all_data = {}
    
    # 查找所有以_update结尾的目录
    all_dirs = [d for d in os.listdir(base_folder) 
                if os.path.isdir(os.path.join(base_folder, d)) and d.endswith('_update')]
    
    print("找到的模型目录和映射:")
    directory_mapping = {}
    for dir_name in all_dirs:
        raw_model_name = dir_name.replace('_update', '')
        standardized_name = normalize_model_name(raw_model_name)
        directory_mapping[standardized_name] = dir_name
        print(f"  {dir_name} -> {standardized_name}")
    
    # 按照标准顺序处理模型
    for standard_name in STANDARD_MODEL_ORDER:
        print(f"\n处理模型: {standard_name}")
        
        # 查找匹配的目录
        matched_dir = directory_mapping.get(standard_name)
        
        if not matched_dir:
            print(f"  警告: 未找到匹配的目录 for {standard_name}")
            all_data[standard_name] = None
            continue
        
        # 构建文件路径
        metric_file = os.path.join(base_folder, matched_dir, 'OmniBrainBench-Open', 'open_metrics_v1.json')
        
        if not os.path.exists(metric_file):
            print(f"  警告: 未找到文件 {metric_file}")
            all_data[standard_name] = None
            continue
        
        try:
            with open(metric_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # 提取five_metrics中的指标
            five_metrics = metrics_data.get('five_metrics', {})
            ROUGE1 = five_metrics.get('ROUGE1', 0)
            ROUGEL = five_metrics.get('ROUGEL', 0)
            BLEU = five_metrics.get('BLEU', 0)
            BERTScore = five_metrics.get('BERTScore', 0)
            average_metric = five_metrics.get('average_metric', 0)
            
            # 提取metrics_summary中的统计数据
            metrics_summary = metrics_data.get('metrics_summary', {})
            summary_stats = {}
            for metric_name, stats in metrics_summary.items():
                if isinstance(stats, dict):
                    summary_stats[f"{metric_name}_max"] = stats.get('max', 0)
                    summary_stats[f"{metric_name}_min"] = stats.get('min', 0)
                    summary_stats[f"{metric_name}_mean"] = stats.get('mean', 0)
                    summary_stats[f"{metric_name}_std"] = stats.get('std', 0)
            
            # 统计sample_details中response包含"[ERROR]"的行数量
            sample_details = metrics_data.get('sample_details', [])
            error_count = 0
            valid_samples_metrics = []  # 存储没有错误的样本的metrics
            
            for sample in sample_details:
                response = sample.get('response', '')
                if '[ERROR]' in str(response):
                    error_count += 1
                else:
                    # 收集没有错误的样本的metrics
                    sample_metrics = sample.get('metrics', {})
                    valid_samples_metrics.append({
                        'ROUGE1': sample_metrics.get('ROUGE1', 0),
                        'ROUGEL': sample_metrics.get('ROUGEL', 0),
                        'BLEU': sample_metrics.get('BLEU', 0),
                        'BERTScore': sample_metrics.get('BERTScore', 0)
                    })
            
            # 计算有效样本的metrics平均值
            if valid_samples_metrics:
                valid_metrics_df = pd.DataFrame(valid_samples_metrics)
                valid_ROUGE1_mean = valid_metrics_df['ROUGE1'].mean()
                valid_ROUGEL_mean = valid_metrics_df['ROUGEL'].mean()
                valid_BLEU_mean = valid_metrics_df['BLEU'].mean()
                valid_BERTScore_mean = valid_metrics_df['BERTScore'].mean()
            else:
                valid_ROUGE1_mean = 0
                valid_ROUGEL_mean = 0
                valid_BLEU_mean = 0
                valid_BERTScore_mean = 0
            
            # 保存模型数据
            all_data[standard_name] = {
                'five_metrics': {
                    'ROUGE1': ROUGE1,
                    'ROUGEL': ROUGEL,
                    'BLEU': BLEU,
                    'BERTScore': BERTScore,
                    'average_metric': average_metric
                },
                'metrics_summary_stats': summary_stats,
                'error_count': error_count,
                'valid_samples_count': len(valid_samples_metrics),
                'valid_samples_metrics': {
                    'ROUGE1_mean': valid_ROUGE1_mean,
                    'ROUGEL_mean': valid_ROUGEL_mean,
                    'BLEU_mean': valid_BLEU_mean,
                    'BERTScore_mean': valid_BERTScore_mean
                }
            }
            
            print(f"  成功处理: {standard_name}")
            
        except Exception as e:
            print(f"  处理模型 {standard_name} 时出错: {str(e)}")
            all_data[standard_name] = None
            continue
    
    return all_data

def create_excel_report():
    """创建Excel报告"""
    # 收集数据
    all_data = collect_all_data()
    
    if not all_data:
        print("未找到任何有效的模型数据！")
        return None
    
    # 创建工作表数据 - 严格按照STANDARD_MODEL_ORDER顺序
    five_metrics_rows = []
    valid_samples_rows = []
    
    for model_name in STANDARD_MODEL_ORDER:
        data = all_data.get(model_name)
        
        if data is None:
            # 对于找不到数据的模型，创建空行
            five_metrics_row = {'Model Name': model_name}
            # 为所有可能的metrics_summary列添加空值
            five_metrics_row.update({
                'ROUGE1': 0, 'ROUGEL': 0, 'BLEU': 0, 'BERTScore': 0, 'average_metric': 0
            })
            
            valid_samples_row = {
                'Model Name': model_name,
                'valid_samples_count': 0,
                'error_count': 0,
                'ROUGE1_mean': 0,
                'ROUGEL_mean': 0,
                'BLEU_mean': 0,
                'BERTScore_mean': 0
            }
        else:
            # 工作表1: five_metrics数据
            five_metrics_row = {
                'Model Name': model_name,
                'ROUGE1': data['five_metrics']['ROUGE1'],
                'ROUGEL': data['five_metrics']['ROUGEL'],
                'BLEU': data['five_metrics']['BLEU'],
                'BERTScore': data['five_metrics']['BERTScore'],
                'average_metric': data['five_metrics']['average_metric']
            }
            
            # 添加metrics_summary统计数据
            for stat_name, stat_value in data['metrics_summary_stats'].items():
                five_metrics_row[stat_name] = stat_value
            
            # 工作表2: 有效样本的metrics平均值
            valid_samples_row = {
                'Model Name': model_name,
                'valid_samples_count': data['valid_samples_count'],
                'error_count': data['error_count'],
                'ROUGE1_mean': data['valid_samples_metrics']['ROUGE1_mean'],
                'ROUGEL_mean': data['valid_samples_metrics']['ROUGEL_mean'],
                'BLEU_mean': data['valid_samples_metrics']['BLEU_mean'],
                'BERTScore_mean': data['valid_samples_metrics']['BERTScore_mean']
            }
        
        five_metrics_rows.append(five_metrics_row)
        valid_samples_rows.append(valid_samples_row)
    
    # 创建DataFrames - 不进行任何排序
    df_five_metrics = pd.DataFrame(five_metrics_rows)
    df_valid_samples = pd.DataFrame(valid_samples_rows)
    
    # 确保列的顺序一致：Model Name + five_metrics + metrics_summary
    base_columns = ['Model Name', 'ROUGE1', 'ROUGEL', 'BLEU', 'BERTScore', 'average_metric']
    other_columns = [col for col in df_five_metrics.columns if col not in base_columns]
    df_five_metrics = df_five_metrics[base_columns + other_columns]
    
    return {
        'five_metrics': df_five_metrics,
        'valid_samples': df_valid_samples
    }

def save_to_excel(dataframes, filename='model_evaluation_results_new.xlsx'):
    """保存到Excel文件"""
    if dataframes is None:
        print("没有数据可保存！")
        return
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 保存工作表1: five_metrics - 严格按照STANDARD_MODEL_ORDER顺序
        dataframes['five_metrics'].to_excel(writer, sheet_name='five_metrics', index=False)
        
        # 保存工作表2: valid_samples - 严格按照STANDARD_MODEL_ORDER顺序
        dataframes['valid_samples'].to_excel(writer, sheet_name='valid_samples', index=False)
        
        # 格式化工作表
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            
            # 调整列宽
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Excel文件已保存为: {filename}")

if __name__ == "__main__":
    # 生成Excel报告
    dataframes = create_excel_report()
    
    if dataframes:
        save_to_excel(dataframes)
        
        # 显示预览 - 显示完整列表确认顺序
        print("\n=== five_metrics工作表（按指定顺序）===")
        print(dataframes['five_metrics'][['Model Name', 'ROUGE1', 'ROUGEL', 'BLEU', 'BERTScore', 'average_metric']].to_string(index=False))
        
        print("\n=== valid_samples工作表（按指定顺序）===")
        print(dataframes['valid_samples'].to_string(index=False))
        
        # 统计有效模型数量
        valid_models = sum(1 for model in dataframes['five_metrics']['Model Name'] 
                          if dataframes['five_metrics'].loc[dataframes['five_metrics']['Model Name'] == model, 'ROUGE1'].iloc[0] != 0)
        
        print(f"\n总共包含 {len(dataframes['five_metrics'])} 个模型（严格按照指定顺序）")
        print(f"其中 {valid_models} 个模型有有效数据")
        
        # 确认顺序
        print(f"\n模型顺序确认:")
        for i, model in enumerate(dataframes['five_metrics']['Model Name'], 1):
            has_data = "有数据" if dataframes['five_metrics'].loc[dataframes['five_metrics']['Model Name'] == model, 'ROUGE1'].iloc[0] != 0 else "无数据"
            print(f"{i:2d}. {model} - {has_data}")
    else:
        print("未能生成任何数据！")