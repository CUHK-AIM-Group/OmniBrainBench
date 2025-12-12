import pandas as pd
import os
import json
import openpyxl

base_folder = './eval_results'

# Strictly follow the specified model names and order
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

# Model name mapping rules
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
    """Normalize raw model name to standard name"""
    # Remove _update suffix
    if raw_name.endswith('_update'):
        raw_name = raw_name[:-7]
    
    # Convert to lowercase for matching
    lower_name = raw_name.lower()
    
    # Find matching standardized name
    for pattern, standard_name in MODEL_NAME_MAPPING.items():
        if pattern in lower_name:
            return standard_name
    
    # If no match found, try fuzzy matching
    for standard_name in STANDARD_MODEL_ORDER:
        std_lower = standard_name.lower()
        if std_lower in lower_name or lower_name in std_lower:
            return standard_name
    
    # If still no match, return original name (capitalize first letter)
    return raw_name

def collect_all_data():
    """Collect data from all models"""
    all_data = {}
    
    # Find all directories ending with _update
    all_dirs = [d for d in os.listdir(base_folder) 
                if os.path.isdir(os.path.join(base_folder, d)) and d.endswith('_update')]
    
    print("Found model directories and mappings:")
    directory_mapping = {}
    for dir_name in all_dirs:
        raw_model_name = dir_name.replace('_update', '')
        standardized_name = normalize_model_name(raw_model_name)
        directory_mapping[standardized_name] = dir_name
        print(f"  {dir_name} -> {standardized_name}")
    
    # Process models in standard order
    for standard_name in STANDARD_MODEL_ORDER:
        print(f"\nProcessing model: {standard_name}")
        
        # Find matching directory
        matched_dir = directory_mapping.get(standard_name)
        
        if not matched_dir:
            print(f"  Warning: No matching directory found for {standard_name}")
            all_data[standard_name] = None
            continue
        
        # Build file path
        metric_file = os.path.join(base_folder, matched_dir, 'OmniBrainBench-Open', 'open_metrics_v1.json')
        
        if not os.path.exists(metric_file):
            print(f"  Warning: File not found {metric_file}")
            all_data[standard_name] = None
            continue
        
        try:
            with open(metric_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # Extract metrics from five_metrics
            five_metrics = metrics_data.get('five_metrics', {})
            ROUGE1 = five_metrics.get('ROUGE1', 0)
            ROUGEL = five_metrics.get('ROUGEL', 0)
            BLEU = five_metrics.get('BLEU', 0)
            BERTScore = five_metrics.get('BERTScore', 0)
            average_metric = five_metrics.get('average_metric', 0)
            
            # Extract statistical data from metrics_summary
            metrics_summary = metrics_data.get('metrics_summary', {})
            summary_stats = {}
            for metric_name, stats in metrics_summary.items():
                if isinstance(stats, dict):
                    summary_stats[f"{metric_name}_max"] = stats.get('max', 0)
                    summary_stats[f"{metric_name}_min"] = stats.get('min', 0)
                    summary_stats[f"{metric_name}_mean"] = stats.get('mean', 0)
                    summary_stats[f"{metric_name}_std"] = stats.get('std', 0)
            
            # Count samples where response contains "[ERROR]"
            sample_details = metrics_data.get('sample_details', [])
            error_count = 0
            valid_samples_metrics = []  # Store metrics from samples without errors
            
            for sample in sample_details:
                response = sample.get('response', '')
                if '[ERROR]' in str(response):
                    error_count += 1
                else:
                    # Collect metrics from samples without errors
                    sample_metrics = sample.get('metrics', {})
                    valid_samples_metrics.append({
                        'ROUGE1': sample_metrics.get('ROUGE1', 0),
                        'ROUGEL': sample_metrics.get('ROUGEL', 0),
                        'BLEU': sample_metrics.get('BLEU', 0),
                        'BERTScore': sample_metrics.get('BERTScore', 0)
                    })
            
            # Calculate average metrics for valid samples
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
            
            # Save model data
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
            
            print(f"  Successfully processed: {standard_name}")
            
        except Exception as e:
            print(f"  Error processing model {standard_name}: {str(e)}")
            all_data[standard_name] = None
            continue
    
    return all_data

def create_excel_report():
    """Create Excel report"""
    # Collect data
    all_data = collect_all_data()
    
    if not all_data:
        print("No valid model data found!")
        return None
    
    # Create worksheet data - strictly follow STANDARD_MODEL_ORDER
    five_metrics_rows = []
    valid_samples_rows = []
    
    for model_name in STANDARD_MODEL_ORDER:
        data = all_data.get(model_name)
        
        if data is None:
            # Create empty row for models without data
            five_metrics_row = {'Model Name': model_name}
            # Add null values for all possible metrics_summary columns
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
            # Worksheet 1: five_metrics data
            five_metrics_row = {
                'Model Name': model_name,
                'ROUGE1': data['five_metrics']['ROUGE1'],
                'ROUGEL': data['five_metrics']['ROUGEL'],
                'BLEU': data['five_metrics']['BLEU'],
                'BERTScore': data['five_metrics']['BERTScore'],
                'average_metric': data['five_metrics']['average_metric']
            }
            
            # Add metrics_summary statistics
            for stat_name, stat_value in data['metrics_summary_stats'].items():
                five_metrics_row[stat_name] = stat_value
            
            # Worksheet 2: Average metrics for valid samples
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
    
    # Create DataFrames - no sorting
    df_five_metrics = pd.DataFrame(five_metrics_rows)
    df_valid_samples = pd.DataFrame(valid_samples_rows)
    
    # Ensure consistent column order: Model Name + five_metrics + metrics_summary
    base_columns = ['Model Name', 'ROUGE1', 'ROUGEL', 'BLEU', 'BERTScore', 'average_metric']
    other_columns = [col for col in df_five_metrics.columns if col not in base_columns]
    df_five_metrics = df_five_metrics[base_columns + other_columns]
    
    return {
        'five_metrics': df_five_metrics,
        'valid_samples': df_valid_samples
    }

def save_to_excel(dataframes, filename='model_evaluation_results_new.xlsx'):
    """Save to Excel file"""
    if dataframes is None:
        print("No data to save!")
        return
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Save worksheet 1: five_metrics - strictly follow STANDARD_MODEL_ORDER
        dataframes['five_metrics'].to_excel(writer, sheet_name='five_metrics', index=False)
        
        # Save worksheet 2: valid_samples - strictly follow STANDARD_MODEL_ORDER
        dataframes['valid_samples'].to_excel(writer, sheet_name='valid_samples', index=False)
        
        # Format worksheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            
            # Adjust column width
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
    
    print(f"Excel file saved as: {filename}")

if __name__ == "__main__":
    # Generate Excel report
    dataframes = create_excel_report()
    
    if dataframes:
        save_to_excel(dataframes)
        
        # Display preview - show complete list to confirm order
        print("\n=== five_metrics worksheet (in specified order) ===")
        print(dataframes['five_metrics'][['Model Name', 'ROUGE1', 'ROUGEL', 'BLEU', 'BERTScore', 'average_metric']].to_string(index=False))
        
        print("\n=== valid_samples worksheet (in specified order) ===")
        print(dataframes['valid_samples'].to_string(index=False))
        
        # Count valid models
        valid_models = sum(1 for model in dataframes['five_metrics']['Model Name'] 
                          if dataframes['five_metrics'].loc[dataframes['five_metrics']['Model Name'] == model, 'ROUGE1'].iloc[0] != 0)
        
        print(f"\nTotal models: {len(dataframes['five_metrics'])} (strictly in specified order)")
        print(f"Models with valid data: {valid_models}")
        
        # Confirm order
        print(f"\nModel order confirmation:")
        for i, model in enumerate(dataframes['five_metrics']['Model Name'], 1):
            has_data = "has data" if dataframes['five_metrics'].loc[dataframes['five_metrics']['Model Name'] == model, 'ROUGE1'].iloc[0] != 0 else "no data"
            print(f"{i:2d}. {model} - {has_data}")
    else:
        print("Failed to generate any data!")