import pandas as pd
import os
import json
import openpyxl


base_folder = '/data_hdd/syliu/workspace/Med-VLM/MedEvalKit/eval_results'

def collect_all_data():
    """收集所有模型的数据"""
    all_data = {}
    all_task_types = set()
    all_clinical_phases = set()
    
    models_name = os.listdir(base_folder)
    
    for model_name in models_name:
        model_folder = os.path.join(base_folder, model_name)
        
        # 检查是否是文件夹
        if not os.path.isdir(model_folder):
            continue
        
        # 检查是否存在OmniBrainBench子文件夹，如果不存在则跳过
        omnibrain_folder = os.path.join(model_folder, 'OmniBrainBench')
        if not os.path.exists(omnibrain_folder):
            print(f"[SKIP] {model_name}: 没有OmniBrainBench子文件夹")
            continue
        
        metric_file = os.path.join(model_folder, 'total_results.json')
        if not os.path.exists(metric_file):
            continue
        
        with open(metric_file, 'r') as f:
            metrics = json.load(f)
            metrics = metrics['OmniBrainBench']
            
            total_metrics = metrics['total metrics']
            task_type_metrics = metrics['task type metrics']
            clinical_phase_type_metrics = metrics['clinical phase type metrics']
            
            # 收集所有task type和clinical phase类别
            all_task_types.update(task_type_metrics.keys())
            all_clinical_phases.update(clinical_phase_type_metrics.keys())
            
            # 保存模型数据
            all_data[model_name] = {
                'total_acc': total_metrics['acc'],
                'task_types': task_type_metrics,
                'clinical_phases': clinical_phase_type_metrics
            }
    
    return all_data, sorted(all_task_types), sorted(all_clinical_phases)

def create_excel_report():
    """创建Excel报告"""
    # 收集数据
    all_data, task_types, clinical_phases = collect_all_data()
    
    # 创建列名
    columns = ['Model Name', 'Total Accuracy']
    
    # 添加task type列
    for task_type in task_types:
        columns.append(task_type)
    
    # 添加clinical phase列
    for clinical_phase in clinical_phases:
        columns.append(clinical_phase)
    
    # 创建DataFrame
    rows = []
    for model_name, data in all_data.items():
        row = [model_name, round(data['total_acc'], 4)]
        
        # 添加task type准确率
        for task_type in task_types:
            if task_type in data['task_types']:
                acc = data['task_types'][task_type]['acc']
                row.append(round(acc, 4))
            else:
                row.append('N/A')
        
        # 添加clinical phase准确率
        for clinical_phase in clinical_phases:
            if clinical_phase in data['clinical_phases']:
                acc = data['clinical_phases'][clinical_phase]['acc']
                row.append(round(acc, 4))
            else:
                row.append('N/A')
        
        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # 按总准确率排序
    df = df.sort_values('Total Accuracy', ascending=False)
    
    return df

def save_to_excel(df, filename='model_evaluation_results_20251126.xlsx'):
    """保存到Excel文件"""
    # 获取task types和clinical phases用于分表
    _, task_types, clinical_phases = collect_all_data()
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 保存主表
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # 创建分别的task type和clinical phase表
        # 使用集合来准确匹配列名
        task_type_set = set(task_types)
        clinical_phase_set = set(clinical_phases)
        
        task_cols = ['Model Name', 'Total Accuracy'] + [col for col in df.columns if col in task_type_set]
        phase_cols = ['Model Name', 'Total Accuracy'] + [col for col in df.columns if col in clinical_phase_set]
        
        df[task_cols].to_excel(writer, sheet_name='Task Types', index=False)
        df[phase_cols].to_excel(writer, sheet_name='Clinical Phases', index=False)
        
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
    df = create_excel_report()
    save_to_excel(df)
    
    # 获取task types和clinical phases用于统计
    _, task_types, clinical_phases = collect_all_data()
    
    # 显示预览
    print("\n=== 模型评估结果预览 ===")
    print(df.to_string(index=False, max_cols=5))
    print(f"\n总共包含 {len(df)} 个模型")
    print(f"Task类型数量: {len(task_types)}")
    print(f"Clinical Phase类型数量: {len(clinical_phases)}")
            
