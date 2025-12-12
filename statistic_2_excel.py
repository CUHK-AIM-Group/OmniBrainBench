import pandas as pd
import os
import json
import openpyxl


base_folder = '/data_hdd/syliu/workspace/Med-VLM/MedEvalKit/eval_results'

def collect_all_data():
    """Collect data from all models"""
    all_data = {}
    all_task_types = set()
    all_clinical_phases = set()
    
    models_name = os.listdir(base_folder)
    
    for model_name in models_name:
        model_folder = os.path.join(base_folder, model_name)
        
        # Check if it's a directory
        if not os.path.isdir(model_folder):
            continue
        
        # Check if OmniBrainBench subfolder exists, skip if not
        omnibrain_folder = os.path.join(model_folder, 'OmniBrainBench')
        if not os.path.exists(omnibrain_folder):
            print(f"[SKIP] {model_name}: No OmniBrainBench subfolder")
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
            
            # Collect all task types and clinical phase categories
            all_task_types.update(task_type_metrics.keys())
            all_clinical_phases.update(clinical_phase_type_metrics.keys())
            
            # Save model data
            all_data[model_name] = {
                'total_acc': total_metrics['acc'],
                'task_types': task_type_metrics,
                'clinical_phases': clinical_phase_type_metrics
            }
    
    return all_data, sorted(all_task_types), sorted(all_clinical_phases)

def create_excel_report():
    """Create Excel report"""
    # Collect data
    all_data, task_types, clinical_phases = collect_all_data()
    
    # Create column names
    columns = ['Model Name', 'Total Accuracy']
    
    # Add task type columns
    for task_type in task_types:
        columns.append(task_type)
    
    # Add clinical phase columns
    for clinical_phase in clinical_phases:
        columns.append(clinical_phase)
    
    # Create DataFrame
    rows = []
    for model_name, data in all_data.items():
        row = [model_name, round(data['total_acc'], 4)]
        
        # Add task type accuracy
        for task_type in task_types:
            if task_type in data['task_types']:
                acc = data['task_types'][task_type]['acc']
                row.append(round(acc, 4))
            else:
                row.append('N/A')
        
        # Add clinical phase accuracy
        for clinical_phase in clinical_phases:
            if clinical_phase in data['clinical_phases']:
                acc = data['clinical_phases'][clinical_phase]['acc']
                row.append(round(acc, 4))
            else:
                row.append('N/A')
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Sort by total accuracy
    df = df.sort_values('Total Accuracy', ascending=False)
    
    return df

def save_to_excel(df, filename='model_evaluation_results_20251126.xlsx'):
    """Save to Excel file"""
    # Get task types and clinical phases for separate sheets
    _, task_types, clinical_phases = collect_all_data()
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Save main sheet
        df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Create separate task type and clinical phase sheets
        # Use sets to accurately match column names
        task_type_set = set(task_types)
        clinical_phase_set = set(clinical_phases)
        
        task_cols = ['Model Name', 'Total Accuracy'] + [col for col in df.columns if col in task_type_set]
        phase_cols = ['Model Name', 'Total Accuracy'] + [col for col in df.columns if col in clinical_phase_set]
        
        df[task_cols].to_excel(writer, sheet_name='Task Types', index=False)
        df[phase_cols].to_excel(writer, sheet_name='Clinical Phases', index=False)
        
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
    df = create_excel_report()
    save_to_excel(df)
    
    # Get task types and clinical phases for statistics
    _, task_types, clinical_phases = collect_all_data()
    
    # Display preview
    print("\n=== Model Evaluation Results Preview ===")
    print(df.to_string(index=False, max_cols=5))
    print(f"\nTotal models: {len(df)}")
    print(f"Number of task types: {len(task_types)}")
    print(f"Number of clinical phase types: {len(clinical_phases)}")
            
