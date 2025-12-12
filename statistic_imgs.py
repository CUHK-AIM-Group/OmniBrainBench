import pandas as pd
import os
import json
import openpyxl

base_folder = '/home/zhpeng/OmniBrainBench/OmniBrainBench/eval_results'

def collect_all_data():
    """收集所有模型的数据"""
    all_data = {}
    all_image_numbers = set()
    
    models_name = os.listdir(base_folder)
    
    for model_name in models_name:
        # 根据实际路径结构读取 metrics_img_num.json 文件
        metric_file = os.path.join(base_folder, model_name, 'OmniBrainBench', 'metrics_img_num.json')
        if not os.path.exists(metric_file):
            print(f"跳过 {model_name}，未找到 {metric_file}")
            continue
        
        try:
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
            
            # 获取总指标和图片数量指标 - 使用新的字段名称
            total_metrics = metrics.get('total_metrics', {})
            image_num_metrics = metrics.get('image_number_metrics', {})
            
            # 收集所有图片数量类别
            all_image_numbers.update(image_num_metrics.keys())
            
            # 保存模型数据
            all_data[model_name] = {
                'total_acc': total_metrics.get('accuracy', 0),  # 使用 accuracy 字段
                'total_count': total_metrics.get('total', 0),
                'total_right': total_metrics.get('right', 0),
                'image_numbers': image_num_metrics
            }
            
            print(f"成功处理模型: {model_name}")
            
        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
            continue
    
    return all_data, sorted(all_image_numbers, key=lambda x: int(x) if x.isdigit() else 0)

def create_excel_report():
    """创建Excel报告"""
    # 收集数据
    all_data, image_numbers = collect_all_data()
    
    if not all_data:
        print("未找到任何有效数据！")
        return pd.DataFrame()
    
    # 创建列名
    columns = ['Model Name', 'Total Accuracy', 'Total Samples', 'Correct Samples']
    
    # 添加图片数量列（只显示准确率）
    for img_num in image_numbers:
        columns.append(f"{img_num}img_acc")

    # 创建DataFrame
    rows = []
    for model_name, data in all_data.items():
        row = [
            model_name, 
            round(data['total_acc'], 4),
            data['total_count'],
            data['total_right']
        ]
        
        # 添加图片数量准确率
        for img_num in image_numbers:
            if img_num in data['image_numbers']:
                img_data = data['image_numbers'][img_num]
                acc = img_data.get('accuracy', 0)  # 使用 accuracy 字段
                row.append(round(acc, 4))
            else:
                row.append('N/A')

        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # 按总准确率排序
    df = df.sort_values('Total Accuracy', ascending=False)
    
    return df

def save_to_excel(df, filename='model_evaluation_results_image_num.xlsx'):
    """保存到Excel文件"""
    if df.empty:
        print("没有数据可保存！")
        return
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 保存主表（所有结果）
        df.to_excel(writer, sheet_name='All Results', index=False)
        
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

def analyze_image_number_distribution(all_data, image_numbers):
    """分析图片数量分布"""
    print("\n=== 图片数量分布统计 ===")
    
    # 统计每个图片数量的总样本数
    img_num_distribution = {}
    for img_num in image_numbers:
        total_samples = 0
        for model_data in all_data.values():
            if img_num in model_data['image_numbers']:
                total_samples += model_data['image_numbers'][img_num].get('total', 0)
        img_num_distribution[img_num] = total_samples
    
    # 按图片数量排序
    sorted_distribution = dict(sorted(img_num_distribution.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0))
    
    total_all_samples = sum(sorted_distribution.values())
    print(f"总样本数: {total_all_samples}")
    print("各图片数量样本分布:")
    for img_num, count in sorted_distribution.items():
        percentage = (count / total_all_samples) * 100 if total_all_samples > 0 else 0
        print(f"  {img_num}张图片: {count}个样本 ({percentage:.2f}%)")

def create_detailed_stats_excel(all_data, image_numbers):
    """创建详细的统计Excel文件"""
    if not all_data:
        return
    
    # 创建详细统计表
    detailed_rows = []
    for model_name, data in all_data.items():
        for img_num in image_numbers:
            if img_num in data['image_numbers']:
                img_data = data['image_numbers'][img_num]
                detailed_rows.append([
                    model_name,
                    img_num,
                    img_data.get('total', 0),
                    img_data.get('right', 0),
                    round(img_data.get('accuracy', 0), 4)  # 使用 accuracy 字段
                ])
    
    detailed_df = pd.DataFrame(detailed_rows, 
                              columns=['Model Name', 'Image Number', 'Total Samples', 'Correct Samples', 'Accuracy'])
    
    # 保存详细统计
    detailed_filename = 'detailed_image_number_stats.xlsx'
    with pd.ExcelWriter(detailed_filename, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed Stats', index=False)
        
        # 创建透视表
        pivot_df = detailed_df.pivot_table(
            index='Model Name', 
            columns='Image Number', 
            values='Accuracy', 
            aggfunc='first'
        )
        pivot_df.to_excel(writer, sheet_name='Accuracy Pivot')
        
        # 格式化工作表
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
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
    
    print(f"详细统计文件已保存为: {detailed_filename}")

def check_directory_structure():
    """检查目录结构"""
    print("=== 检查目录结构 ===")
    models_name = os.listdir(base_folder)
    found_models = []
    missing_models = []
    
    for model_name in models_name:
        metric_file = os.path.join(base_folder, model_name, 'OmniBrainBench', 'metrics_img_num.json')
        if os.path.exists(metric_file):
            found_models.append(model_name)
            print(f"✓ 找到: {model_name}")
        else:
            missing_models.append(model_name)
            print(f"✗ 缺失: {model_name}")
    
    print(f"\n找到 {len(found_models)} 个模型的数据文件")
    print(f"缺失 {len(missing_models)} 个模型的数据文件")
    return found_models, missing_models

def display_sample_data(all_data):
    """显示样本数据示例"""
    print("\n=== 数据样本示例 ===")
    for model_name, data in list(all_data.items())[:2]:  # 只显示前2个模型
        print(f"\n模型: {model_name}")
        print(f"总准确率: {data['total_acc']:.4f}")
        print(f"总样本数: {data['total_count']}")
        print(f"正确样本数: {data['total_right']}")
        print("图片数量统计:")
        for img_num, img_data in data['image_numbers'].items():
            print(f"  {img_num}张图片: {img_data.get('total', 0)}样本, {img_data.get('right', 0)}正确, 准确率: {img_data.get('accuracy', 0):.4f}")

if __name__ == "__main__":
    # 首先检查目录结构
    found_models, missing_models = check_directory_structure()
    
    if not found_models:
        print("没有找到任何模型的数据文件！")
        exit(1)
    
    # 生成Excel报告
    df = create_excel_report()
    
    if not df.empty:
        # 获取数据用于统计
        all_data, image_numbers = collect_all_data()
        
        # 显示数据样本
        display_sample_data(all_data)
        
        save_to_excel(df)
        
        # 显示预览
        print("\n=== 模型评估结果预览 ===")
        # 只显示前几列避免输出过长
        preview_cols = ['Model Name', 'Total Accuracy'] + [f"{img_num}img_acc" for img_num in image_numbers[:3]]
        if len(preview_cols) > 6:
            preview_cols = preview_cols[:6]
        print(df[preview_cols].to_string(index=False))
        print(f"\n总共包含 {len(df)} 个模型")
        print(f"图片数量类型: {len(image_numbers)}")
        print(f"图片数量列表: {', '.join(image_numbers)}")
        
        # 分析图片数量分布
        analyze_image_number_distribution(all_data, image_numbers)
        
        # 创建详细统计文件
        create_detailed_stats_excel(all_data, image_numbers)
    else:
        print("没有生成有效的数据！")