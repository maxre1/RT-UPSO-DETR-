###################################
# 本工具用于分析车位数据集中车位的尺寸分布，
# 为车位补全模块的距离阈值提供数据驱动的设定依据。
# 通过统计所有标注车位的宽度、高度、面积及宽高比，
# 生成详细的统计报告和可视化图表，并给出不同严格程度的阈值建议。
####################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import platform
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] 

plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams.update({
    'font.size': 14,           
    'axes.titlesize': 18,      
    'axes.labelsize': 16,      
    'xtick.labelsize': 14,     
    'ytick.labelsize': 14,     
    'legend.fontsize': 16,     
    'figure.titlesize': 18,    
})

system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] 
plt.rcParams['axes.unicode_minus'] = False 
# ==========================================================

def analyze_parking_slot_sizes(dataset_root, output_dir="size_analysis"):
    
    img_dir = os.path.join(dataset_root, "image")
    label_dir = os.path.join(dataset_root, "label")
    
    if not os.path.exists(img_dir):
        img_dir = os.path.join(dataset_root, "image")  
    
    if not os.path.exists(label_dir):
        label_dir = os.path.join(dataset_root, "label")  
    
    os.makedirs(output_dir, exist_ok=True)

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    if not label_files:
        print(f"错误: 在 {label_dir} 中未找到标签文件")
        return
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    all_widths = []
    all_heights = []
    all_areas = []
    all_aspect_ratios = []
    per_image_stats = []    
    class_counts = {}

    for label_file in tqdm(label_files, desc="分析车位尺寸"):
        img_name = os.path.splitext(os.path.basename(label_file))[0]
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_path = None
        
        for ext in img_extensions:
            potential_path = os.path.join(img_dir, img_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            print(f"警告: 未找到 {img_name} 的图像文件")
            continue
        try:
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
            img_h, img_w = img.shape[:2]
        except ImportError:
            print("警告: 未安装OpenCV，将使用默认图像尺寸")
            img_w, img_h = 1920, 1080  
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        image_widths = []
        image_heights = []
        image_areas = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            nx, ny, nw, nh = map(float, parts[1:])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            width_px = nw * img_w
            height_px = nh * img_h
            area_px = width_px * height_px
            aspect_ratio = width_px / height_px if height_px > 0 else 0
            all_widths.append(width_px)
            all_heights.append(height_px)
            all_areas.append(area_px)
            all_aspect_ratios.append(aspect_ratio)
            
            image_widths.append(width_px)
            image_heights.append(height_px)
            image_areas.append(area_px)
        if image_widths:
            per_image_stats.append({
                'image': img_name,
                'slot_count': len(image_widths),
                'avg_width': np.mean(image_widths),
                'avg_height': np.mean(image_heights),
                'avg_area': np.mean(image_areas)
            })
    
    if not all_widths:
        print("错误: 未找到有效的车位标注")
        return

    print("\n" + "="*70)
    print("车位尺寸统计分析结果")
    print("="*70)
    
    # 基本统计
    print(f"总车位数量: {len(all_widths)}")
    print(f"总图像数量: {len(per_image_stats)}")
    print("\n类别分布:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  类别 {class_id}: {count} 个 ({count/len(all_widths)*100:.1f}%)")
    
    print("\n尺寸统计 (像素):")
    print(f"  宽度 - 平均值: {np.mean(all_widths):.2f} ± {np.std(all_widths):.2f}")
    print(f"        最小值: {np.min(all_widths):.2f}, 最大值: {np.max(all_widths):.2f}")
    print(f"  高度 - 平均值: {np.mean(all_heights):.2f} ± {np.std(all_heights):.2f}")
    print(f"        最小值: {np.min(all_heights):.2f}, 最大值: {np.max(all_heights):.2f}")
    print(f"  面积 - 平均值: {np.mean(all_areas):.2f} ± {np.std(all_areas):.2f}")
    print(f"  宽高比 - 平均值: {np.mean(all_aspect_ratios):.2f} ± {np.std(all_aspect_ratios):.2f}")
    
    # 百分位数统计
    print("\n宽度百分位数 (像素):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        percentile = np.percentile(all_widths, p)
        print(f"  {p}%: {percentile:.2f}")
    
    print("\n高度百分位数 (像素):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        percentile = np.percentile(all_heights, p)
        print(f"  {p}%: {percentile:.2f}")
    
    # ==================== 距离阈值建议 ====================
    print("\n" + "="*70)
    print("距离阈值 (distance_threshold) 建议")
    print("="*70)
    
    avg_width = np.mean(all_widths)
    avg_height = np.mean(all_heights)
    
    # 基于车位尺寸的建议阈值
    suggestions = {
        '保守阈值 (严格)': avg_width * 0.3,  # 车位宽度的30%
        '适中阈值 (推荐)': avg_width * 0.5,  # 车位宽度的50%
        '宽松阈值 (容错)': avg_width * 0.8,  # 车位宽度的80%
        '基于高度': avg_height * 0.6,       # 车位高度的60%
        '基于对角线': np.sqrt(avg_width**2 + avg_height**2) * 0.3,  # 对角线长度的30%
    }
    
    print("基于车位平均尺寸的建议阈值 (像素):")
    for name, value in suggestions.items():
        print(f"  {name}: {value:.1f}")
    
    # 基于实际分布的建议
    width_95th = np.percentile(all_widths, 95)
    height_95th = np.percentile(all_heights, 95)
    
    print(f"\n基于分布的建议:")
    print(f"  (宽度95%分位数): {width_95th:.1f} 像素")
    print(f"  (高度95%分位数): {height_95th:.1f} 像素")
    print(f"  综合建议阈值: {np.mean([width_95th*0.5, height_95th*0.8]):.1f} 像素")
    
    print(f"\n生成可视化图表...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 宽度分布
    axes[0, 0].hist(all_widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(avg_width, color='red', linestyle='--', label=f'平均: {avg_width:.1f}px')
    axes[0, 0].set_xlabel('宽度 (像素)')
    axes[0, 0].set_ylabel('数量')
    axes[0, 0].set_title('车位宽度分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 高度分布
    axes[0, 1].hist(all_heights, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(all_heights), color='red', linestyle='--', label=f'平均: {np.mean(all_heights):.1f}px')
    axes[0, 1].set_xlabel('高度 (像素)')
    axes[0, 1].set_ylabel('数量')
    axes[0, 1].set_title('车位高度分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 面积分布
    axes[0, 2].hist(all_areas, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 2].axvline(np.mean(all_areas), color='red', linestyle='--', label=f'平均: {np.mean(all_areas):.1f}px^2')
    axes[0, 2].set_xlabel('面积 (像素^2)')
    axes[0, 2].set_ylabel('数量')
    axes[0, 2].set_title('车位面积分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 宽高比分布
    axes[1, 0].hist(all_aspect_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(np.mean(all_aspect_ratios), color='red', linestyle='--', label=f'平均: {np.mean(all_aspect_ratios):.2f}')
    axes[1, 0].set_xlabel('宽高比 (宽度/高度)')
    axes[1, 0].set_ylabel('数量')
    axes[1, 0].set_title('车位宽高比分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 宽度vs高度散点图
    axes[1, 1].scatter(all_widths, all_heights, alpha=0.5, s=10)
    axes[1, 1].set_xlabel('宽度 (像素)')
    axes[1, 1].set_ylabel('高度 (像素)')
    axes[1, 1].set_title('车位宽度 vs 高度')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 每图像车位数量分布
    slot_counts = [stat['slot_count'] for stat in per_image_stats]
    axes[1, 2].hist(slot_counts, bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[1, 2].axvline(np.mean(slot_counts), color='red', linestyle='--', label=f'平均: {np.mean(slot_counts):.1f}')
    axes[1, 2].set_xlabel('每图像车位数量')
    axes[1, 2].set_ylabel('图像数量')
    axes[1, 2].set_title('每图像车位数量分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slot_size_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 阈值建议图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(suggestions))
    

    bars = ax.bar(x_pos, list(suggestions.values()), alpha=0.7, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'])
    
    for i, (bar, (name, value)) in enumerate(zip(bars, suggestions.items())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value:.1f}', ha='center', va='bottom')
    
    ax.set_xlabel('阈值类型')
    ax.set_ylabel('建议阈值 (像素)')
    ax.set_title('距离阈值 (distance_threshold) 建议')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.replace(' (', '\n(') for name in suggestions.keys()], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.axhline(y=avg_width, color='red', linestyle='--', alpha=0.5, label=f'平均宽度: {avg_width:.1f}px')
    ax.axhline(y=np.mean(all_heights), color='green', linestyle='--', alpha=0.5, label=f'平均高度: {np.mean(all_heights):.1f}px')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_suggestions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    stats_data = {
        'summary': {
            'total_slots': len(all_widths),
            'total_images': len(per_image_stats),
            'class_distribution': class_counts,
            'width_stats': {
                'mean': float(np.mean(all_widths)),
                'std': float(np.std(all_widths)),
                'min': float(np.min(all_widths)),
                'max': float(np.max(all_widths)),
                'percentiles': {p: float(np.percentile(all_widths, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
            },
            'height_stats': {
                'mean': float(np.mean(all_heights)),
                'std': float(np.std(all_heights)),
                'min': float(np.min(all_heights)),
                'max': float(np.max(all_heights)),
                'percentiles': {p: float(np.percentile(all_heights, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
            },
            'area_stats': {
                'mean': float(np.mean(all_areas)),
                'std': float(np.std(all_areas)),
                'min': float(np.min(all_areas)),
                'max': float(np.max(all_areas))
            },
            'aspect_ratio_stats': {
                'mean': float(np.mean(all_aspect_ratios)),
                'std': float(np.std(all_aspect_ratios)),
                'min': float(np.min(all_aspect_ratios)),
                'max': float(np.max(all_aspect_ratios))
            }
        },
        'threshold_suggestions': {k: float(v) for k, v in suggestions.items()},
        'per_image_stats': per_image_stats
    }
    
    with open(os.path.join(output_dir, 'slot_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("车位尺寸统计分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. 数据集概览\n")
        f.write(f"   总车位数量: {len(all_widths)}\n")
        f.write(f"   总图像数量: {len(per_image_stats)}\n")
        f.write(f"   平均每图像车位数: {np.mean([s['slot_count'] for s in per_image_stats]):.1f}\n\n")
        
        f.write("2. 尺寸统计 (像素)\n")
        f.write(f"   宽度 - 平均值: {np.mean(all_widths):.2f} ± {np.std(all_widths):.2f}\n")
        f.write(f"           范围: [{np.min(all_widths):.2f}, {np.max(all_widths):.2f}]\n")
        f.write(f"   高度 - 平均值: {np.mean(all_heights):.2f} ± {np.std(all_heights):.2f}\n")
        f.write(f"           范围: [{np.min(all_heights):.2f}, {np.max(all_heights):.2f}]\n")
        f.write(f"   面积 - 平均值: {np.mean(all_areas):.0f} ± {np.std(all_areas):.0f}\n")
        f.write(f"   宽高比 - 平均值: {np.mean(all_aspect_ratios):.2f} ± {np.std(all_aspect_ratios):.2f}\n\n")
        
        f.write("3. 距离阈值建议\n")
        f.write("   基于车位尺寸的建议阈值 (像素):\n")
        for name, value in suggestions.items():
            f.write(f"     {name}: {value:.1f}\n")
        
        f.write(f"\n   推荐设置:\n")
        f.write(f"     • 严格评估: {suggestions['保守阈值 (严格)']:.1f} 像素\n")
        f.write(f"     • 平衡评估: {suggestions['适中阈值 (推荐)']:.1f} 像素\n")
        f.write(f"     • 宽松评估: {suggestions['宽松阈值 (容错)']:.1f} 像素\n\n")
        
        f.write("4. 在评估程序中的使用\n")
        f.write("   在您的补全评估程序中，建议将 distance_threshold 设置为:\n")
        recommended = suggestions['适中阈值 (推荐)']
        f.write(f"   distance_threshold = {recommended:.1f}  # {recommended/avg_width*100:.0f}% 的平均车位宽度\n\n")
        
        f.write("5. 文件说明\n")
        f.write(f"   • 统计图表: {output_dir}/slot_size_distributions.png\n")
        f.write(f"   • 阈值建议图: {output_dir}/threshold_suggestions.png\n")
        f.write(f"   • 详细数据: {output_dir}/slot_statistics.json\n")
        f.write("="*70 + "\n")
    
    print(f"\n分析完成！结果已保存至: {output_dir}/")
    print(f"1. 统计图表: {output_dir}/slot_size_distributions.png")
    print(f"2. 阈值建议图: {output_dir}/threshold_suggestions.png")
    print(f"3. 详细数据: {output_dir}/slot_statistics.json")
    print(f"4. 分析报告: {output_dir}/analysis_report.txt")
    
    print("\n" + "="*70)
    print("【最终建议】")
    print("="*70)
    print(f"基于您的数据集分析，建议将 distance_threshold 设置为:")
    print(f"  推荐值: {suggestions['适中阈值 (推荐)']:.1f} 像素")
    print(f"  范围: {suggestions['保守阈值 (严格)']:.1f} - {suggestions['宽松阈值 (容错)']:.1f} 像素")
    print("="*70)
    
    return stats_data


# 使用示例

if __name__ == "__main__":
    dataset_root = "Threshold_setting_judgment_testimage"  # 设置您的数据集路径
    
    # 运行分析
    try:
        results = analyze_parking_slot_sizes(dataset_root)
        
        if results:
            print("\n" + "="*70)
            print("="*70)
            recommended = results['threshold_suggestions']['适中阈值 (推荐)']
            print(f"# 在您的评估程序中，使用以下设置:")
            print(f"evaluator = DistanceBasedEvaluator(")
            print(f"    distance_threshold={recommended:.1f},  # 基于车位尺寸分析的建议值")
            print(f"    iou_threshold=0.25")
            print(f")")
            print("="*70)
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("\n请检查:")
        print("1. 数据集路径是否正确")
        print("2. 是否安装了必要的库: pip install numpy matplotlib opencv-python tqdm")