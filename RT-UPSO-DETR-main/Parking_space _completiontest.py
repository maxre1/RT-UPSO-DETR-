###################################
# 车位补全实验程序
#输出评价指标结果以及对比图片
# 数据集使用yolo格式
####################################

import os
import glob
import cv2
import numpy as np
import warnings
from ultralytics import RTDETR
from tqdm import tqdm
import json
import math

warnings.filterwarnings('ignore')


def load_yolo_labels(txt_path, img_w, img_h, class_mapping=None):

    gt_boxes = []
    if not os.path.exists(txt_path):
        return gt_boxes
         
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: 
            continue
        
        class_id = int(parts[0])
        nx, ny, nw, nh = map(float, parts[1:])
        
        w = nw * img_w
        h = nh * img_h
        cx = nx * img_w
        cy = ny * img_h
        
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        gt_boxes.append({
            'bbox': [x1, y1, x2, y2],
            'center': [cx, cy],
            'class_id': class_id
        })
        
    return gt_boxes


class PerspectiveTopologySystem:
    def __init__(self):
        self.default_vp_y_ratio = -0.5 

    def process(self, frame, det_results, names):
        if not det_results: 
            return [], [], None
        raw_objects = []
        if det_results[0].boxes is not None:
            det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
            det_cls = det_results[0].boxes.cls.cpu().numpy()
            det_conf = det_results[0].boxes.conf.cpu().numpy()

            for i, (box, cls_id, conf) in enumerate(zip(det_boxes, det_cls, det_conf)):
                label = names[int(cls_id)].lower()
                state = 'occupy' if 'occupy' in label else 'empty'
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w/2, y1 + h/2
                
                raw_objects.append({
                    'id': i,
                    'bbox': [x1, y1, x2, y2],
                    'state': state,
                    'conf': float(conf),
                    'center': np.array([cx, cy]),
                    'size': np.array([w, h]),
                    'layer_id': -1,
                    'is_inferred': False
                })

        if len(raw_objects) < 2:
            return raw_objects, [], None

        processed_objs = [obj.copy() for obj in raw_objects]
        processed_objs.sort(key=lambda o: o['center'][1])
        
        rows = []
        if processed_objs:
            curr_row = [processed_objs[0]]
            for obj in processed_objs[1:]:
                if abs(obj['center'][1] - curr_row[0]['center'][1]) < curr_row[0]['size'][1] * 0.6:
                    curr_row.append(obj)
                else:
                    rows.append(curr_row)
                    curr_row = [obj]
            rows.append(curr_row)
        
        if len(rows) < 2:
            return raw_objects, [], None
        
        row_far = rows[0]
        row_near = rows[1]
        
        anchor_lines = []
        matched_near_ids = set()
        
        for n_obj in row_near:
            for f_obj in row_far:
                dx = abs(n_obj['center'][0] - f_obj['center'][0])
                max_w = max(n_obj['size'][0], f_obj['size'][0])
                if dx < max_w * 0.4:
                    anchor_lines.append((n_obj['center'], f_obj['center']))
                    matched_near_ids.add(n_obj['id'])
                    break
        
        img_h, img_w = frame.shape[:2]
        vp = self._estimate_vanishing_point(anchor_lines, img_w, img_h)
        
        completion_boxes = []
        
        for n_obj in row_near:
            if n_obj['id'] not in matched_near_ids:
                valid_ratios = []
                valid_sizes = []
                for start, end in anchor_lines:
                    dist_near = np.linalg.norm(start - vp)
                    dist_far = np.linalg.norm(end - vp)
                    if dist_near > 1e-5: 
                        valid_ratios.append(dist_far / dist_near)
                    
                    for f in row_far:
                        if np.linalg.norm(f['center'] - end) < 5:
                            valid_sizes.append(f['size'])
                            break
                
                avg_ratio = np.mean(valid_ratios) if valid_ratios else 0.7
                if valid_sizes:
                    avg_far_size = np.mean(valid_sizes, axis=0)
                elif row_far:
                    avg_far_size = row_far[0]['size']
                else:
                    avg_far_size = n_obj['size'] * 0.8

                pred_center = vp + (n_obj['center'] - vp) * avg_ratio
                
                new_w, new_h = avg_far_size
                new_x1 = pred_center[0] - new_w/2
                new_y1 = pred_center[1] - new_h/2
                new_x2 = pred_center[0] + new_w/2
                new_y2 = pred_center[1] + new_h/2
                
                completion_boxes.append({
                    'bbox': [new_x1, new_y1, new_x2, new_y2],
                    'center': pred_center,
                    'size': np.array([new_w, new_h]),
                    'is_inferred': True,
                    'confidence': 0.5
                })

        return raw_objects, completion_boxes, vp

    def _estimate_vanishing_point(self, lines, img_w, img_h):
        if len(lines) == 0:
            return np.array([img_w / 2, img_h * self.default_vp_y_ratio])
        elif len(lines) == 1:
            p1, p2 = lines[0]
            if abs(p2[0] - p1[0]) < 1e-5: 
                return np.array([p1[0], -500])
            k = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - k * p1[0]
            return np.array([img_w / 2, k * (img_w / 2) + b])
        else:
            A_list = []
            C_list = []
            for p1, p2 in lines:
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = A * p1[0] + B * p1[1]
                A_list.append([A, B])
                C_list.append([C])
            try:
                vp = np.linalg.lstsq(A_list, C_list, rcond=None)[0]
                return vp.flatten()
            except:
                return np.array([img_w / 2, -500])


class DistanceBasedEvaluator:
    def __init__(self, distance_threshold=50.0, iou_threshold=0.1):
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou
    
    def calculate_center_distance(self, box1, box2):
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        
        distance = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return distance
    
    def match_boxes_by_distance(self, gt_boxes, comp_boxes):
        matches = []  
        matched_gt_indices = set()
        matched_comp_indices = set()
        
        distance_matrix = []
        for i, gt_box in enumerate(gt_boxes):
            distances = []
            for j, comp_box in enumerate(comp_boxes):
                dist = self.calculate_center_distance(gt_box['bbox'], comp_box['bbox'])
                distances.append(dist)
            distance_matrix.append(distances)
        
        while True:
            min_dist = float('inf')
            best_gt_idx = -1
            best_comp_idx = -1
            
            for i in range(len(gt_boxes)):
                if i in matched_gt_indices:
                    continue
                for j in range(len(comp_boxes)):
                    if j in matched_comp_indices:
                        continue
                    dist = distance_matrix[i][j]
                    if dist < min_dist:
                        min_dist = dist
                        best_gt_idx = i
                        best_comp_idx = j
            
            if best_gt_idx == -1 or best_comp_idx == -1 or min_dist > self.distance_threshold:
                break
            matches.append((best_gt_idx, best_comp_idx, min_dist))
            matched_gt_indices.add(best_gt_idx)
            matched_comp_indices.add(best_comp_idx)
        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt_indices]
        unmatched_comp = [i for i in range(len(comp_boxes)) if i not in matched_comp_indices]
        
        return matches, unmatched_gt, unmatched_comp
    
    def evaluate(self, gt_boxes, comp_boxes, baseline_boxes):
        missed_gt_boxes = []
        
        for gt_box in gt_boxes:
            matched = False
            for base_box in baseline_boxes:
                iou = self.calculate_iou(gt_box['bbox'], base_box['bbox'])
                if iou > self.iou_threshold:
                    matched = True
                    break
            
            if not matched:
                missed_gt_boxes.append(gt_box)
        
        matches, unmatched_gt, unmatched_comp = self.match_boxes_by_distance(
            missed_gt_boxes, comp_boxes
        )
        
        # 3.评价指标
        N_missed = len(missed_gt_boxes)
        N_comp = len(comp_boxes)
        TP = len(matches)
        
        # 漏检召回率 (MRR)
        if N_missed > 0:
            MRR = TP / N_missed
        else:
            MRR = 0.0
        
        # 补全准确率 (CA)
        if N_comp > 0:
            CA = TP / N_comp
        else:
            CA = 0.0
        
        # 平均中心点误差 (MCE)
        if TP > 0:
            center_errors = [dist for _, _, dist in matches]
            MCE = np.mean(center_errors)
        else:
            MCE = 0.0
        
        return {
            'MRR': MRR,
            'CA': CA,
            'MCE': MCE,
            'N_missed': N_missed,
            'N_comp': N_comp,
            'TP': TP,
            'FP': len(unmatched_comp),
            'FN': len(unmatched_gt),
            'matches': matches,
            'missed_gt_boxes': missed_gt_boxes,
            'matched_pairs': matches
        }


def create_visualization(frame, baseline_boxes, comp_boxes, gt_boxes, vp, evaluation_result):
    """
    创建可视化对比图
    """
    h, w = frame.shape[:2]
    

    box_thickness = 4        
    gt_thickness = 4         
    comp_thickness = 5       
    line_thickness = 4       
    title_scale = 1.0        
    label_scale = 0.6        
    left_canvas = frame.copy()
    right_canvas = frame.copy()
    

    cv2.rectangle(left_canvas, (0, 0), (w, 40), (50, 50, 50), -1)
    cv2.putText(left_canvas, "Baseline Detection", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 2)
    
    for box in baseline_boxes:
        x1, y1, x2, y2 = map(int, box['bbox'])
        state = box['state']
        
        if state == 'occupy':
            color = (0, 0, 255)  
            label = f"Occ {box['conf']:.2f}"
        else:
            color = (0, 255, 0)  
            label = f"Emp {box['conf']:.2f}"
        

        cv2.rectangle(left_canvas, (x1, y1), (x2, y2), color, box_thickness)
        

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, line_thickness)
        cv2.rectangle(left_canvas, (x1, y1 - th - 5), (x1 + tw, y1), color, -1) # 文字背景
        
        cv2.putText(left_canvas, label, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), line_thickness)
    
    cv2.rectangle(right_canvas, (0, 0), (w, 40), (50, 50, 50), -1)
    cv2.putText(right_canvas, "Completion + GT (Blue)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 2)
    
    for gt_box in gt_boxes:
        x1, y1, x2, y2 = map(int, gt_box['bbox'])
        cv2.rectangle(right_canvas, (x1, y1), (x2, y2), (255, 0, 0), gt_thickness) 
        cv2.putText(right_canvas, "GT", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 0, 0), line_thickness)
    
    for box in baseline_boxes:
        x1, y1, x2, y2 = map(int, box['bbox'])
        if box['state'] == 'occupy':
            color = (0, 0, 150)  
        else:
            color = (0, 150, 0)  
        cv2.rectangle(right_canvas, (x1, y1), (x2, y2), color, 4)
    
    for comp_box in comp_boxes:
        x1, y1, x2, y2 = map(int, comp_box['bbox'])
        cv2.rectangle(right_canvas, (x1, y1), (x2, y2), (0, 255, 255), comp_thickness) 
        
        label = "Comp"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, line_thickness)
        cv2.rectangle(right_canvas, (x1, y1 - th - 5), (x1 + tw, y1), (0, 255, 255), -1) 
        
        cv2.putText(right_canvas, label, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), line_thickness) 
    
    if vp is not None:
        vp_x, vp_y = int(vp[0]), int(vp[1])
        if -2000 < vp_x < 4000 and -2000 < vp_y < 4000:
            if 0 <= vp_x < w and 0 <= vp_y < h:
                cv2.circle(right_canvas, (vp_x, vp_y), 8, (0, 255, 255), -1)
                cv2.putText(right_canvas, "VP", (vp_x+10, vp_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 255, 255), line_thickness)
    

    matches = evaluation_result.get('matches', [])
    missed_gt_boxes = evaluation_result.get('missed_gt_boxes', [])
    
    for gt_idx, comp_idx, distance in matches:
        if gt_idx < len(missed_gt_boxes) and comp_idx < len(comp_boxes):
            gt_center = missed_gt_boxes[gt_idx]['center']
            comp_center = comp_boxes[comp_idx]['center']
            
            cv2.line(right_canvas, 
                     (int(gt_center[0]), int(gt_center[1])),
                     (int(comp_center[0]), int(comp_center[1])),
                     (0, 255, 0), line_thickness)  # 绿色连线
    
    metrics_text = f"MRR: {evaluation_result['MRR']:.3f}  CA: {evaluation_result['CA']:.3f}  MCE: {evaluation_result['MCE']:.1f}px"
    cv2.putText(right_canvas, metrics_text, (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), line_thickness)
    
    combined = np.hstack([left_canvas, right_canvas])
    
    return combined


def evaluate_completion_module():
    """
    车位补全模块评估主程序
    """
    model_path = 'Parking_space _completion.pt'
    

    dataset_root = "Threshold_setting_judgment_test"
    img_dir = os.path.join(dataset_root, "image")  
    label_dir = os.path.join(dataset_root, "label")  
    

    output_dir = "completion_evaluation_results"
    visualization_dir = os.path.join(output_dir, "visualizations")
    metrics_dir = os.path.join(output_dir, "metrics")
    

    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    

    print("=" * 70)
    print("车位补全模块评估程序")
    print("=" * 70)
    
    print(f"加载模型: {model_path}")
    model = RTDETR(model_path)
    system = PerspectiveTopologySystem()
    evaluator = DistanceBasedEvaluator(
        distance_threshold=100.0,  # 距离阈值（像素）
        iou_threshold=0.1         # IoU阈值
    )
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    
    if not img_paths:
        print(f"错误: 在 {img_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(img_paths)} 张测试图像")

    all_results = []
    summary_stats = {
        'total_images': 0,
        'total_gt': 0,
        'total_missed': 0,
        'total_comp': 0,
        'total_tp': 0,
        'mrr_values': [],
        'ca_values': [],
        'mce_values': []
    }
    
    print("\n开始评估...")
    for img_idx, img_path in enumerate(tqdm(img_paths, desc="处理图像")):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
        
        h, w = frame.shape[:2]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        gt_boxes = load_yolo_labels(label_path, w, h)
        results = model(frame, conf=0.25, verbose=False)
        names = results[0].names if results else {0: 'empty', 1: 'occupy'}
        baseline_boxes, completion_boxes, vp = system.process(frame, results, names)
        eval_result = evaluator.evaluate(gt_boxes, completion_boxes, baseline_boxes)
        eval_result['image_name'] = img_name
        eval_result['gt_count'] = len(gt_boxes)
        eval_result['baseline_count'] = len(baseline_boxes)
        eval_result['completion_count'] = len(completion_boxes)       
        all_results.append(eval_result)
        summary_stats['total_images'] += 1
        summary_stats['total_gt'] += len(gt_boxes)
        summary_stats['total_missed'] += eval_result['N_missed']
        summary_stats['total_comp'] += eval_result['N_comp']
        summary_stats['total_tp'] += eval_result['TP']
        
        if eval_result['N_missed'] > 0:
            summary_stats['mrr_values'].append(eval_result['MRR'])
        if eval_result['N_comp'] > 0:
            summary_stats['ca_values'].append(eval_result['CA'])
        if eval_result['TP'] > 0:
            summary_stats['mce_values'].append(eval_result['MCE'])
        

        if completion_boxes:  
            vis_img = create_visualization(
                frame, baseline_boxes, completion_boxes, gt_boxes, vp, eval_result
            )
            vis_path = os.path.join(visualization_dir, f"{img_name}_result.jpg")
            cv2.imwrite(vis_path, vis_img)
    

    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    
    # 计算总体指标
    if summary_stats['total_missed'] > 0:
        overall_MRR = summary_stats['total_tp'] / summary_stats['total_missed']
    else:
        overall_MRR = 0.0
    
    if summary_stats['total_comp'] > 0:
        overall_CA = summary_stats['total_tp'] / summary_stats['total_comp']
    else:
        overall_CA = 0.0
    
    if len(summary_stats['mce_values']) > 0:
        overall_MCE = np.mean(summary_stats['mce_values'])
        mce_std = np.std(summary_stats['mce_values'])
    else:
        overall_MCE = 0.0
        mce_std = 0.0
    
    mrr_std = np.std(summary_stats['mrr_values']) if summary_stats['mrr_values'] else 0.0
    ca_std = np.std(summary_stats['ca_values']) if summary_stats['ca_values'] else 0.0

    print(f"测试图像总数: {summary_stats['total_images']}")
    print(f"总GT标注数: {summary_stats['total_gt']}")
    print(f"总漏检车位数: {summary_stats['total_missed']}")
    print(f"总补全框数: {summary_stats['total_comp']}")
    print(f"正确补全数 (TP): {summary_stats['total_tp']}")
    print(f"误补数 (FP): {summary_stats['total_comp'] - summary_stats['total_tp']}")
    print(f"漏补数 (FN): {summary_stats['total_missed'] - summary_stats['total_tp']}")
    print("-" * 70)
    
    print(f"【漏检召回率 (MRR)】: {overall_MRR:.3%} (±{mrr_std:.3f})")
    print(f"【补全准确率 (CA)】: {overall_CA:.3%} (±{ca_std:.3f})")
    print(f"【平均中心点误差 (MCE)】: {overall_MCE:.2f} ± {mce_std:.2f} 像素")
    print("=" * 70)
    

    csv_path = os.path.join(metrics_dir, "detailed_results.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("图像,GT数,基线检测数,补全数,漏检数,TP,FP,FN,MRR,CA,MCE(px)\n")
        for result in all_results:
            f.write(f"{result['image_name']},"
                   f"{result['gt_count']},"
                   f"{result['baseline_count']},"
                   f"{result['completion_count']},"
                   f"{result['N_missed']},"
                   f"{result['TP']},"
                   f"{result['FP']},"
                   f"{result['FN']},"
                   f"{result['MRR']:.4f},"
                   f"{result['CA']:.4f},"
                   f"{result['MCE']:.2f}\n")
    

    summary_path = os.path.join(metrics_dir, "summary.json")
    summary_data = {
        'overall_metrics': {
            'MRR': float(overall_MRR),
            'CA': float(overall_CA),
            'MCE': float(overall_MCE),
            'MRR_std': float(mrr_std),
            'CA_std': float(ca_std),
            'MCE_std': float(mce_std)
        },
        'statistics': {
            'total_images': summary_stats['total_images'],
            'total_gt': summary_stats['total_gt'],
            'total_missed': summary_stats['total_missed'],
            'total_comp': summary_stats['total_comp'],
            'total_tp': summary_stats['total_tp'],
            'total_fp': summary_stats['total_comp'] - summary_stats['total_tp'],
            'total_fn': summary_stats['total_missed'] - summary_stats['total_tp']
        },
        'evaluation_settings': {
            'distance_threshold': evaluator.distance_threshold,
            'iou_threshold': evaluator.iou_threshold
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("车位补全模块评估报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. 实验设置\n")
        f.write(f"   测试图像数量: {summary_stats['total_images']}\n")
        f.write(f"   匹配距离阈值: {evaluator.distance_threshold} 像素\n")
        f.write(f"   IoU匹配阈值: {evaluator.iou_threshold}\n\n")
        
        f.write("2. 数据统计\n")
        f.write(f"   总GT标注框数: {summary_stats['total_gt']}\n")
        f.write(f"   总漏检车位数: {summary_stats['total_missed']}\n")
        f.write(f"   总补全框数: {summary_stats['total_comp']}\n")
        f.write(f"   正确补全数 (TP): {summary_stats['total_tp']}\n")
        f.write(f"   误补数 (FP): {summary_stats['total_comp'] - summary_stats['total_tp']}\n")
        f.write(f"   漏补数 (FN): {summary_stats['total_missed'] - summary_stats['total_tp']}\n\n")
        
        f.write("3. 性能指标\n")
        f.write(f"   漏检召回率 (MRR): {overall_MRR:.3%} (±{mrr_std:.3f})\n")
        f.write(f"   补全准确率 (CA):  {overall_CA:.3%} (±{ca_std:.3f})\n")
        f.write(f"   平均中心点误差 (MCE): {overall_MCE:.2f} ± {mce_std:.2f} 像素\n\n")
        
        f.write("4. 结果说明\n")
        f.write("   - 漏检召回率 (MRR): 衡量模块找回漏检车位的能力\n")
        f.write("   - 补全准确率 (CA): 衡量补全框的可靠性\n")
        f.write("   - 平均中心点误差 (MCE): 衡量补全框的位置精度\n\n")
        
        f.write("5. 文件说明\n")
        f.write(f"   - 可视化结果: {visualization_dir}/\n")
        f.write(f"   - 详细数据: {metrics_dir}/detailed_results.csv\n")
        f.write(f"   - 总体统计: {metrics_dir}/summary.json\n")
        f.write("=" * 70 + "\n")

    print(f"\n评估完成！结果已保存至以下位置:")
    print(f"1. 可视化对比图: {visualization_dir}/")
    print(f"2. 详细评估数据: {metrics_dir}/detailed_results.csv")
    print(f"3. 总体统计结果: {metrics_dir}/summary.json")
    print(f"4. 评估报告: {output_dir}/evaluation_report.txt")
    
    return {
        'overall_metrics': summary_data['overall_metrics'],
        'statistics': summary_data['statistics'],
        'all_results': all_results
    }

if __name__ == "__main__":

    results = evaluate_completion_module()
    
