import os
import cv2
import pandas as pd
from ultralytics import YOLO
from paddleocr import TextRecognition
import torch
import time

# --- 图像处理和模型配置 ---
DETECTION_MODEL_PATH = './models/best.pt'

RECOGNITION_MODEL_DIR = './models/paddle' 
RECOGNITION_MODEL_NAME = "PP-OCRv5_server_rec"
RECOGNITION_DICT_NAME = "ppocr_keys_v1.txt" # 
RECOGNITION_DICT_PATH = os.path.join(RECOGNITION_MODEL_DIR, RECOGNITION_DICT_NAME)



INPUT_IMAGE_DIR = 'C:/Users/Lucky/Desktop/wanz/input_image'
OUTPUT_CSV_PATH = 'results.csv'
OUTPUT_VIS_DIR = 'output'


DETECTION_CONFIDENCE_THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_models():
    """加载和初始化所有模型"""
    print(f"使用设备 (YOLO): {DEVICE.upper()}")

    try:
        detector = YOLO(DETECTION_MODEL_PATH)
        detector.to(DEVICE)
        print(f"YOLOv8 检测模型 '{DETECTION_MODEL_PATH}' 加载成功。")
    except Exception as e:
        print(f"错误: 加载 YOLOv8 检测模型失败: {e}")
        return None, None


    print(f"\n正在初始化 PaddleOCR TextRecognition 模型...")
    print(f"  模型目录: {RECOGNITION_MODEL_DIR}")
    print(f"  模型名称: {RECOGNITION_MODEL_NAME}")
    print(f"  预期字典: {RECOGNITION_DICT_PATH} (将由TextRecognition隐式加载)")



    try:
        ocr_engine = TextRecognition(
            model_dir=RECOGNITION_MODEL_DIR,
            model_name=RECOGNITION_MODEL_NAME

        )
        print("PaddleOCR TextRecognition 模型加载成功。")
    except ImportError:
        print("错误：无法从 'paddleocr' 导入 'TextRecognition'。请检查 PaddleOCR 安装。")
        return detector, None
    except Exception as e:
        print(f"错误: 加载 PaddleOCR TextRecognition 模型失败: {e}")
        import traceback
        traceback.print_exc()
        return detector, None

    return detector, ocr_engine

def process_images(detector, ocr_engine, input_dir, output_csv, output_vis_dir):
    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹 '{input_dir}' 未找到。")
        return

    os.makedirs(output_vis_dir, exist_ok=True)
    print(f"可视化结果将保存在: {output_vis_dir}")
    print(f"CSV结果将保存在: {output_csv}")

    results_data = []
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    if not image_files:
        print(f"在 '{input_dir}' 中未找到图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片进行处理...")

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        print(f"\n--- 处理图片: {filename} ---")
        img_original = cv2.imread(image_path)
        if img_original is None:
            print(f"警告: 无法读取图片 {image_path}，跳过。")
            results_data.append({
                'filename': filename,
                'recognized_degree': 'Error - Could not read image',
                'reading_area_coordinates': 'N/A'
            })
            continue

        img_for_vis = img_original.copy()

        try:
            detection_results = detector.predict(source=img_original, verbose=False, conf=DETECTION_CONFIDENCE_THRESHOLD)
        except Exception as e:
            print(f"错误: 图片 '{filename}' 的区域检测失败: {e}")
            results_data.append({
                'filename': filename,
                'recognized_degree': 'Error - Detection failed',
                'reading_area_coordinates': 'N/A'
            })
            cv2.imwrite(os.path.join(output_vis_dir, f"error_det_{filename}"), img_for_vis)
            continue

        detected_reading = "N/A"
        reading_coords_str = "N/A"
        best_box_coords = None

        if detection_results and detection_results[0].boxes:
            boxes = detection_results[0].boxes.xyxy
            confs = detection_results[0].boxes.conf
            clss = detection_results[0].boxes.cls
            highest_conf = 0
            
            for i in range(len(boxes)):
                if int(clss[i]) == 0 and confs[i] > highest_conf: 
                    highest_conf = confs[i]
                    best_box_coords = boxes[i].cpu().numpy().astype(int)

            if best_box_coords is not None:
                x1, y1, x2, y2 = best_box_coords
                reading_coords_str = f"({x1},{y1},{x2},{y2})"
                print(f"检测到读数区域坐标: {reading_coords_str} (置信度: {highest_conf:.2f})")
                cv2.rectangle(img_for_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_for_vis, f"ReadingArea: {highest_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                padding = 0
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(img_original.shape[1], x2 + padding)
                crop_y2 = min(img_original.shape[0], y2 + padding)

                if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                    print("警告: 裁剪区域无效，跳过识别。")
                    detected_reading = "Error - Invalid crop"
                else:
                    cropped_img = img_original[crop_y1:crop_y2, crop_x1:crop_x2]
                    # cv2.imwrite(os.path.join(output_vis_dir, f"crop_{filename}"), cropped_img) 

                    try:

                        ocr_pred_results = ocr_engine.predict(input=cropped_img)
                        detected_reading = "N/A - No text recognized"
                        highest_confidence_text = ""
                        current_highest_score = 0.0

                        if ocr_pred_results and isinstance(ocr_pred_results, list):
                            if len(ocr_pred_results) == 0:
                                print("TextRecognition predict() 返回空列表。")
                            
                            # 遍历列表中的每个结果字典
                            for result_dict in ocr_pred_results:
                                if isinstance(result_dict, dict) and \
                                   'rec_text' in result_dict and 'rec_score' in result_dict:
                                    
                                    text_candidate = result_dict['rec_text']
                                    score_candidate = result_dict['rec_score']
                                    
                                    print(f"  候选文本: '{text_candidate}', 置信度: {score_candidate:.4f}")
                                    
                                    if isinstance(text_candidate, str) and isinstance(score_candidate, (float, int)):
                                        if text_candidate and score_candidate > current_highest_score:
                                            current_highest_score = score_candidate
                                            highest_confidence_text = text_candidate
                                    else:
                                        print(f"  警告: 候选结果文本 '{text_candidate}' 或分数 {score_candidate} 类型不正确，已跳过。")
                                else:
                                    print(f"  警告: 结果列表中的项目不是预期的字典格式或缺少键: {result_dict}")
                            
                            if highest_confidence_text:
                                detected_reading = highest_confidence_text.strip()
                                print(f"最终识别结果: '{detected_reading}' (置信度: {current_highest_score:.4f})")
                            elif len(ocr_pred_results) > 0 and any(isinstance(item, dict) for item in ocr_pred_results):
                                print("所有候选文本为空或置信度过低，或者未能从结果字典中正确提取。")
                                detected_reading = "N/A - Low confidence or empty"

                        else:
                            print(f"TextRecognition predict() 返回结果格式不正确 (期望列表): {ocr_pred_results}")
                            detected_reading = "N/A - Rec format error"

                        cv2.putText(img_for_vis, f"Reading: {detected_reading}", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    except Exception as e:
                        import traceback
                        print(f"错误: 图片 '{filename}' 的度数识别失败。原始异常信息如下:")
                        traceback.print_exc()
                        detected_reading = "Error - Recognition failed"
                        cv2.putText(img_for_vis, "Reading: Error", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print("未检测到读数区域。")
                cv2.putText(img_for_vis, "No ReadingArea Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            print("YOLO 模型未返回有效检测结果。")
            cv2.putText(img_for_vis, "Detection Failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        formatted_reading = detected_reading  
        if isinstance(detected_reading, str) and detected_reading.isdigit() and detected_reading:
            cleaned_str = str(int(detected_reading))
            if len(cleaned_str) > 1:
                formatted_reading = f"{cleaned_str[:-1]}.{cleaned_str[-1]}"
            else:
                formatted_reading = f"0.{cleaned_str}"
        
        print(f"原始识别读数: '{detected_reading}' -> 格式化后: '{formatted_reading}'") 
        file_id = os.path.splitext(filename)[0]
        results_data.append({
            'id': file_id,
            'number': formatted_reading
        })
        
        vis_output_path = os.path.join(output_vis_dir, filename)
        cv2.imwrite(vis_output_path, img_for_vis)

    if results_data:
        df = pd.DataFrame(results_data)
        try:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n所有图片处理完成。结果已保存到 CSV 文件: {output_csv}")
        except Exception as e:
            print(f"错误: 保存CSV文件 '{output_csv}' 失败: {e}")
    else:
        print("没有处理任何图片，未生成CSV文件。")

if __name__ == '__main__':
    print("开始电表读数识别流程...")
    if not os.path.exists(DETECTION_MODEL_PATH):
        print(f"关键错误: YOLOv8检测模型未找到于 '{DETECTION_MODEL_PATH}'")
        exit()
    if not os.path.isdir(RECOGNITION_MODEL_DIR): 
         print(f"关键错误: PaddleOCR识别模型目录未找到于 '{RECOGNITION_MODEL_DIR}'")
         exit()
    if not os.path.exists(os.path.join(RECOGNITION_MODEL_DIR, RECOGNITION_DICT_NAME)):
        print(f"关键警告: PaddleOCR字典文件 '{RECOGNITION_DICT_NAME}' 未在目录 '{RECOGNITION_MODEL_DIR}' 中找到。")
        print("TextRecognition 可能因此无法正确加载。请确保字典文件存在且名称正确。")
    if not os.path.isdir(INPUT_IMAGE_DIR):
        print(f"关键错误: 输入图片文件夹未找到于 '{INPUT_IMAGE_DIR}'")
        exit()

    start_time = time.time()
    
    yolo_detector, paddle_ocr_engine = initialize_models()

    if yolo_detector and paddle_ocr_engine:
        process_images(yolo_detector, paddle_ocr_engine, INPUT_IMAGE_DIR, OUTPUT_CSV_PATH, OUTPUT_VIS_DIR)
    else:
        print("模型初始化失败，程序退出。")

    end_time = time.time()
    print(f"总处理时间: {end_time - start_time:.2f} 秒")