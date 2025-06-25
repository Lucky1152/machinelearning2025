import os
import cv2
import pandas as pd
from paddleocr import TextRecognition
import torch 
import time
from PIL import Image

RECOGNITION_MODEL_DIR = './models/paddle' 
RECOGNITION_MODEL_NAME = "PP-OCRv5_server_rec"
RECOGNITION_DICT_NAME = "ppocr_keys_v1.txt"
RECOGNITION_DICT_PATH = os.path.join(RECOGNITION_MODEL_DIR, RECOGNITION_DICT_NAME)

INPUT_IMAGE_DIR = 'input_image'  # 存放预先裁剪好的图片
OUTPUT_CSV_PATH = 'results.csv'
OUTPUT_VIS_DIR = 'output'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
C_DIR = 'E:/'
RAW_DATA_DIR = os.path.join(C_DIR, 'Test2')
CSV_FILE = os.path.join(RAW_DATA_DIR, 'labels.csv')
IMAGE_DIR = RAW_DATA_DIR
CROPPED_IMAGE_DIR = os.path.join(BASE_DIR, 'input_image')
os.makedirs(CROPPED_IMAGE_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 


def preprocess_data(csv_path, raw_image_dir, cropped_image_dir):

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: CSV文件未找到 {csv_path}")
        return

    print(f"开始从 {raw_image_dir} 预处理和裁剪图片...")
    for index, row in df.iterrows():
        filename = row['filename']
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        original_image_path = os.path.join(raw_image_dir, filename)
        # 统一保存为jpg格式，方便后续处理
        cropped_image_filename = f"{os.path.splitext(filename)[0]}.jpg"
        cropped_image_path_absolute = os.path.join(cropped_image_dir, cropped_image_filename)

        if not os.path.exists(original_image_path):
            print(f"警告: 原始图片未找到 {original_image_path}，跳过。")
            continue

        try:
            with Image.open(original_image_path) as img:
                width, height = img.size
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)

                if xmin >= xmax or ymin >= ymax:
                    print(f"警告: 文件 {filename} 的裁剪区域无效 ({xmin},{ymin},{xmax},{ymax})，已跳过。")
                    continue

                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                # 确保是RGB模式，避免一些格式问题
                if cropped_img.mode != 'RGB':
                    cropped_img = cropped_img.convert('RGB')
                cropped_img.save(cropped_image_path_absolute)
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
    print(f"图片预处理完成，裁剪后的图片已保存到: {cropped_image_dir}")


def initialize_ocr_engine():
    """加载和初始化 PaddleOCR 文本识别模型"""
    print(f"\n正在初始化 PaddleOCR TextRecognition 模型...")
    print(f"  模型目录: {RECOGNITION_MODEL_DIR}")
    print(f"  模型名称: {RECOGNITION_MODEL_NAME}")
    
    try:
        # PaddleOCR 会自动根据环境选择使用CPU或GPU
        ocr_engine = TextRecognition(
            model_dir=RECOGNITION_MODEL_DIR,
            model_name=RECOGNITION_MODEL_NAME
        )
        print("PaddleOCR TextRecognition 模型加载成功。")
        return ocr_engine
    except ImportError:
        print("错误：无法从 'paddleocr' 导入 'TextRecognition'。请检查 PaddleOCR 安装。")
        return None
    except Exception as e:
        print(f"错误: 加载 PaddleOCR TextRecognition 模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_images(ocr_engine, input_dir, output_csv, output_vis_dir):
    """
    直接对输入文件夹中的图片进行OCR识别。
    假设input_dir中的每张图片都已经是裁剪好的、只包含待识别文本的区域。
    """
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

    print(f"找到 {len(image_files)} 张已裁剪的图片进行处理...")

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        print(f"\n--- 处理图片: {filename} ---")
        
        # 使用cv2读取图片，因为PaddleOCR接受numpy数组
        img_to_process = cv2.imread(image_path)
        if img_to_process is None:
            print(f"警告: 无法读取图片 {image_path}，跳过。")
            results_data.append({
                'id': os.path.splitext(filename)[0],
                'number': 'Error - Could not read image'
            })
            continue

        img_for_vis = img_to_process.copy()
        detected_reading = "N/A"

        try:
            # 直接将整张已裁剪的图片送入OCR引擎
            ocr_pred_results = ocr_engine.predict(input=img_to_process)
            
            detected_reading = "N/A - No text recognized"
            highest_confidence_text = ""
            current_highest_score = 0.0

            if ocr_pred_results and isinstance(ocr_pred_results, list) and len(ocr_pred_results) > 0:
                # 遍历所有识别出的文本片段，找到置信度最高的那个
                for result_dict in ocr_pred_results:
                    if isinstance(result_dict, dict) and 'rec_text' in result_dict and 'rec_score' in result_dict:
                        text_candidate = result_dict['rec_text']
                        score_candidate = result_dict['rec_score']
                        
                        print(f"  候选文本: '{text_candidate}', 置信度: {score_candidate:.4f}")
                        
                        if isinstance(text_candidate, str) and isinstance(score_candidate, (float, int)):
                            if text_candidate and score_candidate > current_highest_score:
                                current_highest_score = score_candidate
                                highest_confidence_text = text_candidate
                
                if highest_confidence_text:
                    detected_reading = highest_confidence_text.strip()
                    print(f"最终识别结果: '{detected_reading}' (置信度: {current_highest_score:.4f})")
                else:
                    print("所有候选文本为空或置信度过低。")
                    detected_reading = "N/A - Low confidence or empty"
            else:
                 print(f"TextRecognition predict() 返回结果为空或格式不正确: {ocr_pred_results}")
                 detected_reading = "N/A - Rec format error"

        except Exception as e:
            import traceback
            print(f"错误: 图片 '{filename}' 的度数识别失败。")
            traceback.print_exc()
            detected_reading = "Error - Recognition failed"

        # 对识别出的纯数字字符串进行格式化（例如 "12345" -> "1234.5"）
        formatted_reading = detected_reading  
        if isinstance(detected_reading, str) and detected_reading.isdigit() and detected_reading:
            cleaned_str =detected_reading
            if len(cleaned_str) > 1:
                formatted_reading = f"{cleaned_str[:-1]}.{cleaned_str[-1]}"
            else:
                formatted_reading = f"0.{cleaned_str}"
        
        print(f"原始识别读数: '{detected_reading}' -> 格式化后: '{formatted_reading}'") 
        
        # 将最终结果写入可视化图片
        cv2.putText(img_for_vis, f"Result: {formatted_reading}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    print("开始电表读数识别流程 (仅使用PaddleOCR)...")
    preprocess_data(CSV_FILE, IMAGE_DIR, CROPPED_IMAGE_DIR)

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
    
    paddle_ocr_engine = initialize_ocr_engine()

    if paddle_ocr_engine:
        process_images(paddle_ocr_engine, INPUT_IMAGE_DIR, OUTPUT_CSV_PATH, OUTPUT_VIS_DIR)
    else:
        print("模型初始化失败，程序退出。")

    end_time = time.time()
    print(f"总处理时间: {end_time - start_time:.2f} 秒")