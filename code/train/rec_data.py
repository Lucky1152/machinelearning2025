import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'Test2')
CSV_FILE = os.path.join(RAW_DATA_DIR, 'labels.csv')
IMAGE_DIR = RAW_DATA_DIR

TRAIN_DATA_ROOT = os.path.join(BASE_DIR, 'train_data')
CROPPED_IMAGE_DIR = os.path.join(TRAIN_DATA_ROOT, 'cropped_images')
TRAIN_LIST_FILE = os.path.join(TRAIN_DATA_ROOT, 'train_list.txt')
VAL_LIST_FILE = os.path.join(TRAIN_DATA_ROOT, 'val_list.txt')
DICT_FILE = os.path.join(TRAIN_DATA_ROOT, 'dict_digit.txt')


os.makedirs(CROPPED_IMAGE_DIR, exist_ok=True)

def create_digit_dict(output_path):
    digits = "0123456789"
    with open(output_path, 'w', encoding='utf-8') as f:
        for digit in digits:
            f.write(digit + '\n')
    print(f"数字字典已创建: {output_path}")

def preprocess_data(csv_path, raw_image_dir, cropped_image_dir, train_list_path, val_list_path, test_size=0.2):

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: CSV文件未找到 {csv_path}")
        return

    processed_labels = []

    for index, row in df.iterrows():
        filename = row['filename']
        number_label_raw = str(row['number'])
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        number_label_digits_only ="".join(filter(str.isdigit, number_label_raw)).zfill(6)
        if not number_label_digits_only:
            print(f"警告: 文件 {filename} 的数字标签 '{number_label_raw}' 处理后为空，已跳过。")
            continue

        original_image_path = os.path.join(raw_image_dir, filename)
        cropped_image_filename = f"{os.path.splitext(filename)[0]}_crop.jpg"
        cropped_image_path_relative = os.path.join('cropped_images', cropped_image_filename) # 相对路径
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
                cropped_img.save(cropped_image_path_absolute)
                processed_labels.append(f"{cropped_image_path_relative}\t{number_label_digits_only}")
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")

    if not processed_labels:
        print("错误: 没有成功处理任何数据，请检查CSV文件和图片路径。")
        return

    train_labels, val_labels = train_test_split(processed_labels, test_size=test_size, random_state=42)

    with open(train_list_path, 'w', encoding='utf-8') as f:
        for line in train_labels:
            f.write(line + '\n')
    print(f"训练标签文件已生成: {train_list_path} (共 {len(train_labels)} 条)")

    with open(val_list_path, 'w', encoding='utf-8') as f:
        for line in val_labels:
            f.write(line + '\n')
    print(f"验证标签文件已生成: {val_list_path} (共 {len(val_labels)} 条)")

if __name__ == "__main__":
    print("开始数据预处理...")
    create_digit_dict(DICT_FILE)
    preprocess_data(CSV_FILE, IMAGE_DIR, CROPPED_IMAGE_DIR, TRAIN_LIST_FILE, VAL_LIST_FILE)
    print("数据预处理完成。")
