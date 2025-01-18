import os
import cv2
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import traceback

# --- ログ設定 ---
LOG_FILE = "error_log.txt"

def log_error(message):
    """エラーログ記録"""
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(message + "\n")

# --- XML解析 ---
def parse_voc_xml(xml_file, target_class):
    """XMLファイルから対象クラスの領域を抽出"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == target_class:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                objects.append((xmin, ymin, xmax, ymax))
        return objects
    except Exception as e:
        log_error(f"XML解析エラー: {str(e)}\n{traceback.format_exc()}")
        return []

# --- ディレクトリ解析 ---
def analyze_and_save_clips(images_dir, annotations_dir, target_class, label, save_base_dir):
    """画像とアノテーションディレクトリを解析しクリップ画像を保存"""
    save_dir = os.path.join(save_base_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(images_dir, filename)
            xml_path = os.path.join(annotations_dir, os.path.splitext(filename)[0] + '.xml')

            # 画像読み込み
            image = cv2.imread(img_path)
            if image is None:
                log_error(f"画像が読み込めません: {img_path}")
                continue

            # XML解析
            if not os.path.exists(xml_path):
                log_error(f"XMLファイルが見つかりません: {xml_path}")
                continue

            # 対象クラス領域取得
            regions = parse_voc_xml(xml_path, target_class)
            for idx, (xmin, ymin, xmax, ymax) in enumerate(regions):
                roi = image[ymin:ymax, xmin:xmax]
                save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_clip_{idx}.jpg")
                cv2.imwrite(save_path, roi)

# --- メイン処理 ---
def main(base_directory, output_clips_dir, target_class):
    """画像データを処理してクリップ画像を保存"""
    os.makedirs(output_clips_dir, exist_ok=True)

    # ディレクトリ探索とラベル判定
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # タイプ別ラベルを判定
        if '_R' in subdir:
            label = 'R'
        elif '_N' in subdir:
            label = 'N'
        elif '_A' in subdir:
            label = 'A'
        else:
            label = 'test'

        # trainディレクトリ内のパス設定
        images_dir = os.path.join(subdir_path, 'train', 'images')
        annotations_dir = os.path.join(subdir_path, 'train', 'annotations')

        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            log_error(f"ディレクトリが見つかりません: {images_dir} または {annotations_dir}")
            continue

        # クリップ画像保存
        analyze_and_save_clips(images_dir, annotations_dir, target_class, label, output_clips_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定したディレクトリから画像をクリップして保存します。")
    parser.add_argument("--base_directory", type=str, required=True, help="ベースディレクトリへのパス")
    parser.add_argument("--output_clips_dir", type=str, required=True, help="クリップ画像の保存先ディレクトリ")
    parser.add_argument("--target_class", type=str, required=True, help="対象となるクラス名")

    args = parser.parse_args()

    main(args.base_directory, args.output_clips_dir, args.target_class)
