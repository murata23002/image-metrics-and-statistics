#!/bin/bash

# エラーハンドリング
set -e  # エラーが発生したらスクリプトを停止
set -o pipefail  # パイプラインでエラーを伝播

# ディレクトリの設定
BASE_DIR="./dataset"  # データセットのベースディレクトリ
CLIPPED_DIR="./clipped_images"  # クリップ画像の保存先ディレクトリ
PLOTS_DIR="./plots"  # プロット結果の保存先ディレクトリ

# 対象クラス
TARGET_CLASS="Body"

# ステップ1: クリップ画像生成スクリプトの実行
echo "=== ステップ1: クリップ画像の生成を開始します ==="
python3 generate_clips.py --base_directory "$BASE_DIR" \
                         --output_clips_dir "$CLIPPED_DIR" \
                         --target_class "$TARGET_CLASS"
echo "=== ステップ1: クリップ画像の生成が完了しました ==="

# ステップ2: PCAとt-SNEプロットスクリプトの実行
echo "=== ステップ2: PCAを開始します ==="
python3 pca_tsne_plot.py --clips_dir "$CLIPPED_DIR" \
                        --output_dir "$PLOTS_DIR"
echo "=== ステップ2: PCAが完了しました ==="

# ステップ3: メトリクス分析と可視化スクリプトの実行
echo "=== ステップ3: メトリクス分析と可視化を開始します ==="
python3 analyze_metrics.py --base_directory "$CLIPPED_DIR" \
                          --output_dir "$PLOTS_DIR"
echo "=== ステップ3: メトリクス分析と可視化が完了しました ==="

echo "=== 全ての処理が完了しました ==="
