import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --- クリップ画像の読み込み ---
def load_clips(base_dir):
    """クリップ画像をロードして特徴量とラベルを抽出"""
    features = []
    labels = []
    for label_dir in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_dir)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(label_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # サイズを統一（例: 64x64）し、1次元ベクトルに変換
                    resized = cv2.resize(image, (64, 64))
                    features.append(resized.flatten())
                    labels.append(label_dir)
    return np.array(features), np.array(labels)

# --- PCAプロット ---
def plot_pca(features, labels, output_path, components=(1, 2)):
    """PCAで指定した主成分を次元削減してプロット"""
    
    label_names = []
    for label in labels:
        if label == "A":
            label_names.append("Anomaly")
        elif label == "N":
            label_names.append("Normal")
        else:
            label_names.append("Random")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA()
    pca_result = pca.fit_transform(scaled_features)

    component_x = components[0] - 1  # 主成分は1から始まるが配列は0から始まる
    component_y = components[1] - 1

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, component_x], y=pca_result[:, component_y], hue=label_names, palette='Set1', legend="full")
    plt.title(f"PCA Visualization (PC{components[0]} vs PC{components[1]})")
    plt.xlabel(f"PC{components[0]}")
    plt.ylabel(f"PC{components[1]}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- PCA分散説明率プロット ---
def plot_pca_variance(features, output_path):
    """PCAの分散説明率をプロット"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA()
    pca.fit(scaled_features)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- t-SNEプロット ---
def plot_tsne(features, labels, output_path):
    """t-SNEで次元削減してプロット"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='Set1', legend="full")
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- t-SNEの分散具合確認 ---
def plot_tsne_variance(features, output_path):
    """t-SNEの分散具合をプロット"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)

    # t-SNEは直接的な分散説明率を提供しないため、次元ごとの分布を確認
    tsne_var = np.var(tsne_result, axis=0)
    tsne_var_ratio = tsne_var / np.sum(tsne_var)

    plt.figure(figsize=(8, 5))
    plt.bar(["Dimension 1", "Dimension 2"], tsne_var_ratio, color=['blue', 'orange'])
    plt.title("t-SNE Variance Ratio")
    plt.ylabel("Variance Ratio")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- メイン処理 ---
def main():
    clips_dir = "./clipped_images"  # クリップ画像のディレクトリ
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)

    # 特徴量とラベルの読み込み
    features, labels = load_clips(clips_dir)

    if len(features) == 0:
        print("特徴量が抽出されませんでした。処理を終了します。")
        return

    # プロット
    plot_pca(features, labels, os.path.join(output_dir, "pca_plot_1_2.png"), components=(1, 2))
    plot_pca(features, labels, os.path.join(output_dir, "pca_plot_3_4.png"), components=(3, 4))
    #plot_tsne(features, labels, os.path.join(output_dir, "tsne_plot.png"))

    print("プロットが完了しました。")


if __name__ == "__main__":
    main()
