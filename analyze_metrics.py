import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import traceback
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

# --- ログ設定 ---
LOG_FILE = "error_log.txt"

def log_error(message):
    """エラーログ記録"""
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(message + "\n")

# --- タイプ間の違いを可視化 --- 
def visualize_differences(metrics_df, output_dir):

    # 1. データ型の確認とログ出力
    print("データ型の確認:\n", metrics_df.dtypes)
    
    # 2. 数値型の列のみ抽出
    numeric_columns = metrics_df.select_dtypes(include=['number']).columns
    print("数値型の列:", numeric_columns)

    # 3. 必要な列の選択（Type列を保持しつつ数値列を選択）
    numeric_metrics_df = metrics_df[numeric_columns].copy()
    numeric_metrics_df["Type"] = metrics_df["Type"]

    # 4. 各Typeの平均値を計算
    mean_df = numeric_metrics_df.groupby("Type").mean()

    # 5. 平均値を棒グラフでプロット
    mean_df.plot(kind="bar", figsize=(14, 8))
    plt.title("Average Metrics by Type")
    plt.xlabel("Type")
    plt.ylabel("Average Metric Value")
    plt.grid(axis="y")
    plt.tight_layout()

    # 出力ディレクトリの確認と保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "average_metrics_by_type.png"))
    plt.close()
    print("タイプ間の平均値の棒グラフを保存しました。")

    # 6. 各数値メトリックの箱ひげ図を作成
    for metric in numeric_columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Type", y=metric, data=numeric_metrics_df, palette="Set3")
        plt.title(f"{metric} Comparison by Type")
        plt.xlabel("Type")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot_by_type.png"))
        plt.close()
        print(f"{metric} の箱ひげ図を保存しました。")
       

# --- チェック関数 ---
def analyze_image_quality(image):
    """画像品質チェックを行い指標を計算"""
    results = {}

    # ブラー検出
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    results['Blur'] = laplacian_var

    # エッジ強度解析
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    results['EdgeDensity'] = edge_density

    # 輝度解析
    mean_brightness = np.mean(image)
    brightness_variance = np.var(image)
    results['MeanBrightness'] = mean_brightness
    results['BrightnessVariance'] = brightness_variance

    # ノイズ検出（周波数解析）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    noise_score = np.sum(magnitude_spectrum > 100)
    results['NoiseScore'] = noise_score

    # コントラスト検出
    contrast = np.std(image)
    results['Contrast'] = contrast

    # 周波数解析
    low_frequency = np.mean(np.abs(f_shift[:10, :10]))
    results['LowFrequency'] = low_frequency

    # 色バランス検出
    mean_r = np.mean(image[:, :, 2])  # Rチャンネル
    mean_g = np.mean(image[:, :, 1])  # Gチャンネル
    mean_b = np.mean(image[:, :, 0])  # Bチャンネル
    color_balance = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_b - mean_r))
    results['ColorBalance'] = color_balance

    return results


def save_metrics_summary(metrics_list, output_dir):
    """
    metrics_listからDataFrameを作成し、統計情報を保存
    """
    # metrics_listの内容をDataFrameに変換
    metrics_df = pd.DataFrame([
        {**entry['metrics'], 'Type': entry['type'], 'FileName': entry['filename']}
        for entry in metrics_list
    ])
    # 統計量を計算して保存
    summary = metrics_df.groupby('Type').describe()
    summary_path = os.path.join(output_dir, "metrics_summary.csv")
    summary.to_csv(summary_path)
    print(f"統計情報を {summary_path} に保存しました。")
    return metrics_df

# --- タイプ別処理 --- 
def process_type_directory(type_dir, type_name, metrics_list):
    """タイプ別のディレクトリを処理"""
    for filename in os.listdir(type_dir):
        img_path = os.path.join(type_dir, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(img_path)
            if image is not None:
                resized = cv2.resize(image, (64, 64))  # サイズを統一
                metrics = analyze_image_quality(resized)
                metrics_list.append({'metrics': metrics, 'filename': filename, 'type': type_name})  # 辞書型で格納

# --- T検定 ---
def perform_t_tests_with_full_log(metrics_df, log_file="df_detailed_log.txt"):
    """
    T検定でタイプ間の統計的な違いを確認（詳細なログを出力）
    メトリック列を対象に、Typeでグループ化し、FileName単位で比較。
    """
    metric_columns = metrics_df.columns.difference(['Type', 'FileName'])
    print(f"対象メトリック列: {metric_columns}")

    results = {}
    types = metrics_df["Type"].unique()

    # ログファイルの初期化と操作をスコープ内にまとめる
    with open(log_file, 'w') as log:
        log.write("T検定 詳細ログ\n")
        log.write("=" * 50 + "\n")

        for i, type1 in enumerate(types):
            for type2 in types[i + 1:]:
                results[f"{type1} vs {type2}"] = {}
                for metric in metric_columns:
                    group1 = metrics_df[metrics_df["Type"] == type1][metric].dropna()
                    group2 = metrics_df[metrics_df["Type"] == type2][metric].dropna()

                    if len(group1) < 2 or len(group2) < 2:
                        print(f"Warning: {metric} の {type1} vs {type2} におけるサンプルサイズが不足しています。")
                        results[f"{type1} vs {type2}"][metric] = {
                            "t_stat": None,
                            "p_value": None,
                            "degrees_of_freedom": None
                        }
                        continue

                    # T検定の実施
                    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

                    # 自由度計算
                    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                    n1, n2 = len(group1), len(group2)
                    numerator = (s1 / n1 + s2 / n2) ** 2
                    term1 = ((s1 / n1) ** 2) / (n1 - 1)
                    term2 = ((s2 / n2) ** 2) / (n2 - 1)
                    denominator = term1 + term2
                    df = numerator / denominator

                    # 結果を保存
                    results[f"{type1} vs {type2}"][metric] = {
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "degrees_of_freedom": df
                    }

                    # ログに記録
                    log.write(f"Comparison: {type1} vs {type2}\n")
                    log.write(f"Metric: {metric}\n")
                    log.write(f"Sample sizes: n1 = {n1}, n2 = {n2}\n")
                    log.write(f"T-statistic: {t_stat:.5f}, p-value: {p_value:.5e}\n")
                    log.write(f"Degrees of Freedom: {df:.5f}\n")
                    log.write("=" * 50 + "\n")

    print(f"T検定の詳細ログを {log_file} に保存しました。")
    return results


def display_images_with_metrics(image_paths, metrics, output_dir, title="Images with Metrics"):
    """画像とその指標を表示・保存"""
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    for ax, img_path, metric in zip(axes, image_paths, metrics):
        # 画像を読み込み、エラーが発生しないようにチェック
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found or cannot be read: {img_path}")
            ax.set_title("Image not found")
            ax.axis("off")
            continue
        
        # 画像をプロット
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVのBGRをRGBに変換
        ax.imshow(image)
        data_type = os.path.basename(os.path.dirname(img_path))
        ax.set_title(f"Metric: {metric:.2f} type: {data_type}") # メトリック値とデータタイプを表示
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"{title} を {plot_path} に保存しました。")

    
# --- T検定結果の保存 --- 
def save_t_test_results_with_df(t_test_results, output_dir):
    """T検定結果をCSVに保存（自由度を含む）"""
    flat_results = []
    for comparison, metrics in t_test_results.items():
        for metric, values in metrics.items():
            flat_results.append({
                "Comparison": comparison,
                "Metric": metric,
                "t_stat": values["t_stat"],
                "p_value": values["p_value"],
                "degrees_of_freedom": values["degrees_of_freedom"]
            })

    # データフレームに変換
    results_df = pd.DataFrame(flat_results)

    # CSVに保存
    results_csv_path = os.path.join(output_dir, "t_test_results_with_df.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"T検定結果を {results_csv_path} に保存しました。")

def plot_t_test_results(t_test_results, output_dir):
    """T検定の結果をプロット"""
    flat_results = []
    for comparison, metrics in t_test_results.items():
        for metric, values in metrics.items():
            flat_results.append({
                "Comparison": comparison,
                "Metric": metric,
                "t_stat": values["t_stat"]
            })

    # データフレームに変換
    results_df = pd.DataFrame(flat_results)

    # プロット作成
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Metric", y="t_stat", hue="Comparison", palette="Set2")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)  # 基準線
    plt.title("T-Statistic Comparison Across Metrics")
    plt.xlabel("Metric")
    plt.ylabel("T-Statistic")
    plt.xticks(rotation=45)
    plt.legend(title="Comparison", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # プロットを保存
    plot_path = os.path.join(output_dir, "t_test_results_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"T検定のプロットを {plot_path} に保存しました。")

def plot_p_value_comparison(t_test_results, output_dir):
    """
    p値の比較をプロットし、統計的有意性のしきい値ラインを表示
    """
    flat_results = []
    for comparison, metrics in t_test_results.items():
        for metric, values in metrics.items():
            flat_results.append({
                "Comparison": comparison,
                "Metric": metric,
                "-log10(p_value)": -np.log10(values["p_value"]) if values["p_value"] else None
            })

    # データフレームに変換
    results_df = pd.DataFrame(flat_results)

    # プロット作成
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Metric", y="-log10(p_value)", hue="Comparison", palette="Set2")
    plt.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p = 0.05")  # p = 0.05ライン
    plt.title("P-Value Comparison Across Metrics (-log10 Transformed)")
    plt.xlabel("Metric")
    plt.ylabel("-log10(p_value)")
    plt.xticks(rotation=45)
    plt.legend(title="Comparison", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # プロットを保存
    plot_path = os.path.join(output_dir, "p_value_comparison_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"p値の比較プロットを {plot_path} に保存しました。")


def process_metrics_and_plot(metrics_list, base_directory, output_base_dir):
    """メトリクスごとの特徴的な画像を選定してプロット"""
    # メトリクス名を取得
    metric_keys = list(metrics_list[0]["metrics"].keys())  # メトリック名を取得

    print("メトリクスごとの画像選定とプロットを開始します。")
    print("メトリクス:", metric_keys)
    print("メトリクスリスト:", metrics_list)
    
    for metric in metric_keys:
        # 指定のメトリックに基づいてリストをソート
        sorted_data = sorted(metrics_list, key=lambda x: x["metrics"][metric])

        # 上位3件
        top_images = sorted_data[-3:]

        top_image_paths = [
            os.path.join(base_directory, entry["type"], entry["filename"]) for entry in top_images
        ]

        top_metrics = [entry["metrics"][metric] for entry in top_images]

        # プロット（上位3件）
        display_images_with_metrics(
            image_paths=top_image_paths,
            metrics=top_metrics,
            output_dir=output_base_dir,
            title=f"Top 3 {metric} Images"
        )

        # 下位3件
        bottom_images = sorted_data[:3]
        bottom_image_paths = [
            os.path.join(base_directory, entry["type"], entry["filename"]) for entry in bottom_images
        ]
        bottom_metrics = [entry["metrics"][metric] for entry in bottom_images]

        # プロット（下位3件）
        display_images_with_metrics(
            image_paths=bottom_image_paths,
            metrics=bottom_metrics,
            output_dir=output_base_dir,
            title=f"Bottom 3 {metric} Images"
        )

        # 中央値に近い3件
        median_index = len(sorted_data) // 2
        median_images = sorted_data[max(0, median_index - 1):median_index + 2]
        median_image_paths = [
            os.path.join(base_directory, entry["type"], entry["filename"]) for entry in median_images
        ]
        median_metrics = [entry["metrics"][metric] for entry in median_images]

        # プロット（中央値付近）
        display_images_with_metrics(
            image_paths=median_image_paths,
            metrics=median_metrics,
            output_dir=output_base_dir,
            title=f"Median 3 {metric} Images"
        )

    print("メトリクスごとの画像選定とプロットを完了しました。")


# --- メイン処理 ---
def main():
    # ディレクトリ設定
    base_directory = "./clipped_images"
    output_base_dir = "./plots"
    os.makedirs(output_base_dir, exist_ok=True)

    metrics_list = []

    # 各タイプのディレクトリを処理
    for type_name in ["A", "N", "R"]:
        type_dir = os.path.join(base_directory, type_name)
        if os.path.exists(type_dir):
            process_type_directory(type_dir, type_name, metrics_list)
        else:
            log_error(f"タイプディレクトリが見つかりません: {type_dir}")
    
    # metrics_listが空でなければ処理を続行
    if metrics_list:
        metrics_df = save_metrics_summary(metrics_list, output_base_dir)  # DataFrame作成
        visualize_differences(metrics_df, output_base_dir)                # 平均値や箱ひげ図
        t_test_results = perform_t_tests_with_full_log(metrics_df)        # T検定
        save_t_test_results_with_df(t_test_results, output_base_dir)      # T検定結果の保存
        plot_t_test_results(t_test_results, output_base_dir)              # T検定プロット
        plot_p_value_comparison(t_test_results, output_base_dir)          # p値プロットを追加
        process_metrics_and_plot(metrics_list, base_directory, output_base_dir)  # 特徴画像の選定
    else:
        print("特徴量が抽出されませんでした。処理を終了します。")
        log_error("特徴量が抽出されませんでした。処理を終了します。")

if __name__ == "__main__":
    main()