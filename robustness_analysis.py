# robustness_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings('ignore')

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
output_dir = "IABT_DW_Robustness_Analysis"
os.makedirs(output_dir, exist_ok=True)


class RobustnessEvaluator:
    def __init__(self, model_params, random_seed=0):
        self.model_params = model_params
        self.model = None
        self.scaler = MinMaxScaler()
        self.random_seed = random_seed
        self.X_test = None
        self.y_test = None
        self.output_dir = output_dir

    def prepare_data(self):
        """准备测试数据"""
        # 读取数据
        data = pd.read_excel("Darknet_Market_processed_filtered.xlsx")
        x = data.drop('label', axis=1)
        y = data['label']

        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.4, random_state=self.random_seed, stratify=y
        )

        # 归一化处理
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_test_normalized = self.scaler.transform(X_test)

        # SMOTE采样
        smote = SMOTE(random_state=self.random_seed)
        X_res, y_res = smote.fit_resample(X_train_normalized, y_train)

        # 训练模型
        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(X_res, y_res, verbose=False)

        self.X_test = X_test_normalized
        self.y_test = y_test

        print(f"数据预处理完成:")
        print(f"- 训练集形状: {X_res.shape} (SMOTE后)")
        print(f"- 测试集形状: {X_test_normalized.shape}")

        return X_test_normalized, y_test

    def calculate_all_metrics(self, y_true, y_pred, y_prob=None):
        """计算所有6个评估指标"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }

        # AUC需要概率值
        if y_prob is not None:
            metrics['AUC'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['AUC'] = 0

        return metrics

    def noise_injection_test_comprehensive(self, noise_levels=np.linspace(0, 0.5, 11)):
        """全面的噪声注入测试 - 计算6个指标"""
        print("正在进行全面的噪声注入测试...")

        # 原始性能
        y_pred_original = self.model.predict(self.X_test)
        y_prob_original = self.model.predict_proba(self.X_test)[:, 1]
        original_metrics = self.calculate_all_metrics(self.y_test, y_pred_original, y_prob_original)

        # 存储所有噪声水平下的指标
        all_metrics = {metric: [] for metric in original_metrics.keys()}

        for noise_level in noise_levels:
            metrics_sum = {metric: 0 for metric in original_metrics.keys()}

            # 多次实验取平均
            for _ in range(5):
                # 添加高斯噪声
                noise = np.random.normal(0, noise_level, self.X_test.shape)
                X_noisy = self.X_test + noise
                X_noisy = np.clip(X_noisy, 0, 1)  # 确保数据在[0,1]范围内

                # 预测
                y_pred = self.model.predict(X_noisy)
                y_prob = self.model.predict_proba(X_noisy)[:, 1]

                # 计算指标
                current_metrics = self.calculate_all_metrics(self.y_test, y_pred, y_prob)

                for metric in metrics_sum:
                    metrics_sum[metric] += current_metrics[metric]

            # 计算平均值
            for metric in all_metrics:
                all_metrics[metric].append(metrics_sum[metric] / 5)

        # 保存结果
        noise_results = pd.DataFrame({
            'noise_level': noise_levels,
            **all_metrics
        })
        noise_results.to_csv(os.path.join(self.output_dir, "noise_injection_comprehensive_results.csv"), index=False)

        return noise_levels, original_metrics, all_metrics

    def adversarial_attack_test_comprehensive(self, epsilon_values=np.linspace(0, 0.3, 11)):
        """全面的对抗样本测试 - 计算6个指标"""
        print("正在进行全面的对抗样本测试...")

        # 原始性能
        y_pred_original = self.model.predict(self.X_test)
        y_prob_original = self.model.predict_proba(self.X_test)[:, 1]
        original_metrics = self.calculate_all_metrics(self.y_test, y_pred_original, y_prob_original)

        # 存储所有扰动强度下的指标
        all_metrics = {metric: [] for metric in original_metrics.keys()}

        # 获取预测概率（用于计算梯度近似）
        pred_proba = self.model.predict_proba(self.X_test)

        for epsilon in epsilon_values:
            metrics_sum = {metric: 0 for metric in original_metrics.keys()}

            # 多次实验取平均
            for _ in range(5):
                X_adv = self.X_test.copy()

                # 对每个样本添加基于预测不确定性的扰动
                for i in range(len(X_adv)):
                    uncertainty = np.abs(pred_proba[i, 0] - pred_proba[i, 1])
                    perturbation = epsilon * uncertainty * np.random.randn(X_adv.shape[1])
                    X_adv[i] += perturbation

                X_adv = np.clip(X_adv, 0, 1)  # 确保数据在[0,1]范围内

                # 预测
                y_pred = self.model.predict(X_adv)
                y_prob = self.model.predict_proba(X_adv)[:, 1]

                # 计算指标
                current_metrics = self.calculate_all_metrics(self.y_test, y_pred, y_prob)

                for metric in metrics_sum:
                    metrics_sum[metric] += current_metrics[metric]

            # 计算平均值
            for metric in all_metrics:
                all_metrics[metric].append(metrics_sum[metric] / 5)

        # 保存结果
        adversarial_results = pd.DataFrame({
            'epsilon': epsilon_values,
            **all_metrics
        })
        adversarial_results.to_csv(os.path.join(self.output_dir, "adversarial_attack_comprehensive_results.csv"),
                                   index=False)

        return epsilon_values, original_metrics, all_metrics


def plot_noise_test_results(noise_levels, original_metrics, noise_metrics, output_dir):
    """绘制噪声注入测试的6个指标变化趋势图"""
    print("绘制噪声注入测试结果...")

    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, metric in enumerate(metrics_list):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        # 绘制当前指标的变化趋势
        ax.plot(noise_levels, noise_metrics[metric], 'o-',
                color=colors[i], linewidth=2.5, markersize=6,
                label=f'{metric} under noise')

        # 添加原始性能水平线
        ax.axhline(y=original_metrics[metric], color=colors[i],
                   linestyle='--', alpha=0.7,
                   label=f'Original {metric}: {original_metrics[metric]:.3f}')

        ax.set_xlabel('Noise Level (Standard Deviation)')
        ax.set_ylabel(f'{metric} Score')
        ax.set_title(f'{metric} vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 统一设置y轴范围为0.0到1.0
        ax.set_ylim(0.0, 1.01)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_injection_comprehensive_metrics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 创建综合趋势图（所有指标在一张图上）
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics_list):
        plt.plot(noise_levels, noise_metrics[metric], 'o-',
                 color=colors[i], linewidth=2, markersize=5,
                 label=metric)

    plt.xlabel('Noise Level (Standard Deviation)')
    plt.ylabel('Metric Score')
    plt.title('All Metrics Trend under Noise Injection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.01)
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_injection_all_metrics_combined.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 噪声注入测试图表已保存")


def plot_adversarial_test_results(epsilon_values, original_metrics, adversarial_metrics, output_dir):
    """绘制对抗样本测试的6个指标变化趋势图"""
    print("绘制对抗样本测试结果...")

    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, metric in enumerate(metrics_list):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        # 绘制当前指标的变化趋势
        ax.plot(epsilon_values, adversarial_metrics[metric], 's-',
                color=colors[i], linewidth=2.5, markersize=6,
                label=f'{metric} under attack')

        # 添加原始性能水平线
        ax.axhline(y=original_metrics[metric], color=colors[i],
                   linestyle='--', alpha=0.7,
                   label=f'Original {metric}: {original_metrics[metric]:.3f}')

        ax.set_xlabel('Perturbation Strength (ε)')
        ax.set_ylabel(f'{metric} Score')
        ax.set_title(f'{metric} vs Adversarial Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 统一设置y轴范围为0.0到1.0
        ax.set_ylim(0.0, 1.01)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adversarial_attack_comprehensive_metrics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 创建综合趋势图（所有指标在一张图上）
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics_list):
        plt.plot(epsilon_values, adversarial_metrics[metric], 's-',
                 color=colors[i], linewidth=2, markersize=5,
                 label=metric)

    plt.xlabel('Perturbation Strength (ε)')
    plt.ylabel('Metric Score')
    plt.title('All Metrics Trend under Adversarial Attack')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.01)
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adversarial_attack_all_metrics_combined.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 对抗样本测试图表已保存")


def print_comprehensive_analysis(noise_results, adversarial_results):
    """打印全面的鲁棒性分析报告"""
    noise_levels, noise_original, noise_metrics = noise_results
    epsilon_values, adversarial_original, adversarial_metrics = adversarial_results

    analysis_report = f"""
{'=' * 80}
               IABT-DW 模型鲁棒性综合分析报告
{'=' * 80}

测试配置:
   - 数据集: Darknet_Market_processed_filtered.xlsx
   - 归一化: MinMaxScaler
   - 采样: SMOTE
   - 测试集比例: 0.4

1. 噪声注入测试分析:
"""

    # 噪声测试分析
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']:
        original_val = noise_original[metric]
        final_val = noise_metrics[metric][-1]  # 最高噪声水平下的值
        degradation = (original_val - final_val) / original_val * 100

        analysis_report += f"   - {metric}: {original_val:.4f} → {final_val:.4f} (下降 {degradation:.2f}%)\n"

    analysis_report += f"""
2. 对抗样本测试分析:
"""

    # 对抗测试分析
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']:
        original_val = adversarial_original[metric]
        final_val = adversarial_metrics[metric][-1]  # 最高扰动强度下的值
        degradation = (original_val - final_val) / original_val * 100

        analysis_report += f"   - {metric}: {original_val:.4f} → {final_val:.4f} (下降 {degradation:.2f}%)\n"

    # 计算总体鲁棒性评分
    noise_robustness = np.mean([noise_metrics[metric][-1] / noise_original[metric] for metric in noise_original])
    adversarial_robustness = np.mean(
        [adversarial_metrics[metric][-1] / adversarial_original[metric] for metric in adversarial_original])
    overall_robustness = (noise_robustness + adversarial_robustness) / 2

    analysis_report += f"""
3. 总体鲁棒性评估:
   - 噪声鲁棒性评分: {noise_robustness:.4f}
   - 对抗鲁棒性评分: {adversarial_robustness:.4f}
   - 综合鲁棒性指数: {overall_robustness:.4f}

4. 关键发现:
"""

    # 分析各指标的敏感性
    metrics_sensitivity = {}
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']:
        noise_sens = (noise_original[metric] - noise_metrics[metric][-1]) / noise_original[metric]
        adv_sens = (adversarial_original[metric] - adversarial_metrics[metric][-1]) / adversarial_original[metric]
        metrics_sensitivity[metric] = (noise_sens + adv_sens) / 2

        sensitivity_level = "高" if metrics_sensitivity[metric] > 0.3 else "中等" if metrics_sensitivity[metric] > 0.15 else "低"
        analysis_report += f"   - {metric} 敏感性: {sensitivity_level} (下降 {metrics_sensitivity[metric] * 100:.1f}%)\n"

    analysis_report += f"""
5. 模型鲁棒性等级: {'优秀' if overall_robustness > 0.9 else '良好' if overall_robustness > 0.8 else '中等'}

6. 建议:
"""

    if overall_robustness > 0.9:
        analysis_report += "   - 模型在当前配置下表现出优秀的鲁棒性，建议保持现有参数配置\n"
    elif overall_robustness > 0.8:
        analysis_report += "   - 模型鲁棒性良好，可考虑进一步优化正则化参数\n"
    else:
        analysis_report += "   - 建议增强模型的正则化或重新评估特征工程\n"

    analysis_report += f"\n{'=' * 80}"

    # 打印到控制台
    print(analysis_report)

    # 保存到文件
    with open(os.path.join(output_dir, "comprehensive_robustness_analysis.txt"), "w") as f:
        f.write(analysis_report)

    return analysis_report


# 主程序
if __name__ == "__main__":
    print(f"创建输出目录: {output_dir}")

    # 模型参数
    model_params = {
        'iterations': 100,
        'learning_rate': 0.15,
        'depth': 5,
        'l2_leaf_reg': 0.5,
        'min_data_in_leaf': 3,
        'verbose': False,
        'random_seed': 0
    }

    # 保存模型参数
    params_df = pd.DataFrame([model_params])
    params_df.to_csv(os.path.join(output_dir, "model_parameters.csv"), index=False)

    # 创建评估器
    evaluator = RobustnessEvaluator(model_params, random_seed=0)

    # 准备数据
    print("准备数据并进行预处理...")
    evaluator.prepare_data()

    # 执行全面的噪声注入测试
    print("\n开始全面的噪声注入测试...")
    noise_levels, noise_original_metrics, noise_all_metrics = evaluator.noise_injection_test_comprehensive()
    noise_results = (noise_levels, noise_original_metrics, noise_all_metrics)

    # 执行全面的对抗样本测试
    print("\n开始全面的对抗样本测试...")
    epsilon_values, adversarial_original_metrics, adversarial_all_metrics = evaluator.adversarial_attack_test_comprehensive()
    adversarial_results = (epsilon_values, adversarial_original_metrics, adversarial_all_metrics)

    # 绘制噪声测试结果
    plot_noise_test_results(noise_levels, noise_original_metrics, noise_all_metrics, output_dir)

    # 绘制对抗测试结果
    plot_adversarial_test_results(epsilon_values, adversarial_original_metrics, adversarial_all_metrics, output_dir)

    # 打印全面分析报告
    analysis_report = print_comprehensive_analysis(noise_results, adversarial_results)

    print(f"\n全面的鲁棒性评估完成!")
    print(f" 所有结果已保存到: {output_dir}/")
    print(f"生成的主要图表:")
    print(f"   - noise_injection_comprehensive_metrics.png (噪声测试6指标图)")
    print(f"   - adversarial_attack_comprehensive_metrics.png (对抗测试6指标图)")
    print(f"   - 其他综合图表和详细结果文件")