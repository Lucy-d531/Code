# hyperparameter_trend_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_excel("/root/sj-tmp/暗网/Darknet_Market_processed_filtered.xlsx")
x = data.drop('label', axis=1)
y = data['label']

# 最优参数组合
optimal_params = {
    'iterations': 100,
    'learning_rate': 0.15,
    'depth': 5,
    'l2_leaf_reg': 0.5,
    'min_data_in_leaf': 3,
    'verbose': False,
    'random_seed': 0
}

# 参数扫描范围（只保留四个关键参数）
param_grids = {
    'iterations': list(range(50, 351, 50)),
    'learning_rate': [round(x, 2) for x in np.arange(0.05, 0.351, 0.05)],
    'depth': list(range(3, 9)),
    'l2_leaf_reg': list(np.arange(0.1, 1.5, 0.2))
}


def evaluate_param(param_name, param_values, x, y, cv_splits=5, random_seed=0):
    """评估单个参数"""
    metrics = {
        'Accuracy': [], 'Precision': [], 'Recall': [],
        'F1': [], 'AUC': [], 'MCC': []
    }

    for val in param_values:
        params = optimal_params.copy()
        params[param_name] = val

        fold_metrics = {
            'Accuracy': 0, 'Precision': 0, 'Recall': 0,
            'F1': 0, 'AUC': 0, 'MCC': 0
        }

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = MinMaxScaler()
            X_train_normalized = scaler.fit_transform(X_train)
            X_val_normalized = scaler.transform(X_val)

            smote = SMOTE(random_state=random_seed + fold)
            X_res, y_res = smote.fit_resample(X_train_normalized, y_train)

            model = CatBoostClassifier(**params)
            model.fit(X_res, y_res)

            y_pred = model.predict(X_val_normalized)
            y_prob = model.predict_proba(X_val_normalized)[:, 1]

            fold_metrics['Accuracy'] += accuracy_score(y_val, y_pred)
            fold_metrics['Precision'] += precision_score(y_val, y_pred, zero_division=0)
            fold_metrics['Recall'] += recall_score(y_val, y_pred, zero_division=0)
            fold_metrics['F1'] += f1_score(y_val, y_pred, zero_division=0)
            fold_metrics['AUC'] += roc_auc_score(y_val, y_prob)
            fold_metrics['MCC'] += matthews_corrcoef(y_val, y_pred)

        for metric in metrics:
            metrics[metric].append(fold_metrics[metric] / cv_splits)

    return metrics


def create_individual_parameter_charts(all_results, output_dir):
    """为每个参数创建单独的性能趋势图"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']

    for param_name, result in all_results.items():
        plt.figure(figsize=(12, 8))

        values = result['values']
        for i, metric in enumerate(metrics_list):
            scores = result['metrics'][metric]
            plt.plot(values, scores, label=metric, marker='o',
                     linewidth=2.5, markersize=6, color=colors[i])

        plt.xlabel(param_name, fontsize=12)
        plt.ylabel("Score", fontsize=12)

        plt.ylim(0.995, 1.000)
        plt.yticks(np.arange(0.995, 1.000, 0.001))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{param_name}_performance_trend.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 参数趋势图已保存: {filename}")


def focused_hyperparameter_analysis(x, y, optimal_params, param_grids, cv_splits=5, random_seed=0):
    """专注的超参数趋势分析 - 只生成四个参数趋势图"""
    print("开始超参数趋势分析...")

    # 创建输出目录
    output_dir = "Parameter_Trend_Analysis"
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有参数评估结果
    all_results = {}

    # 只评估四个关键参数
    print("\n步骤 1/2: 评估四个关键超参数...")
    target_params = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg']

    for param_name in target_params:
        if param_name in param_grids:
            print(f"  正在评估: {param_name}")
            values = param_grids[param_name]
            metrics = evaluate_param(param_name, values, x, y, cv_splits, random_seed)
            all_results[param_name] = {'values': values, 'metrics': metrics}

    # 只创建参数趋势图
    print("\n步骤 2/2: 创建参数趋势图...")
    create_individual_parameter_charts(all_results, output_dir)

    print(f"\n参数趋势分析完成! 图表已保存到: {output_dir}")
    print(f"📊 生成图表数量: 4 张参数趋势图")

    # 返回结果用于可能的进一步分析
    return all_results


# 运行专注的趋势分析
if __name__ == "__main__":
    all_results = focused_hyperparameter_analysis(x, y, optimal_params, param_grids)
    print("\n超参数趋势分析完成!")