import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use(['default'])

warnings.filterwarnings('ignore')

# === 1. 读取数据 ===
data = pd.read_excel('/root/sj-tmp/暗网/Darknet_Market_processed_filtered.xlsx')
x = data.drop('label', axis=1)
y = data['label']

# === 2. 定义所有模型 ===
def get_all_classifiers(random_seed=0):
    return {
        'PassiveAggressive': PassiveAggressiveClassifier(random_state=random_seed),
        'SGD': SGDClassifier(random_state=random_seed),
        'Ridge': RidgeClassifier(random_state=random_seed),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'DecisionTree': DecisionTreeClassifier(random_state=random_seed),
        'ExtraTree': ExtraTreeClassifier(random_state=random_seed),
        'MLP': MLPClassifier(random_state=random_seed),
        'RandomForest': RandomForestClassifier(random_state=random_seed),
        'ExtraTrees': ExtraTreesClassifier(random_state=random_seed),
        'KNeighbors': KNeighborsClassifier(),
        'SVC': SVC(random_state=random_seed, probability=True),
        'XGBoost': XGBClassifier(random_state=random_seed),
        'IABT-DW': CatBoostClassifier(random_state=random_seed, verbose=0)
    }

# === 3. 可视化函数 ===
def create_comprehensive_visualizations(result_df, output_prefix):
    """
    创建模型对比可视化图表
    包含雷达图和性能折线图
    """
    # 设置颜色
    your_model_color = '#FF6B6B'
    other_model_color = '#4ECDC4'
    top_model_colors = ['#FF6B6B', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    # 定义指标和对应的列名
    metrics_display = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
    metrics_columns = ['test_acc', 'test_pre', 'test_rec', 'test_f1', 'test_auc', 'test_mcc']

    # 统一的字体设置
    title_font = {'fontsize': 16, 'fontweight': 'bold', 'fontfamily': 'DejaVu Sans'}
    axis_font = {'fontsize': 12, 'fontfamily': 'DejaVu Sans'}
    legend_font = {'size': 10, 'family': 'DejaVu Sans'}

    # 1. 雷达图 - 显示前5个模型
    if len(result_df) >= 5:
        top_models = result_df.nlargest(5, 'test_f1')

        categories = metrics_display
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        fill_color = '#FFF8E1'

        for i, (idx, row) in enumerate(top_models.iterrows()):
            values = [row[metric_col] for metric_col in metrics_columns]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=row['classifier'], color=top_model_colors[i])
            ax.fill(angles, values, alpha=0.2, color=fill_color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontdict=axis_font)
        ax.set_ylim(0.98, 1.0)
        ax.set_title('Top 5 Models Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), prop=legend_font)
        plt.savefig(f'{output_prefix}_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 雷达图已生成")

    # 2. 性能对比折线图 - 只显示前5个模型
    plt.figure(figsize=(14, 8))
    top_models = result_df.nlargest(5, 'test_f1')
    x_pos = np.arange(len(metrics_display))

    for i, (_, row) in enumerate(top_models.iterrows()):
        values = [row[metric_col] for metric_col in metrics_columns]
        color = your_model_color if 'IABT' in str(row['classifier']) else top_model_colors[i]
        plt.plot(x_pos, values, 'o-', linewidth=2.5, label=row['classifier'],
                 color=color, markersize=8)

    plt.xticks(x_pos, metrics_display, fontsize=axis_font['fontsize'])
    plt.ylabel('Performance Score', **axis_font)
    plt.title('Top 5 Models Performance Comparison')
    plt.yticks(fontsize=axis_font['fontsize'])
    plt.ylim(0.990, 1.001)
    plt.yticks(np.arange(0.990, 1.001, 0.001))
    plt.legend(bbox_to_anchor=(0.00, 0.00), loc='lower left', prop=legend_font)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_linechart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 性能折线图已生成")

    print(f"所有可视化图表已保存至 {output_prefix}_*.png")

# === 4. 主函数 ===
def evaluate_and_visualize_models(
        x, y, splits_num=5, random_seed=0,
        outputName='', scaling_method='minmax', test_size=0.4):
    """
    模型训练和可视化的主函数
    """
    classifiers = get_all_classifiers(random_seed)
    results = []

    # 划分数据集
    X_train_all, X_test_final, y_train_all, y_test_final = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_seed)
    print(f"数据划分完成: 训练集 {X_train_all.shape}, 测试集 {X_test_final.shape}")

    # 定义交叉验证
    skf = StratifiedKFold(n_splits=splits_num, shuffle=True, random_state=random_seed)

    # 遍历每个模型
    for name, model in classifiers.items():
        print(f"\n开始训练模型：{name}")
        cv_metrics = {
            'acc': [], 'pre': [], 'rec': [],
            'f1': [], 'auc': [], 'mcc': []
        }

        # 交叉验证
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all), 1):
            X_train, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
            y_train, y_val = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]

            # 使用Pipeline确保预处理一致性
            pipeline_steps = []

            # 1. 缩放
            if scaling_method == 'minmax':
                pipeline_steps.append(('scaler', MinMaxScaler()))
            elif scaling_method == 'standard':
                pipeline_steps.append(('scaler', StandardScaler()))

            # 2. SMOTE过采样
            pipeline_steps.append(('smote', SMOTE(random_state=random_seed)))

            # 3. 分类器
            pipeline_steps.append(('classifier', model))

            # 创建pipeline
            pipeline = Pipeline(pipeline_steps)

            try:
                # 训练
                pipeline.fit(X_train, y_train)

                # 预测
                y_pred = pipeline.predict(X_val)
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_val)[:, 1]
                else:
                    y_dec = pipeline.decision_function(X_val)
                    y_prob = (y_dec - y_dec.min()) / (y_dec.max() - y_dec.min())

                # 计算指标
                cv_metrics['acc'].append(accuracy_score(y_val, y_pred))
                cv_metrics['pre'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_metrics['rec'].append(recall_score(y_val, y_pred, zero_division=0))
                cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                cv_metrics['auc'].append(roc_auc_score(y_val, y_prob))
                cv_metrics['mcc'].append(matthews_corrcoef(y_val, y_pred))

            except Exception as e:
                print(f"  折 {fold} 训练失败: {e}")
                for metric in cv_metrics:
                    cv_metrics[metric].append(0 if metric != 'auc' else 0.5)

        # 计算交叉验证平均指标
        avg_metrics = {}
        for metric, values in cv_metrics.items():
            avg_metrics[f'cv_{metric}'] = np.mean(values)

        print(f"{name} CV 平均 F1={avg_metrics['cv_f1']:.4f}, AUC={avg_metrics['cv_auc']:.4f}")

        # 最终测试集评估
        try:
            # 重新构建最终pipeline
            final_pipeline_steps = []
            if scaling_method == 'minmax':
                final_pipeline_steps.append(('scaler', MinMaxScaler()))
            elif scaling_method == 'standard':
                final_pipeline_steps.append(('scaler', StandardScaler()))

            final_pipeline_steps.append(('smote', SMOTE(random_state=random_seed)))
            final_pipeline_steps.append(('classifier', model))

            final_pipeline = Pipeline(final_pipeline_steps)

            # 在完整训练集上训练
            final_pipeline.fit(X_train_all, y_train_all)

            # 测试集预测
            y_test_pred = final_pipeline.predict(X_test_final)
            if hasattr(final_pipeline, "predict_proba"):
                y_test_prob = final_pipeline.predict_proba(X_test_final)[:, 1]
            else:
                y_test_dec = final_pipeline.decision_function(X_test_final)
                y_test_prob = (y_test_dec - y_test_dec.min()) / (y_test_dec.max() - y_test_dec.min())

            test_metrics = {
                'test_acc': accuracy_score(y_test_final, y_test_pred),
                'test_pre': precision_score(y_test_final, y_test_pred, zero_division=0),
                'test_rec': recall_score(y_test_final, y_test_pred, zero_division=0),
                'test_f1': f1_score(y_test_final, y_test_pred, zero_division=0),
                'test_auc': roc_auc_score(y_test_final, y_test_prob),
                'test_mcc': matthews_corrcoef(y_test_final, y_test_pred)
            }

            print(f"测试集: ACC={test_metrics['test_acc']:.4f}, F1={test_metrics['test_f1']:.4f}")

            # 合并结果
            result_row = {'classifier': name}
            result_row.update(avg_metrics)
            result_row.update(test_metrics)
            results.append(result_row)

        except Exception as e:
            print(f"最终模型训练失败: {e}")
            result_row = {'classifier': name}
            result_row.update(avg_metrics)
            for metric in ['acc', 'pre', 'rec', 'f1', 'auc', 'mcc']:
                result_row[f'test_{metric}'] = 0 if metric != 'auc' else 0.5
            results.append(result_row)

    # 生成结果DataFrame
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('test_f1', ascending=False)

    # 创建综合可视化
    print("\n开始生成可视化图表...")
    create_comprehensive_visualizations(result_df, outputName)

    # 保存结果到Excel
    output_filename = f"{outputName}_results.xlsx"
    result_df.to_excel(output_filename, index=False)
    print(f"结果已保存至 {output_filename}")

    return result_df

# === 运行示例 ===
if __name__ == "__main__":
    result = evaluate_and_visualize_models(
        x, y,
        splits_num=5,
        random_seed=0,
        outputName='Model_Comparison_Visualization',
        scaling_method='minmax',
        test_size=0.4
    )