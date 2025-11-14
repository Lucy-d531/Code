# best_model_visualization.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import time
from catboost import CatBoostClassifier

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
x = data.drop('label', axis=1)
y = data['label']


def plot_iabt_dw_pr_roc(x, y, test_size=0.4, random_seed=0):
    """为IABT-DW模型绘制Precision-Recall和ROC曲线"""
    print(f"\n{'=' * 60}")
    print(f"开始绘制 IABT-DW 模型的PR和ROC曲线")
    print(f"{'=' * 60}")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_seed)

    # 创建模型的pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=random_seed)),
        ('classifier', CatBoostClassifier(random_state=random_seed, verbose=0))
    ])

    # 记录训练时间
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    # 预测
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }

    # 输出分析结果
    print(f"\n📊 IABT-DW 模型性能指标:")
    print(f"{'-' * 50}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"测试集样本数: {len(X_test)}")
    print(f"正样本比例: {y_test.mean():.3f}")
    print(f"\n📈 性能指标:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():<12}: {value:.6f}")

    # 设置颜色
    primary_color = '#FF6B6B'

    # 创建PR和ROC曲线对比图
    plt.figure(figsize=(12, 5))

    # PR曲线
    plt.subplot(1, 2, 1)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, color=primary_color, linewidth=2,
             label='IABT-DW')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ROC曲线
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=primary_color, linewidth=2,
             label=f'IABT-DW (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Top_Models_PR_ROC_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ PR和ROC曲线对比图已生成: Top_Models_PR_ROC_Comparison.png")

    return metrics


def main():
    """主函数 - 专注于IABT-DW模型的PR和ROC曲线绘制"""
    print(f"{'=' * 60}")
    print(f"IABT-DW 模型 PR 和 ROC 曲线分析")
    print(f"{'=' * 60}")

    # 绘制IABT-DW模型的PR和ROC曲线
    metrics = plot_iabt_dw_pr_roc(
        x, y,
        test_size=0.4,
        random_seed=0
    )

    print(f"\n分析完成!")
    print(f"  - PR和ROC曲线已保存至: Top_Models_PR_ROC_Comparison.png")
    print(f"  - 模型AUC: {metrics['auc']:.6f}")
    print(f"  - 模型F1: {metrics['f1']:.6f}")


if __name__ == "__main__":
    main()