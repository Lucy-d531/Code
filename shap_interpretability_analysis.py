# shap_interpretability_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import warnings
import shap
import os
import json

warnings.filterwarnings('ignore')

# 全局设置
save_dir = "IABT_DW_SHAP_analysis"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")

print("=== IABT-DW CatBoost Model SHAP Interpretability Analysis ===")


def load_and_preprocess_data():
    """数据加载和预处理"""
    print("Step 1: Loading data and preprocessing...")

    # 读取数据
    try:
        data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

    X = data.drop('label', axis=1)
    y = data['label']

    print(f"Data loaded. Shape: {data.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # 预处理流程
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0
    )

    # 使用原始数据进行SMOTE过采样
    smote = SMOTE(random_state=0)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 转换为DataFrame
    X_train_processed = pd.DataFrame(X_train_resampled, columns=X.columns)
    X_test_processed = pd.DataFrame(X_test, columns=X.columns)
    y_train_processed = pd.Series(y_train_resampled)
    y_test_processed = pd.Series(y_test.values, index=X_test_processed.index)

    return X_train_processed, y_train_processed, X_test_processed, y_test_processed


def train_model(X_train, y_train, X_test, y_test):
    """训练CatBoost模型"""
    print("\nStep 2: Training CatBoost Model")

    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.15,
        depth=5,
        l2_leaf_reg=0.5,
        min_data_in_leaf=3,
        verbose=False,
        random_state=0
    )

    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model training completed")
    print(f"Accuracy: {accuracy:.4f}")

    return model, accuracy


def initialize_shap_explainer(model, X_test, y_test):
    """初始化SHAP解释器"""
    print(f"\nStep 3: Initializing SHAP Explainer with entire test set ({len(X_test)} samples)")

    # 使用整个测试集
    X_full = X_test.copy()
    y_full = y_test.copy()

    # 创建解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)

    # 处理SHAP值
    if isinstance(shap_values, list):
        shap_values_processed = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                   list) else explainer.expected_value
    else:
        shap_values_processed = shap_values
        expected_value = explainer.expected_value

    print(f"SHAP values shape: {shap_values_processed.shape}")
    print(f"Expected value: {expected_value:.4f}")
    print(f"Number of samples analyzed: {len(X_full)}")

    return explainer, shap_values_processed, expected_value, X_full, y_full


def calculate_feature_importance(shap_values, X_sample):
    """计算特征重要性"""
    print("\nStep 4: Calculating feature importance")

    feature_importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_importance': np.abs(shap_values).mean(0),
        'mean_shap_value': shap_values.mean(0),
        'std_shap_value': shap_values.std(0)
    }).sort_values('shap_importance', ascending=False)

    print("Top 10 features by SHAP importance:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        direction = "Positive" if row['mean_shap_value'] > 0 else "Negative"
        print(f"  {i}. {row['feature']}: {row['shap_importance']:.4f} ({direction})")

    return feature_importance_df


def generate_shap_summary_plot(shap_values, X_sample, save_dir):
    """生成SHAP汇总图"""
    print("\nStep 5: Generating SHAP Summary Plot")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/IABT_DW_shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary plot saved")


def generate_shap_dependence_plots(shap_values, X_sample, save_dir, top_n=8):
    """生成SHAP依赖图"""
    print(f"\nStep 6: Generating SHAP Dependence Plots (Top {top_n} Features)")

    # 计算特征重要性
    top_features = np.abs(shap_values).mean(0)
    top_feature_indices = np.argsort(top_features)[-top_n:]

    print("Selected features for dependence plots:")
    for i, feature_idx in enumerate(top_feature_indices):
        feature_name = X_sample.columns[feature_idx]
        importance_score = np.abs(shap_values[:, feature_idx]).mean()
        print(f"  {i + 1}. {feature_name}: SHAP Importance = {importance_score:.4f}")

    # 生成依赖图
    for i, feature_idx in enumerate(top_feature_indices):
        plt.figure(figsize=(12, 8))
        feature_name = X_sample.columns[feature_idx]

        shap.dependence_plot(feature_idx, shap_values, X_sample, show=False)

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/IABT_DW_shap_dependence_{i + 1}_{feature_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Dependence plot {i + 1}/8 for '{feature_name}' saved")


def generate_waterfall_plots_one_each(model, X_sample, y_sample, save_dir):
    """绘制瀑布图（一正一负样本）"""
    print("\nStep 7: Generating Waterfall Plots (One Positive and One Negative Sample)")

    # 选择正样本和负样本
    positive_idx = np.where(y_sample == 1)[0]
    negative_idx = np.where(y_sample == 0)[0]

    print(f"Found {len(positive_idx)} positive samples and {len(negative_idx)} negative samples")

    # 创建新的解释器用于瀑布图
    explainer_waterfall = shap.Explainer(model)
    shap_explanation = explainer_waterfall(X_sample)

    # 生成正样本瀑布图
    if len(positive_idx) > 0:
        pos_sample_idx = positive_idx[0]
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            shap_explanation[pos_sample_idx],
            max_display=10,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/IABT_DW_SHAP_waterfall_positive.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Positive sample waterfall plot saved")

        # 打印正样本详细信息
        y_pred_proba = model.predict_proba(X_sample.iloc[[pos_sample_idx]])[0, 1]
        print(f"  Positive sample index: {pos_sample_idx}, Predicted probability: {y_pred_proba:.4f}")

    # 生成负样本瀑布图
    if len(negative_idx) > 0:
        neg_sample_idx = negative_idx[0]
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            shap_explanation[neg_sample_idx],
            max_display=10,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/IABT_DW_SHAP_waterfall_negative.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Negative sample waterfall plot saved")

        # 打印负样本详细信息
        y_pred_proba = model.predict_proba(X_sample.iloc[[neg_sample_idx]])[0, 1]
        print(f"  Negative sample index: {neg_sample_idx}, Predicted probability: {y_pred_proba:.4f}")

    # 返回使用的样本索引
    used_samples = {}
    if len(positive_idx) > 0:
        used_samples['positive'] = int(positive_idx[0])
    if len(negative_idx) > 0:
        used_samples['negative'] = int(negative_idx[0])

    return used_samples


def save_analysis_results(feature_importance_df, used_samples, X_sample, y_sample,
                          shap_values, X_train_processed, X_test_processed, save_dir):
    """保存分析结果"""
    print("\nStep 8: Saving Analysis Results")

    # 保存特征重要性
    feature_importance_df.to_csv(f'{save_dir}/IABT_DW_shap_feature_importance.csv', index=False)
    print("✓ SHAP feature importance saved")

    # 保存样本详情
    sample_details_list = []

    # 保存正样本详情
    if 'positive' in used_samples:
        pos_idx = used_samples['positive']
        y_pred_proba = model.predict_proba(X_sample.iloc[[pos_idx]])[0, 1]

        sample_details = {
            'sample_type': 'Positive',
            'sample_index': pos_idx,
            'actual_label': int(y_sample.iloc[pos_idx]),
            'predicted_probability': float(y_pred_proba),
        }

        for i, feature in enumerate(X_sample.columns):
            sample_details[f'shap_{feature}'] = float(shap_values[pos_idx, i])

        sample_details_list.append(sample_details)

    # 保存负样本详情
    if 'negative' in used_samples:
        neg_idx = used_samples['negative']
        y_pred_proba = model.predict_proba(X_sample.iloc[[neg_idx]])[0, 1]

        sample_details = {
            'sample_type': 'Negative',
            'sample_index': neg_idx,
            'actual_label': int(y_sample.iloc[neg_idx]),
            'predicted_probability': float(y_pred_proba),
        }

        for i, feature in enumerate(X_sample.columns):
            sample_details[f'shap_{feature}'] = float(shap_values[neg_idx, i])

        sample_details_list.append(sample_details)

    sample_details_df = pd.DataFrame(sample_details_list)
    sample_details_df.to_csv(f'{save_dir}/IABT_DW_shap_sample_details.csv', index=False)
    print("✓ Sample details saved")

    # 保存预处理信息
    preprocessing_info = {
        'preprocessing_steps': [
            'SMOTE oversampling (random_state=0)',
            'Train-test split (test_size=0.4, stratify=y, random_state=0)',
            'No normalization applied - using original feature scales'
        ],
        'data_shape_original': f"{X_train_processed.shape[0] + X_test_processed.shape[0]} rows, {X_train_processed.shape[1]} columns",
        'data_shape_after_preprocessing': {
            'X_train_processed': [int(X_train_processed.shape[0]), int(X_train_processed.shape[1])],
            'X_test_processed': [int(X_test_processed.shape[0]), int(X_test_processed.shape[1])]
        },
        'shap_analysis_scope': 'Entire test set used for SHAP analysis',
        'waterfall_samples_used': used_samples
    }

    # 自定义JSON编码器来处理numpy类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super().default(obj)

    with open(f'{save_dir}/IABT_DW_preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2, cls=NumpyEncoder)
    print("✓ Preprocessing information saved")


def generate_analysis_report(accuracy, feature_importance_df, used_samples, X_test_processed, save_dir):
    """生成分析报告"""
    print("\nStep 9: Generating Analysis Report")
    print("=" * 80)
    print("IABT-DW CatBoost Model SHAP Interpretability Analysis Report")
    print("=" * 80)

    print(f"\nModel Performance:")
    print(f"- Accuracy: {accuracy:.4f}")

    print(f"\nDataset Information:")
    print(f"- Test set size: {X_test_processed.shape[0]} samples")
    print(f"- Number of features: {X_test_processed.shape[1]}")

    print(f"\nTop 10 Most Important Features:")
    top_10_features = feature_importance_df.head(10)
    for i, (_, row) in enumerate(top_10_features.iterrows(), 1):
        direction = "Positive" if row['mean_shap_value'] > 0 else "Negative"
        print(f"  {i}. {row['feature']}: Importance={row['shap_importance']:.4f} ({direction})")

    print(f"\nWaterfall Analysis Summary:")
    if 'positive' in used_samples:
        pos_idx = used_samples['positive']
        y_pred_proba = model.predict_proba(X_test_processed.iloc[[pos_idx]])[0, 1]
        print(f"- Positive sample: Index {pos_idx}, Predicted probability: {y_pred_proba:.4f}")
    if 'negative' in used_samples:
        neg_idx = used_samples['negative']
        y_pred_proba = model.predict_proba(X_test_processed.iloc[[neg_idx]])[0, 1]
        print(f"- Negative sample: Index {neg_idx}, Predicted probability: {y_pred_proba:.4f}")

    print(f"- Total SHAP charts generated: {len([f for f in os.listdir(save_dir) if f.endswith('.png')])}")
    print(f"- SHAP analysis performed on entire test set")
    print(f"- No normalization applied to features")

    print(f"\nGenerated Files:")
    for file in os.listdir(save_dir):
        if file.endswith('.png') or file.endswith('.csv') or file.endswith('.json'):
            file_path = os.path.join(save_dir, file)
            file_size = os.path.getsize(file_path) / 1024
            print(f"✅ {file} ({file_size:.1f} KB)")

    print(f"\n🎉 IABT-DW SHAP Analysis Completed!")
    print(f"📊 All results saved to: {save_dir}")


# === 主执行流程 ===
if __name__ == "__main__":
    # 步骤1: 数据预处理
    X_train_processed, y_train_processed, X_test_processed, y_test_processed = load_and_preprocess_data()

    # 步骤2: 训练模型
    model, accuracy = train_model(X_train_processed, y_train_processed, X_test_processed, y_test_processed)

    # 步骤3: 初始化SHAP
    explainer, shap_values, expected_value, X_full, y_full = initialize_shap_explainer(
        model, X_test_processed, y_test_processed
    )

    # 步骤4: 计算特征重要性
    feature_importance_df = calculate_feature_importance(shap_values, X_full)

    # 步骤5: 生成SHAP汇总图
    generate_shap_summary_plot(shap_values, X_full, save_dir)

    # 步骤6: 生成SHAP依赖图
    generate_shap_dependence_plots(shap_values, X_full, save_dir, top_n=8)

    # 步骤7: 绘制瀑布图
    used_samples = generate_waterfall_plots_one_each(model, X_full, y_full, save_dir)

    # 步骤8: 保存分析结果
    save_analysis_results(feature_importance_df, used_samples, X_full, y_full,
                          shap_values, X_train_processed, X_test_processed, save_dir)

    # 步骤9: 生成分析报告
    generate_analysis_report(accuracy, feature_importance_df, used_samples, X_test_processed, save_dir)