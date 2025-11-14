# catboost_smote_gridsearch.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import joblib
from sklearn.base import clone
import os

# 读取数据
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
x = data.drop('label', axis=1)
y = data['label']


def evaluate_baseline_model_minmax_with_split(x, y, splits_num=5, random_seed=0, test_size=0.4):
    """评估基准模型（未调优）的性能（使用归一化和数据集划分）"""
    print("\n开始评估基准模型性能（归一化 + 数据集划分）...")

    # 首先进行数据集划分 6:4
    X_temp, X_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    print(f"训练集大小: {X_temp.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 创建管道（使用MinMaxScaler和SMOTE）
    pipeline = make_pipeline(
        MinMaxScaler(),
        SMOTE(random_state=random_seed),
        CatBoostClassifier(
            random_seed=random_seed,
            verbose=0,
            thread_count=1
        )
    )

    # 在临时训练集上使用交叉验证
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    # 存储结果
    fold_results = []
    all_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': [], 'MCC': []}

    for fold, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
        print(f"处理第 {fold + 1}/{splits_num} 折...")
        X_train, X_val = X_temp.iloc[train_index], X_temp.iloc[val_index]
        y_train, y_val = y_temp.iloc[train_index], y_temp.iloc[val_index]

        # 克隆管道以确保每次交叉验证独立
        model = clone(pipeline)
        model.fit(X_train, y_train)

        # 在验证集上预测
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # 计算指标
        metrics = {
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'F1': f1_score(y_val, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_val, y_prob),
            'MCC': matthews_corrcoef(y_val, y_pred)
        }

        # 存储结果
        fold_results.append({
            'Fold': fold + 1,
            **metrics
        })

        # 添加到指标列表
        for key in metrics:
            all_metrics[key].append(metrics[key])

    # 计算平均指标
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    # 在整个临时训练集上训练最终模型并在独立测试集上评估
    print("在独立测试集上评估基准模型...")
    final_model = clone(pipeline)
    final_model.fit(X_temp, y_temp)

    # 在测试集上预测
    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]

    # 计算测试集指标
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Recall': recall_score(y_test, y_test_pred, zero_division=0),
        'F1': f1_score(y_test, y_test_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_test_prob),
        'MCC': matthews_corrcoef(y_test, y_test_pred)
    }

    # 创建结果DataFrame
    result_df = pd.DataFrame(fold_results)
    avg_row = pd.DataFrame({
        'Fold': ['CV Average'],
        'Accuracy': [avg_metrics['Accuracy']],
        'Precision': [avg_metrics['Precision']],
        'Recall': [avg_metrics['Recall']],
        'F1': [avg_metrics['F1']],
        'AUC': [avg_metrics['AUC']],
        'MCC': [avg_metrics['MCC']]
    })

    test_row = pd.DataFrame({
        'Fold': ['Test Set'],
        'Accuracy': [test_metrics['Accuracy']],
        'Precision': [test_metrics['Precision']],
        'Recall': [test_metrics['Recall']],
        'F1': [test_metrics['F1']],
        'AUC': [test_metrics['AUC']],
        'MCC': [test_metrics['MCC']]
    })

    result_df = pd.concat([result_df, avg_row, test_row], ignore_index=True)

    return result_df, avg_metrics, test_metrics


def grid_search_catboost_smote_split(x, y, splits_num=5, random_seed=0, test_size=0.4,
                                     outputName='CatBoost_SMOTE_MinMax_Split_GridSearch'):
    """使用网格搜索优化CatBoost参数，配合SMOTE采样、归一化和数据集划分"""

    # 首先进行数据集划分 6:4
    X_temp, X_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    print(f"训练集大小: {X_temp.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 首先评估基准模型性能
    print("评估基准模型性能...")
    baseline_df, baseline_cv_metrics, baseline_test_metrics = evaluate_baseline_model_minmax_with_split(
        x, y, splits_num, random_seed, test_size
    )

    # 创建处理管道（使用MinMaxScaler和SMOTE）
    pipeline = make_pipeline(
        MinMaxScaler(),
        SMOTE(random_state=random_seed),
        CatBoostClassifier(
            random_seed=random_seed,
            verbose=0,
            thread_count=1
        )
    )

    param_grid = {
        'catboostclassifier__iterations': [100, 200, 300],
        'catboostclassifier__learning_rate': [0.05, 0.15, 0.25],
        'catboostclassifier__depth': [5, 6, 7],
        'catboostclassifier__l2_leaf_reg': [0.1, 0.5, 1.5],
        'catboostclassifier__min_data_in_leaf': [3, 4, 5]
    }

    # 创建分层K折交叉验证（在临时训练集上）
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    # 创建网格搜索
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        refit=True,
        cv=skf,
        n_jobs=1,
        verbose=3
    )

    # 执行网格搜索（在临时训练集上）
    print("\n开始网格搜索（在训练集上）...")
    grid_search.fit(X_temp, y_temp)
    print("网格搜索完成！")

    # 获取最佳参数
    best_params = grid_search.best_params_
    print("\n最佳参数组合:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 存储交叉验证结果
    cv_results = []
    all_cv_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': [], 'MCC': []}

    # 保存整个训练集训练的最佳模型
    best_model = grid_search.best_estimator_
    model_path = f"{outputName}_best_model.pkl"

    # 确保输出目录存在
    output_dir = os.path.dirname(outputName) if os.path.dirname(outputName) else '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    joblib.dump(best_model, model_path)
    print(f"最佳模型已保存到 '{model_path}'")

    # 使用最佳模型进行交叉验证评估（在临时训练集上）
    print("\n使用最佳参数进行交叉验证评估...")
    for fold, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
        print(f"评估第 {fold + 1}/{splits_num} 折...")
        X_train, X_val = X_temp.iloc[train_index], X_temp.iloc[val_index]
        y_train, y_val = y_temp.iloc[train_index], y_temp.iloc[val_index]

        # 使用最佳参数创建新模型
        model = clone(pipeline).set_params(**best_params)
        model.fit(X_train, y_train)

        # 在验证集上预测
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # 计算指标
        metrics = {
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'F1': f1_score(y_val, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_val, y_prob),
            'MCC': matthews_corrcoef(y_val, y_pred)
        }

        # 存储结果
        cv_results.append({
            'Fold': fold + 1,
            **metrics
        })

        # 添加到指标列表
        for key in metrics:
            all_cv_metrics[key].append(metrics[key])

    # 计算交叉验证平均指标
    avg_cv_metrics = {key: np.mean(values) for key, values in all_cv_metrics.items()}

    # 在独立测试集上评估最佳模型
    print("\n在独立测试集上评估优化后的模型...")
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Recall': recall_score(y_test, y_test_pred, zero_division=0),
        'F1': f1_score(y_test, y_test_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_test_prob),
        'MCC': matthews_corrcoef(y_test, y_test_pred)
    }

    # 创建结果DataFrame
    cv_result_df = pd.DataFrame(cv_results)
    cv_avg_row = pd.DataFrame({
        'Fold': ['CV Average'],
        'Accuracy': [avg_cv_metrics['Accuracy']],
        'Precision': [avg_cv_metrics['Precision']],
        'Recall': [avg_cv_metrics['Recall']],
        'F1': [avg_cv_metrics['F1']],
        'AUC': [avg_cv_metrics['AUC']],
        'MCC': [avg_cv_metrics['MCC']]
    })

    test_row = pd.DataFrame({
        'Fold': ['Test Set'],
        'Accuracy': [test_metrics['Accuracy']],
        'Precision': [test_metrics['Precision']],
        'Recall': [test_metrics['Recall']],
        'F1': [test_metrics['F1']],
        'AUC': [test_metrics['AUC']],
        'MCC': [test_metrics['MCC']]
    })

    result_df = pd.concat([cv_result_df, cv_avg_row, test_row], ignore_index=True)

    # 添加最佳参数信息
    best_params_df = pd.DataFrame([best_params])
    best_params_df.insert(0, 'Description', 'Best Parameters')

    # 创建性能对比表格
    comparison_df = pd.DataFrame({
        'Model': ['Baseline (CV)', 'Baseline (Test)', 'Optimized (CV)', 'Optimized (Test)'],
        'Accuracy': [baseline_cv_metrics['Accuracy'], baseline_test_metrics['Accuracy'],
                     avg_cv_metrics['Accuracy'], test_metrics['Accuracy']],
        'Precision': [baseline_cv_metrics['Precision'], baseline_test_metrics['Precision'],
                      avg_cv_metrics['Precision'], test_metrics['Precision']],
        'Recall': [baseline_cv_metrics['Recall'], baseline_test_metrics['Recall'],
                   avg_cv_metrics['Recall'], test_metrics['Recall']],
        'F1': [baseline_cv_metrics['F1'], baseline_test_metrics['F1'],
               avg_cv_metrics['F1'], test_metrics['F1']],
        'AUC': [baseline_cv_metrics['AUC'], baseline_test_metrics['AUC'],
                avg_cv_metrics['AUC'], test_metrics['AUC']],
        'MCC': [baseline_cv_metrics['MCC'], baseline_test_metrics['MCC'],
                avg_cv_metrics['MCC'], test_metrics['MCC']]
    })

    # 确保输出目录存在
    output_file = f"{outputName}.xlsx"
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    with pd.ExcelWriter(output_file) as writer:
        baseline_df.to_excel(writer, sheet_name='Baseline Performance', index=False)
        result_df.to_excel(writer, sheet_name='Optimized Performance', index=False)
        best_params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        comparison_df.to_excel(writer, sheet_name='Performance Comparison', index=False)

    print(f"\n评估完成！结果已保存到 '{output_file}'")

    return result_df, best_params, comparison_df


# 执行网格搜索和评估（使用SMOTE和数据集划分）
if __name__ == "__main__":
    result, best_params, comparison = grid_search_catboost_smote_split(
        x, y,
        splits_num=5,
        random_seed=0,
        test_size=0.4,
        outputName='NEW_canshu_CatBoost_SMOTE_MinMax_GridSearch'
    )