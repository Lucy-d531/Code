# multiple_model_comparison.py
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
x = data.drop('label', axis=1)
y = data['label']


def get_all_classifiers(random_seed=0):
    """定义所有分类器"""
    return {
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=random_seed),
        'SGDClassifier': SGDClassifier(random_state=random_seed),
        'RidgeClassifier': RidgeClassifier(random_state=random_seed),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_seed),
        'ExtraTreeClassifier': ExtraTreeClassifier(random_state=random_seed),
        'MLPClassifier': MLPClassifier(random_state=random_seed),
        'RandomForestClassifier': RandomForestClassifier(random_state=random_seed),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=random_seed),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(random_state=random_seed, probability=True),
        'XGBClassifier': XGBClassifier(random_state=random_seed),
        'CatBoostClassifier': CatBoostClassifier(random_state=random_seed, verbose=0)
    }


def evaluate_all_with_smote_improved(
        x, y, splits_num=5, random_seed=0,
        outputName='', scaling_method='minmax', test_size=0.4):
    """使用Pipeline避免数据泄露，确保预处理一致性"""
    
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
            'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auc': [], 'mcc': []
        }

        # 交叉验证
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all), 1):
            X_train, X_val = X_train_all.iloc[train_idx], X_train_all.iloc[val_idx]
            y_train, y_val = y_train_all.iloc[train_idx], y_train_all.iloc[val_idx]

            # 使用Pipeline确保预处理一致性
            pipeline_steps = []

            # 缩放处理
            if scaling_method == 'minmax':
                pipeline_steps.append(('scaler', MinMaxScaler()))
            elif scaling_method == 'standard':
                pipeline_steps.append(('scaler', StandardScaler()))

            # SMOTE过采样
            pipeline_steps.append(('smote', SMOTE(random_state=random_seed)))

            # 分类器
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

            print(f"测试集: ACC={test_metrics['test_acc']:.4f}, F1={test_metrics['test_f1']:.4f}, AUC={test_metrics['test_auc']:.4f}")

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

    # 输出结果
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('test_f1', ascending=False)

    output_filename = f"{outputName}_improved.xlsx"
    result_df.to_excel(output_filename, index=False)
    print(f"\n所有模型评估完成！结果已保存至 {output_filename}")

    return result_df


# 运行示例
if __name__ == "__main__":
    result = evaluate_all_with_smote_improved(
        x, y,
        splits_num=5,
        random_seed=0,
        outputName='all_models_SMOTE_minmax_6_4',
        scaling_method='minmax',
        test_size=0.4
    )

    print(result.head())