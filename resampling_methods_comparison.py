# resampling_methods_comparison.py
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
x = data.drop('label', axis=1)
y = data['label']


def evaluate_cat_with_resampling_improved(
        x, y, splits_num=5, random_seed=0,
        outputName='', scaling_method='minmax', test_size=0.4):
    """比较不同重采样方法对CatBoost性能的影响"""
    
    # 重采样方法
    resample_methods = [
        ('SMOTE', SMOTE(random_state=random_seed)),
        ('ADASYN', ADASYN(random_state=random_seed)),
        ('RandomOverSampler', RandomOverSampler(random_state=random_seed)),
        ('SMOTEENN', SMOTEENN(random_state=random_seed)),
        ('TomekLinks', TomekLinks()),
        ('None', 'passthrough')
    ]

    result_list = []

    # 划分训练集和测试集
    X_train_all, X_test_final, y_train_all, y_test_final = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_seed)

    # 交叉验证
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    for method_name, sampler in resample_methods:
        fold_metrics = {'f1': [], 'auc': []}

        # 交叉验证
        for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all)):
            X_train, X_val = X_train_all.iloc[train_index], X_train_all.iloc[val_index]
            y_train, y_val = y_train_all.iloc[train_index], y_train_all.iloc[val_index]

            # Pipeline构建 - 先归一化后重采样
            pipeline_steps = []
            if scaling_method == 'minmax':
                pipeline_steps.append(('scaler', MinMaxScaler()))
            
            if sampler != 'passthrough':
                pipeline_steps.append(('sampler', sampler))
                
            pipeline_steps.append(('classifier', CatBoostClassifier(
                random_seed=random_seed + fold,
                verbose=0,
                thread_count=1,
                allow_writing_files=False)
            ))

            pipeline = Pipeline(pipeline_steps)

            try:
                pipeline.fit(X_train, y_train)
                y_val_pred = pipeline.predict(X_val)
                y_val_prob = pipeline.predict_proba(X_val)[:, 1]

                fold_metrics['f1'].append(f1_score(y_val, y_val_pred, zero_division=0))
                fold_metrics['auc'].append(roc_auc_score(y_val, y_val_prob))

            except Exception as e:
                fold_metrics['f1'].append(0)
                fold_metrics['auc'].append(0.5)

        # 在完整训练集上训练最终模型
        final_pipeline_steps = []
        if scaling_method == 'minmax':
            final_pipeline_steps.append(('scaler', MinMaxScaler()))
            
        if sampler != 'passthrough':
            final_pipeline_steps.append(('sampler', sampler))
            
        final_pipeline_steps.append(('classifier', CatBoostClassifier(
            random_seed=random_seed,
            verbose=0,
            thread_count=1,
            allow_writing_files=False)
        ))

        final_pipeline = Pipeline(final_pipeline_steps)

        try:
            final_pipeline.fit(X_train_all, y_train_all)
            y_test_pred = final_pipeline.predict(X_test_final)
            y_test_prob = final_pipeline.predict_proba(X_test_final)[:, 1]

            test_accuracy = accuracy_score(y_test_final, y_test_pred)
            test_precision = precision_score(y_test_final, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test_final, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test_final, y_test_pred, zero_division=0)
            test_auc = roc_auc_score(y_test_final, y_test_prob)
            test_mcc = matthews_corrcoef(y_test_final, y_test_pred)

            result_list.append({
                'resample_method': method_name,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_F1': test_f1,
                'test_AUC': test_auc,
                'test_MCC': test_mcc
            })

        except Exception as e:
            result_list.append({
                'resample_method': method_name,
                'test_accuracy': 0,
                'test_precision': 0,
                'test_recall': 0,
                'test_F1': 0,
                'test_AUC': 0.5,
                'test_MCC': 0
            })

    # 保存结果
    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values('test_F1', ascending=False)
    result_df = result_df[
        ['resample_method', 'test_F1', 'test_AUC', 'test_accuracy', 'test_precision', 'test_recall', 'test_MCC']]

    output_filename = f"{outputName}_improved.xlsx"
    result_df.to_excel(output_filename, index=False)

    return result_df


# 调用函数
if __name__ == "__main__":
    result = evaluate_cat_with_resampling_improved(
        x, y,
        splits_num=5,
        random_seed=0,
        outputName='improve_new_minmax_6_4_cat_resampling',
        scaling_method='minmax',
        test_size=0.4
    )