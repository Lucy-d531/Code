# learning_curves_comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import os
from matplotlib.ticker import FormatStrFormatter

# 设置字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
x = data.drop('label', axis=1)
y = data['label']

# 定义分类器字典
classifiers = {
    'Ridge': RidgeClassifier(random_state=0),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=0),
    'RandomForest': RandomForestClassifier(random_state=0),
    'ExtraTrees': ExtraTreesClassifier(random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(random_state=0, probability=True),
    'XGBoost': XGBClassifier(random_state=0),
    'IABT-DW': CatBoostClassifier(
        iterations=100,
        learning_rate=0.15,
        depth=5,
        l2_leaf_reg=0.5,
        min_data_in_leaf=3,
        verbose=False,
        random_seed=0
    )
}

# 创建输出目录
output_dir = "Learning_Curves_SMOTE_MinMax"
os.makedirs(output_dir, exist_ok=True)

# 设置公共参数
random_seed = 0
cv = 5
train_sizes = np.linspace(0.1, 1.0, 5)
cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)

# 为每个分类器生成学习曲线
for model_name, model in classifiers.items():
    try:
        print(f"生成 {model_name} 的学习曲线...")

        # 构建管道 - 使用归一化和SMOTE过采样
        steps = [
            ('scaler', MinMaxScaler()),
            ('sampler', SMOTE(random_state=random_seed)),
            ('classifier', model)
        ]
        pipeline = Pipeline(steps=steps)

        # 计算学习曲线
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline,
            x,
            y,
            train_sizes=train_sizes,
            cv=cv_strategy,
            scoring='f1',
            n_jobs=-1,
            shuffle=True,
            random_state=random_seed,
            verbose=0
        )

        # 计算平均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # 创建图表
        plt.figure(figsize=(8, 6))

        # 绘制学习曲线
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_mean - val_std,
                         val_mean + val_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, val_mean, 'o-', color="g", label="Cross-validation score")

        # 设置图表属性
        plt.title(f"{model_name}", fontsize=16)
        plt.xlabel("Training examples", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.legend(loc="best", fontsize=10)

        # 设置Y轴范围
        plt.ylim(0.984, 1.001)
        plt.yticks(np.arange(0.984, 1.0, 0.002))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # 保存图像
        filename = os.path.join(output_dir, f"{model_name}_learning_curve_MinMax.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  {model_name} 学习曲线已保存: {filename}")

    except Exception as e:
        print(f"  生成 {model_name} 学习曲线时出错: {str(e)}")
        plt.close()

print("\n所有学习曲线生成完成！")
print(f"结果已保存到: {output_dir}")