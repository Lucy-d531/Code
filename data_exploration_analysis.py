# data_exploration_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# 创建保存图形的目录
save_dir = "数据探索分析_plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 过滤警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取数据
print("=== 读取数据 ===")
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')
print(f"数据集形状: {data.shape}")

# 基本数据信息
print("\n=== 基本数据信息 ===")
print(data.info())

# 查看标签分布
print("\n=== 标签分布 ===")
label_counts = data['label'].value_counts()
print(label_counts)
if len(label_counts) > 1:
    print(f"标签比例: {label_counts.iloc[1] / label_counts.iloc[0]:.2f}")
else:
    print("只有一个标签类别")

# 数据描述性统计
print("\n=== 数值特征描述性统计 ===")
print(data.describe())

# 检查缺失值
print("\n=== 缺失值检查 ===")
missing_data = data.isnull().sum()
print(f"缺失值总数: {missing_data.sum()}")

# 可视化分析
print("\n=== 开始可视化分析 ===")

# 创建可视化图表
plt.figure(figsize=(20, 15))

# 标签分布饼图
plt.subplot(2, 3, 1)
plt.pie(label_counts.values, labels=['Darknet', 'Non-Darknet'], autopct='%1.1f%%', startangle=90)
plt.title('Label Distribution')

# 选择几个重要特征的分布直方图
important_features = ['avg_path_length', 'pearson_corr', 'avg_in_degree','lifetime']
for i, feature in enumerate(important_features, 2):
    plt.subplot(2, 3, i)
    data[feature].hist(bins=50, alpha=0.7)
    plt.title(f'{feature}')
    plt.ylabel('Frequency')
    plt.grid(False)

# 特征与标签的关系箱线图
plt.subplot(2, 3, 6)
sample_features = ['input_centrality', 'lifetime', 'avg_in_degree','in_degree']

# 检查数据尺度差异
def check_scale_difference(data, features):
    """检查特征的尺度差异"""
    scales = []
    for feature in features:
        if data[feature].std() > 0:
            cv = data[feature].std() / data[feature].mean()
            scales.append(cv)
        else:
            scales.append(0)

    max_cv = max(scales)
    min_cv = min([cv for cv in scales if cv > 0]) if any(scales) else 1
    scale_ratio = max_cv / min_cv if min_cv > 0 else float('inf')

    return scale_ratio > 10

# 根据尺度差异决定是否进行对数变换
need_log_transform = check_scale_difference(data, sample_features)

# 准备数据用于箱线图
plot_data = []
feature_names = []

for i, feature in enumerate(sample_features):
    if need_log_transform:
        feature_values = np.log1p(data[feature])
        display_name = f'{feature} (log)'
    else:
        feature_values = data[feature]
        display_name = feature

    feature_names.append(display_name)

    for label_val in data['label'].unique():
        subset_indices = data['label'] == label_val
        for value in feature_values[subset_indices]:
            plot_data.append({
                'value': value,
                'label': label_val,
                'feature': display_name
            })

plot_df = pd.DataFrame(plot_data)

sns.boxplot(data=plot_df, x='feature', y='value', hue='label')
plt.title(f'Feature-Label Relationship{" (Log Scale)" if need_log_transform else ""}')
plt.xticks(rotation=45)
plt.xlabel('')
plt.tight_layout()
plt.savefig(f'{save_dir}/探索性分析.png', dpi=300, bbox_inches='tight')
plt.close()

# 相关性分析
print("\n=== 相关性分析 ===")

# 计算特征与标签的相关性
correlation_with_label = data.corr()['label'].sort_values(ascending=False)
print("\n特征与标签相关性排序（前10和后10）:")
print("Top 10 positive correlations:")
print(correlation_with_label.head(10))
print("\nTop 10 negative correlations:")
print(correlation_with_label.tail(10))

# 绘制相关性热力图（显示所有特征）
plt.figure(figsize=(16, 14))
correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0)
plt.tight_layout()
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig(f'{save_dir}/correlation_heatmap_all.png', dpi=300, bbox_inches='tight')
print(f"已保存相关性热力图: {save_dir}/correlation_heatmap_all.png")
plt.close()

# 处理高度相关特征（相关系数大于等于0.97的仅保留一个）
print("\n=== 处理高度相关特征 ===")

def remove_highly_correlated_features(df, threshold=0.97):
    """移除高度相关的特征，仅保留一个"""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = []
    for col in upper_tri.columns:
        high_corr_features = upper_tri[col][upper_tri[col] >= threshold].index.tolist()
        for feature in high_corr_features:
            high_corr_pairs.append((col, feature, upper_tri.loc[col, feature]))

    if not high_corr_pairs:
        print("没有发现高度相关的特征对")
        return df, []

    print(f"发现 {len(high_corr_pairs)} 对高度相关的特征 (相关系数 >= {threshold}):")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")

    to_remove = set()

    if 'label' in df.columns:
        label_correlations = df.corr()['label'].abs()

    for col1, col2, corr_value in high_corr_pairs:
        if col1 in to_remove or col2 in to_remove:
            continue

        if 'label' in df.columns:
            corr1 = label_correlations.get(col1, 0)
            corr2 = label_correlations.get(col2, 0)

            if corr1 >= corr2:
                to_remove.add(col2)
                print(f"  移除 {col2} (保留 {col1}, 与标签相关性: {corr1:.4f} vs {corr2:.4f})")
            else:
                to_remove.add(col1)
                print(f"  移除 {col1} (保留 {col2}, 与标签相关性: {corr1:.4f} vs {corr2:.4f})")
        else:
            col1_mean_corr = corr_matrix[col1].mean()
            col2_mean_corr = corr_matrix[col2].mean()

            if col1_mean_corr <= col2_mean_corr:
                to_remove.add(col2)
                print(f"  移除 {col2} (保留 {col1})")
            else:
                to_remove.add(col1)
                print(f"  移除 {col1} (保留 {col2})")

    df_filtered = df.drop(columns=to_remove)

    print(f"\n原始特征数量: {len(df.columns)}")
    print(f"过滤后特征数量: {len(df_filtered.columns)}")
    print(f"移除的特征数量: {len(to_remove)}")
    print(f"移除的特征: {list(to_remove)}")

    return df_filtered, list(to_remove)

# 应用高度相关特征过滤
data_filtered, removed_features = remove_highly_correlated_features(data, threshold=0.97)

# 绘制过滤后的特征的相关性热力图
plt.figure(figsize=(16, 14))
correlation_matrix_filtered = data_filtered.corr()
mask = np.triu(np.ones_like(correlation_matrix_filtered, dtype=bool))

sns.heatmap(correlation_matrix_filtered, mask=mask, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0, cbar_kws={"shrink": .8})
plt.title('Correlated Features Heatmap (After Removing Highly Correlated Features)')
plt.tight_layout()
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig(f'{save_dir}/correlation_heatmap_filtered.png', dpi=300, bbox_inches='tight')
print(f"已保存过滤后的相关性热力图: {save_dir}/correlation_heatmap_filtered.png")
plt.close()

# 检查过滤后的相关性矩阵中是否还有高度相关的特征
print("\n=== 检查过滤后的相关性 ===")
corr_matrix_filtered = data_filtered.corr().abs()
upper_tri_filtered = corr_matrix_filtered.where(np.triu(np.ones(corr_matrix_filtered.shape), k=1).astype(bool))
high_corr_remaining = (upper_tri_filtered >= 0.97).sum().sum()
print(f"过滤后仍然高度相关(>=0.97)的特征对数量: {high_corr_remaining}")

if high_corr_remaining > 0:
    print("仍然存在高度相关的特征对:")
    high_corr_pairs = []
    for col in upper_tri_filtered.columns:
        high_corr = upper_tri_filtered[col][upper_tri_filtered[col] >= 0.97].index.tolist()
        for feature in high_corr:
            high_corr_pairs.append((col, feature, upper_tri_filtered.loc[col, feature]))

    for pair in high_corr_pairs:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")

# 异常值检查
print("\n=== 异常值检查 ===")
numeric_data = data_filtered.select_dtypes(include=[np.number])
numeric_features = [col for col in numeric_data.columns if col != 'label']
numeric_data_features = data_filtered[numeric_features]

Q1 = numeric_data_features.quantile(0.25)
Q3 = numeric_data_features.quantile(0.75)
IQR = Q3 - Q1
outliers = ((numeric_data_features < (Q1 - 1.5 * IQR)) | (numeric_data_features > (Q3 + 1.5 * IQR))).sum()
print("所有特征的异常值数量:")
print(outliers.sort_values(ascending=False))

print("\n=== EDA完成 ===")
print("关键发现总结:")
print(f"1. 数据集大小: {data.shape}")
print(f"2. 标签分布: 暗网 {label_counts.iloc[1]} 个, 非暗网 {label_counts.iloc[0]} 个")

if len(correlation_with_label) > 1:
    positive_corr = correlation_with_label[correlation_with_label.index != 'label'].nlargest(1)
    negative_corr = correlation_with_label.nsmallest(1)

    if not positive_corr.empty:
        print(f"3. 最正相关特征: {positive_corr.index[0]} (相关系数: {positive_corr.iloc[0]:.3f})")
    if not negative_corr.empty:
        print(f"4. 最负相关特征: {negative_corr.index[0]} (相关系数: {negative_corr.iloc[0]:.3f})")

print(f"5. 高度相关特征处理: 移除了 {len(removed_features)} 个特征")
print(f"6. 过滤后数据集: {data_filtered.shape}")

print(f"\n所有图形已保存到 '{save_dir}' 目录中:")
for file in os.listdir(save_dir):
    if file.endswith(('.png', '.pdf')):
        print(f"  - {save_dir}/{file}")