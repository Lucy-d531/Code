# feature_density_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300

# 读取数据
data = pd.read_excel('Darknet_Market_processed_filtered.xlsx')

# 要绘图的特征列
features = ["close_centrality", "avg_in_degree", "min_interval", "input_centrality"]

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# 浅蓝色和浅橙色
colors = ['#87CEEB', '#FFD8A6']
line_styles = ['-', '--']
line_widths = [2.5, 2]

# 循环生成图
for i, feature in enumerate(features):
    ax = axes[i]

    # 为每个类别分别绘制核密度估计图
    for label_val, color, line_style, lw in zip([0, 1], colors, line_styles, line_widths):
        subset = data[data['label'] == label_val]
        sns.kdeplot(
            data=subset,
            x=feature,
            color=color,
            linestyle=line_style,
            linewidth=lw,
            alpha=0.9,
            label=f'Label {label_val}',
            ax=ax,
            fill=True
        )

    # 美化图形
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

    # 设置纯白色背景
    ax.set_facecolor('white')

# 调整布局
plt.tight_layout(pad=3.0)

# 保存图片，确保背景为白色
plt.savefig('kde_plots_white_bg.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()