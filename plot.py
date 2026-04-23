import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = 'fusion_gat_gru_tsne_2d.csv'
OUTPUT_PNG = 'fusion_gat_gru_tsne_beautified.png'

print(f"正在读取数据文件: {INPUT_CSV} ...")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"错误：找不到文件 {INPUT_CSV}，请确认文件路径是否正确。")
    exit()

labels = df['label'].values
vectors_2d = df[['dim1', 'dim2']].values

print("正在生成美化后的图片...")

# 分开取两类数据
normal_x = vectors_2d[labels == 0, 0]
normal_y = vectors_2d[labels == 0, 1]
attack_x = vectors_2d[labels == 1, 0]
attack_y = vectors_2d[labels == 1, 1]

plt.figure(figsize=(9, 6.8), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# 先画 Normal，颜色柔和一些
plt.scatter(
    normal_x, normal_y,
    c='#4C72B0',          # 柔和蓝
    alpha=0.38,
    s=28,
    label='Normal',
    edgecolors='white',
    linewidths=0.25,
    zorder=2
)

# 再画 Attack，避免过亮过炸
plt.scatter(
    attack_x, attack_y,
    c='#C44E52',          # 低饱和砖红
    alpha=0.58,
    s=24,
    marker='^',
    label='Attack',
    edgecolors='white',
    linewidths=0.3,
    zorder=3
)

plt.title('t-SNE Visualization of Fusion-GAT-GRU Features', fontsize=16, pad=12)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)

plt.legend(
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='#CCCCCC',
    fontsize=11
)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"图片已保存为: {OUTPUT_PNG}")
plt.show()