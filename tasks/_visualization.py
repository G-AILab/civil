import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # 假设你的表征数据存储在变量 'features' 中
# features = np.random.rand(1024, 128, 128)

# # 使用全局平均池化将每个样本转换成一个向量
# features = features.reshape(features.shape[0],-1)


def t_sne_visual(features,labels,dataset="HAR"):
    print("start to draw ")
    # 创建一个 TSNE 对象，将数据降维到 2 维
    tsne = TSNE(n_components=2, random_state=42)

    # 使用 TSNE 对数据进行降维
    embeddings = tsne.fit_transform(features)

    # 定义每个类别的颜色
    colors = ['#CC2B52', '#37AFE1', '#3D0301','#FF6500', '#FFF100', '#7E60BF']

    # 画图
    plt.figure(figsize=(8, 6))
    for i in range(np.unique(labels).size):
        # 获取属于类别 i 的样本索引
        indices = np.where(labels == i)[0]
        # 使用 scatter 函数绘制类别 i 的样本
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[i], label=f"类别 {i}")



    plt.title(f"{dataset}")
    plt.savefig(f"exp_fig/{dataset}.png")

# labels = np.random.randint(0, 2, size=features.shape[0])

# t_sne_visual(features,labels)

