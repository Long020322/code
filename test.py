import numpy as np
from utils import idx2seq, tensor2tuple, tuple2tensor
from sklearn.cluster import KMeans

original_data = np.load("./data/Abilene.npy")  #(428536, 4)

# 将数组分割为 indices 和 values
indices = original_data[:, :3]  # 取前3列作为索引
values = original_data[:, 3]    # 第4列作为值
indices = indices.astype(int)
# 使用 tuple2tensor 函数转换为三维张量
tensor = tuple2tensor(indices, values)
data = tensor.reshape(144,3000)
data_for_kmeans = data[:, :100]
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(data_for_kmeans)
# 创建路径到标签的映射
path_labels = {}
for i in range(144):
    path_labels[(i // 12, i % 12)] = labels[i]  # 路径是 (行, 列)

# 初始化一个新列以存储聚类标签
label_column = np.full(original_data.shape[0], -1)  # 默认值为 -1

# 遍历原始数据，填充标签列
for i in range(original_data.shape[0]):
    path = (original_data[i, 0], original_data[i, 1])  # 获取路径
    if path in path_labels:
        label_column[i] = path_labels[path]  # 填充标签

# 将标签添加到原始数据中
new_data = np.column_stack((original_data, label_column))  # 合并
