
import  torch

# # ck = torch.load("lightning_logs/version_66/checkpoints/epoch=9-step=6780.ckpt")
# ck = torch.load(r"H:\myProject\python\YOLOV-master\best_ckpt.pth")
#
# print(ck["model"].keys()
#
#       )


# import numpy as np
# from scipy.stats import zscore
#
# data = np.array([1] * 49 + [-1] + [1] * 50)
# mean_value = np.mean(data)
# std_dev = np.std(data)
# element_50 = data[49]
# grubbs_statistic = abs(element_50 - mean_value) / std_dev
# sample_size = len(data)
# alpha = 0.05  # 显著性水平
# critical_value = (sample_size - 1) / np.sqrt(sample_size) * np.sqrt(
#     zscore(np.arange(1, sample_size + 1), ddof=1).max()**2 / (sample_size - 2 + zscore(np.arange(1, sample_size + 1), ddof=1).max()**2)
# )
# is_significant = grubbs_statistic > critical_value
#
# if is_significant:
#     print("第五十个元素为-1是显著的异常值。")
# else:
#     print("第五十个元素为-1不是显著的异常值。")

# from sklearn.ensemble import IsolationForest
# import numpy as np
# data = np.array([1] * 49 + [-1]*5 + [1] * 50)
# data = data.reshape(-1, 1)  # 将数据转换为二维数组
# clf = IsolationForest(contamination=0.1)  # contamination参数表示异常值的比例，根据实际情况调整
# clf.fit(data)
# predictions = clf.predict(data)
# is_anomaly = predictions[49] == -1
#
# if is_anomaly:
#     print("第五十个元素为-1是异常值。")
# else:
#     print("第五十个元素为-1不是异常值。")
#
# from sklearn.svm import OneClassSVM
# import numpy as np
# clf = OneClassSVM(nu=0.01)  # nu参数表示异常值的上限比例，根据实际情况调整
# clf.fit(data)
# predictions = clf.predict(data)
# is_anomaly = predictions[49] == -1
#
# if is_anomaly:
#     print("第五十个元素为-1是异常值。")
# else:
#     print("第五十个元素为-1不是异常值。")




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
#
# # 准备数据
# data = np.array([1] * 49 + [-1] + [1] * 50, dtype=np.float32)
# data = data.reshape(-1, 1)  # 将数据转换为二维数组
#
# # 定义 Autoencoder 模型
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(1, 1, bias=False),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(1, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
# # 初始化模型、损失函数和优化器
# model = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 将数据转换为 PyTorch 的 Tensor
# data_tensor = torch.from_numpy(data)
#
# # 训练 Autoencoder 模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(data_tensor)
#     loss = criterion(outputs, data_tensor)
#     loss.backward()
#     optimizer.step()
#
# # 使用 Autoencoder 模型进行异常检测
# model.eval()
# with torch.no_grad():
#     predictions = model(data_tensor)
#
#     mse = ((data_tensor - predictions)**2).numpy()
#
# # 设定阈值，根据实际情况调整
# threshold = 0.01
# is_anomaly = mse[49] > threshold
#
# if is_anomaly:
#     print("第五十个元素为-1是异常值。")
# else:
#     print("第五十个元素为-1不是异常值。")
#


import torch
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
#
# # 准备数据
# data = np.array([1] * 49 + [-1] + [1] * 50, dtype=np.float32)
# data = data.reshape(-1, 1)  # 将数据转换为二维数组
#
# # 标准化数据
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)
#
# # 使用 DBSCAN 进行聚类
# dbscan = DBSCAN(eps=0.5, min_samples=3)  # 根据实际情况调整参数
# labels = dbscan.fit_predict(data_scaled)
#
# # 判断第五十个元素是否为噪声点
# is_noise = labels[49] == -1
#
# if is_noise:
#     print("第五十个元素为-1是噪声点。")
# else:
#     print("第五十个元素为-1不是噪声点。")






import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# 准备数据
data = np.array([1] * 49 + [-1]*20 + [1] * 50, dtype=np.float32)
data = data.reshape(-1, 1)  # 将数据转换为二维数组

# 使用局部异常因子 (LOF) 进行异常检测
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)  # 调整参数以适应数据
predictions = lof.fit_predict(data)

# 判断第五十个元素是否为异常点
is_anomaly = predictions[49] == -1

if is_anomaly:
    print("第五十个元素为-1是异常值。")
else:
    print("第五十个元素为-1不是异常值。")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 准备数据
data = np.array([1] * 49 + [-1]*20 + [1] * 50, dtype=np.float32)

# 使用 seaborn 绘制核密度估计图
sns.kdeplot(data, fill=True)
plt.title("Kernel Density Estimation for Data")
plt.show()

# 判断第五十个元素是否为异常点
kde_values = sns.kdeplot(data).get_lines()[0].get_data()
density_at_50th = np.interp(49, kde_values[0], kde_values[1])

if density_at_50th < 0.01:  # 根据实际情况调整密度的阈值
    print("第五十个元素为-1是异常值。")
else:
    print("第五十个元素为-1不是异常值。")
