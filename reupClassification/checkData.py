import pandas as pd

# 读取数据
data = pd.read_csv('trainSet_reUp.csv')

# 查看数据的前几行
print(data.head())

# 提取特征和标签
X = data[['x', 'y']].values
y = data['label'].values

# 检查每一类的点的数量
class_counts = data['label'].value_counts()

print("每一类的点的数量:")
print(class_counts)

# 可视化数据
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='purple', s=1, label='Class 1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=1, label='Class 2')
plt.scatter(X[y == 2, 0], X[y == 2, 1], c='orange', s=1, label='Class 3')

# 画出两个圆
circle1 = plt.Circle((0, 0), 0.4, color='black', fill=False)
circle2 = plt.Circle((0, 0), 0.8, color='black', fill=False)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.plot([-1, 1], [-1, 1], color='black', linewidth=1)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.savefig('trainSet_reUp.png')
plt.show()
