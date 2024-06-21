import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_circular_data(num_points_list, inner_radius=0.4, outer_radius=0.8, filter_condition=None):
    assert len(num_points_list) == 3, "num_points_list must contain三个元素."

    def generate_points(num_points, radius_start, radius_end):
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = (radius_end - radius_start) * np.sqrt(np.random.uniform(0, 1, num_points)) + radius_start
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def generateGivenNumberPoints(num_points, radius_start, radius_end, class_type=None):
        x_all, y_all = [], []

        while len(x_all) < num_points:

            if class_type == 2:
                # 生成正方形区域内的数据点
                x = np.random.uniform(-1, 1, num_points * 2)
                y = np.random.uniform(-1, 1, num_points * 2)
            else:
                x, y = generate_points(num_points, radius_start, radius_end)

            mask = None
            if class_type == 0:
                mask = np.sqrt(x**2 + y**2) <= inner_radius
            elif class_type == 1:
                mask = np.sqrt(x**2 + y**2) <= outer_radius
            elif class_type == 2:
                mask = np.sqrt(x**2 + y**2) > outer_radius

            x = x[mask]
            y = y[mask]

            if filter_condition:
                mask = filter_condition(x, y)
                x = x[mask]
                y = y[mask]

            # 只保留所需数量的点
            x_all.extend(x[:num_points - len(x_all)])
            y_all.extend(y[:num_points - len(y_all)])

        return np.array(x_all), np.array(y_all)

    # 生成第一类数据：内圆 (半径 inner_radius)
    x_inner, y_inner = generateGivenNumberPoints(num_points_list[0], 0, inner_radius, 0)
    # 生成第二类数据：外圆和内圆之间 (半径 inner_radius 到 outer_radius)
    x_middle, y_middle = generateGivenNumberPoints(num_points_list[1], inner_radius, outer_radius, 1)
    # 生成第三类数据：外圆之外直到边界
    x_outer, y_outer = generateGivenNumberPoints(num_points_list[2], outer_radius, 1, 2)

    # 合并所有数据点
    x_all = np.concatenate([x_inner, x_middle, x_outer])
    y_all = np.concatenate([y_inner, y_middle, y_outer])
    labels_all = np.concatenate([np.zeros(len(x_inner)), np.ones(len(x_middle)), np.full(len(x_outer), 2)])

    return np.array(x_all), np.array(y_all), np.array(labels_all)

# 定义整个右下三角形区域的过滤条件
def right_bottom_triangle(x, y):
    return y <= x


if __name__ == "__main__":

    # 生成数据
    num_points_list = [126, 377, 497]  # 每类数据点的数量，占比约12.57: 37.70: 49.73
    x, y, labels = generate_circular_data(num_points_list, filter_condition=right_bottom_triangle)

    # 保存数据到文件
    data = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    data.to_csv('./reupClassification/trainSet_reUp.csv', index=False)

    # 绘制数据
    plt.figure(figsize=(6, 6))
    plt.scatter(x[labels == 0], y[labels == 0], c='purple', s=1, label='Class 1')
    plt.scatter(x[labels == 1], y[labels == 1], c='red', s=1, label='Class 2')
    plt.scatter(x[labels == 2], y[labels == 2], c='orange', s=1, label='Class 3')

    # 画出两个圆
    circle1 = plt.Circle((0, 0), 0.4, color='black', fill=False)
    circle2 = plt.Circle((0, 0), 0.8, color='black', fill=False)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)

    # 画出右下三角形区域
    plt.plot([-1, 1], [-1, 1], color='black', linewidth=1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

    class_counts = data['label'].value_counts()
    print("每一类的点的数量:")
    print(class_counts)
