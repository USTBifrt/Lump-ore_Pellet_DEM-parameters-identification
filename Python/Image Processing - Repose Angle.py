import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取图像
image_path = r'C:\Users\15482\Desktop\Angle of repose experiment\100Lump\test-2-5-8-9\1.png'
img = cv2.imread(image_path)

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用阈值将图像二值化
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# 获取图像尺寸
height, width = binary.shape

# 定义物理尺寸
box_size_mm = 400  # 盒子的实际尺寸是400mm
num_points = 20  # 我们将x轴等分为20个点

# 计算每个区间的x步长
x_step = width // num_points

# 存储轮廓坐标
pile_contour = []

# 遍历每个x区间，找到最高的y值
for i in range(num_points):
    x_start = i * x_step
    x_end = (i + 1) * x_step

    # 在这个x区间中，找到每列中最高的黑色像素
    max_y = height  # 初始设置为最大值
    for x in range(x_start, x_end):
        column = binary[:, x]
        black_pixels = np.where(column == 255)[0]  # 找到黑色像素的索引

        if len(black_pixels) > 0:
            highest_y = black_pixels[0]  # 找到最高的黑色像素
            max_y = min(max_y, highest_y)  # 找到最小的y值（即最高的像素）

    # 将图像坐标转换为物理尺寸坐标
    x_mm = (x_start + x_end) // 2 * box_size_mm / width  # 取区间的中间点
    y_mm = (height - max_y) * box_size_mm / height  # 反转Y轴映射

    # 只添加X坐标小于300的点
    if x_mm < 300:
        pile_contour.append((x_mm, y_mm))

# 打印轮廓坐标
for coord in pile_contour:
    print(f"x: {coord[0]:.2f} mm, y: {coord[1]:.2f} mm")

# 提取X和Y坐标
x_coords, y_coords = zip(*pile_contour)

# 使用线性回归拟合为直线
coeffs = np.polyfit(x_coords, y_coords, 1)  # 1 表示一阶多项式，即直线
slope = coeffs[0]  # 斜率
intercept = coeffs[1]  # 截距

# 计算斜率的角度
angle_radians = np.arctan(slope)
angle_degrees = np.degrees(angle_radians)

print(f"拟合直线的方程为: y = {slope:.4f}x + {intercept:.4f}")
print(f"直线与水平轴的夹角为: {angle_degrees:.2f}°")

# 绘制料堆轮廓和拟合的直线
plt.figure(figsize=(8, 6))
plt.plot(x_coords, y_coords, marker='o', color='r', label="Pile Contour")
plt.plot(x_coords, np.polyval(coeffs, x_coords), label=f"Fitted Line: {angle_degrees:.2f}°", color='b')

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title(f'Pile Contour with Fitted Line (Angle: {angle_degrees:.2f}°)')
plt.legend()

# 保存图像到读取图像的路径下
output_image_path = os.path.join(os.path.dirname(image_path), 'pile_contour_fitted_line.png')
plt.savefig(output_image_path, dpi=300)  # 可以调整dpi以改变分辨率
plt.close()  # 关闭当前图形，防止显示在屏幕上

print(f"图像已保存到: {output_image_path}")


