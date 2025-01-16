from stl import mesh
import numpy as np
import trimesh
from sklearn.cluster import KMeans  # 用于聚类

# 读取 STL 文件并提取点云数据
def read_stl_file(stl_file):
    stl_mesh = mesh.Mesh.from_file(stl_file)
    points = np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0)  # 提取唯一的点云坐标
    return points

# 使用 KMeans 聚类减少球体数量
def cluster_points(points, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(points)  # 显式设置 n_init
    cluster_centers = kmeans.cluster_centers_  # 聚类中心作为球体中心
    return cluster_centers

# 基于聚类结果生成球体模型
def generate_spheres(cluster_centers, min_radius=2.0, max_radius=5.0):
    spheres = []
    for center in cluster_centers:
        # 为每个聚类中心生成球体
        radius = np.random.uniform(min_radius, max_radius)  # 随机选择半径
        sphere = {
            'center': center,
            'radius': radius
        }
        spheres.append(sphere)
    return spheres

# 保存球体信息到 TXT 文件
def save_spheres_to_txt(spheres, output_file):
    # 打开文件进行写入
    with open(output_file, 'w') as f:
        # 写入每个球体的中心坐标和半径
        f.write("Center_X Center_Y Center_Z Radius\n")  # 写入标题
        for sphere in spheres:
            center = sphere['center']
            radius = sphere['radius']
            # 写入中心坐标和半径
            f.write(f"{center[0]} {center[1]} {center[2]} {radius}\n")


# 主函数：读取 STL 文件，生成球体模型，并保存球体信息到 TXT 文件
if __name__ == "__main__":
    # STL 文件路径
    stl_file = r'C:\Users\15482\Desktop\休止角实验\块矿形状扫描\模型6\40mm-2#.stl'

    # 读取点云
    points = read_stl_file(stl_file)

    # 使用 KMeans 聚类减少球体数量，假设我们希望有 100 个球体
    cluster_centers = cluster_points(points, n_clusters=40)

    # 生成球体模型，增大半径
    spheres = generate_spheres(cluster_centers, min_radius=3, max_radius=8)

    # 输出文件路径
    output_file = r'C:\Users\15482\Desktop\休止角实验\块矿形状扫描\模型6\40mm-2#-点云.txt'

    # 保存球体信息到 txt 文件
    save_spheres_to_txt(spheres, output_file)

    print(f"球体信息已保存至 {output_file}")




