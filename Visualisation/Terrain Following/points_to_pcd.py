# import os
# current_working_directory = os.getcwd()
# print(current_working_directory)

import numpy as np
import open3d as o3d

terrain_data = np.load('terrain2.npz')
x = terrain_data['x']
y = terrain_data['y']
z = terrain_data['z']

xyz = np.zeros((np.size(x), 3))
xyz[:, 0] = np.reshape(y, -1)
xyz[:, 1] = np.reshape(x, -1)
xyz[:, 2] = np.reshape(z, -1)

print(xyz.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

o3d.io.write_point_cloud("terrain_new.ply", pcd)
# downpcd = pcd.uniform_down_sample(10)

# print(np.asarray(downpcd.points).shape)

# o3d.visualization.draw_geometries([downpcd])