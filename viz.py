import open3d as o3d
import glob
import matplotlib.pyplot as plt
import quick_utils
import numpy as np

intrinsic = o3d.io.read_pinhole_camera_intrinsic("intrinsic.json")

c_imgs = glob.glob("./input/*.jpg")
c_imgs.sort()

d_imgs = glob.glob("./output/*.png")
d_imgs.sort()

# for idx in range(1):
#     color = o3d.io.read_image(c_imgs[idx])
#     depth = o3d.io.read_image(d_imgs[idx])
#     rgbdi = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc=50)
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdi, intrinsic)
#     o3d.visualization.draw_geometries([pcd])

camera_params = {}
camera_params['height'] = 480
camera_params['width'] = 640
camera_params['fx'] = 517.306408
camera_params['fy'] = 516.469215
camera_params['x_offset'] = 318.643040
camera_params['y_offset'] = 255.313989

color = plt.imread(c_imgs[0]).astype(np.float32)/255
depth = plt.imread(d_imgs[0]).astype(np.float32)

depth = 1. / depth

mask = depth.flatten() < 10

xyz = quick_utils.compute_xyz(depth, camera_params)
origin=[0.,0.,0.]

points = xyz.reshape(-1,3)
color = color.reshape(-1, 3)

points = points[mask, :]
color = color[mask, :]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(color)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=origin)
geometries = list([pcd, coordinate_frame])

o3d.visualization.draw_geometries(geometries)
