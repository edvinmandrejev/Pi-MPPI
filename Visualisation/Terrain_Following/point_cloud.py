#!/usr/bin/env python
import rospy
import math
import sys

from sensor_msgs.msg import PointCloud2,PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header

import numpy as np
import open3d as o3d

def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )

if __name__ == '__main__':
    '''
    Sample code to publish a pcl2 with python
    '''
    rospy.init_node('pcl2_pub_example')
    pcl_pub = rospy.Publisher("/my_pcl_topic", PointCloud2, queue_size=1)
    rospy.loginfo("Initializing sample pcl2 publisher node...")

    pcd = o3d.io.read_point_cloud("terrain_new.ply")
    # downpcd = pcd.uniform_down_sample(5)
    # xyz = np.asarray(downpcd.points)
    xyz = np.asarray(pcd.points)
    print(xyz.shape)

    rgba = np.ones((xyz.shape[0], 4))
    rgba[:, :3] = rgba[:, :3]*0.0
    xyzrgba = np.hstack((xyz, rgba))
    pc_msg = point_cloud(xyzrgba, "map")

    #give time to roscore to make the connections
    rospy.sleep(1.)
    #header
    # header = std_msgs.msg.Header()
    # header.stamp = rospy.Time.now()
    # header.frame_id = 'map'
    # #create pcl from points
    # scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
    #publish    
    rospy.loginfo("happily publishing sample pointcloud.. !")
    
    # r = rospy.Rate(0.1) # 10hz
    # while not rospy.is_shutdown():
    pcl_pub.publish(pc_msg)
    
    # r.sleep()

    rospy.spin()