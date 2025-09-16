#!/usr/bin/env python
import rospy

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import numpy as np

if __name__ == '__main__':
    
    rospy.init_node('path_node')

    traj = rospy.Publisher('trajectory', Path, queue_size=1)

    path = Path()

    data = np.load('terrain_proj_and_baseline.npz')

    case_num = 0

    goal_location = data['goal_location']
    states_base_global = data['states_base_global']

    x_goal = goal_location[case_num,0]
    y_goal = goal_location[case_num,1]
    z_goal = goal_location[case_num,2]

    x_base = states_base_global[case_num,:,1]
    y_base = states_base_global[case_num,:,0]
    z_base = states_base_global[case_num,:,2]

    for i in range(len(x_base)):
        pose = PoseStamped()
        pose.pose.position.x = y_base[i]
        pose.pose.position.y = x_base[i]
        pose.pose.position.z = z_base[i]
        path.poses.append(pose)

    path.header.frame_id = "map"
    rospy.loginfo("Publlishing Trajectory")
    
    r = rospy.Rate(0.1) # 10hz
    while not rospy.is_shutdown():
        traj.publish(path)

    r.sleep()

    # rospy.spin()