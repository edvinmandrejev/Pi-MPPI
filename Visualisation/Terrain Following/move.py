#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf

import numpy as np


def handle_turtle_pose():
    br = tf.TransformBroadcaster()
    
    data = np.load('terrain_baseline_cov.npz')

    case_num = 0

    states_base_global = data['states_base_global']
    controls_base = data['controls_base']

    x_base = states_base_global[case_num,:,1]
    y_base = states_base_global[case_num,:,0]
    z_base = states_base_global[case_num,:,2]
    psi_base = states_base_global[case_num,:,3]

    roll_base = controls_base[case_num,:,2]
    pitch_base = controls_base[case_num,:,1]

    for i in range (x_base.shape[0]):

        br.sendTransform((y_base[i], x_base[i], z_base[i]+ 0.2*0),
                        tf.transformations.quaternion_from_euler(roll_base[i], -pitch_base[i], psi_base[i]),
                        rospy.Time.now(),
                        "cessna_c172__body",
                        "map")
        rospy.loginfo("step")
        rospy.sleep(0.2)



if __name__ == '__main__':
    rospy.init_node('move')
    print("hello")
    handle_turtle_pose()
    print("done")
    rospy.spin()