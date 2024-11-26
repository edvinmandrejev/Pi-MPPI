import numpy as np 
import time
from mav_viewer_obstacle import mav_viewer

case_num = 0
timesteps = 150

data = np.load('./example_data/comparison_base_proj_v2.npz')
states_proj_global = data['states_proj_global'] # x, y, z, heading
controls_proj = data['controls_proj']           # vel, pitch, roll

states_base_global = data['states_base_global'] # x, y, z, heading
controls_base = data['controls_base']           # vel, pitch, roll

obs_x_batch = data['obs_x_batch']
obs_y_batch = data['obs_y_batch']
obs_z_batch = data['obs_z_batch']
obs_r_batch = data['obs_r_batch']


obs_x = obs_x_batch[case_num,:]
obs_y = obs_y_batch[case_num,:]
obs_z = - obs_z_batch[case_num,:]
obs_r = obs_r_batch[case_num,:]

mav_view = mav_viewer(obs_x, obs_y, obs_z, obs_r)

class mav_state_proj:
    def __init__(self,x,y,z,roll,pitch,heading):
        self.pn = x
        self.pe = y
        self.h  = z
        self.phi = roll
        self.theta = pitch
        self.psi = heading
mav_states_proj = mav_state_proj(0,0,0,0,0,0)

class mav_state_base:
    def __init__(self,x,y,z,roll,pitch,heading):
        self.pn = x
        self.pe = y
        self.h  = z
        self.phi = roll
        self.theta = pitch
        self.psi = heading
mav_states_base = mav_state_base(0,0,0,0,0,0)


for i in range(timesteps):
    ###### Proj
    mav_states_proj.pn    = states_proj_global[case_num,i,0]
    mav_states_proj.pe    = states_proj_global[case_num,i,1]
    mav_states_proj.h     = -states_proj_global[case_num,i,2]

    mav_states_proj.phi   = controls_proj[case_num,i,2]
    mav_states_proj.theta = controls_proj[case_num,i,1]
    mav_states_proj.psi   = states_proj_global[case_num,i,3]


    ###### Base
    mav_states_base.pn    = states_base_global[case_num,i,0]
    mav_states_base.pe    = states_base_global[case_num,i,1]
    mav_states_base.h     = -states_base_global[case_num,i,2]

    mav_states_base.phi   = controls_base[case_num,i,2]
    mav_states_base.theta = controls_base[case_num,i,1]
    mav_states_base.psi   = states_base_global[case_num,i,3]


    mav_view.update(mav_states_proj,mav_states_base)  # plot body of MAV
    time.sleep(.1)








