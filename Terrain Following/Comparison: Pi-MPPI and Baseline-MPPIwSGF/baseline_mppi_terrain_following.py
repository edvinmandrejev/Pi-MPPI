import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt 

import bernstein_coeff_order10_arbitinterval
from functools import partial
from jax import jit, random,vmap
import jax
import jax.lax as lax
import jax.scipy as js
from jax.scipy.ndimage import map_coordinates

class MPPI_base():
    def __init__(self,num,
                  v_max, v_min, vdot_max, vdot_min, vddot_max,vddot_min,
                    pitch_max,pitch_min, pitchdot_max, pitchdot_min, pitchddot_max, pitchddot_min,
                      roll_max, roll_min, rolldot_max, rolldot_min, rollddot_max, rollddot_min,
                      scale, octaves, persistence, lacunarity, width, height, altitude_scale, altitude_offset,terrain_height,x_range,y_range,num_batch):
        # Altitude constrins

        # Smoothening
        self.window_size = 20

        # Weights
        self.w_1 = 8            # goal reaching weight 1
        self.w_2 = .001            # mppi weight 0.001
        self.w_3_1 = 50000            # min altitude cost 1000
        self.w_3_2 = 50000         # desired altitude cost 10 
        self.w_4 = 100           # vdot weight 100
        self.w_5 = 100            # pitchdot weight 100
        self.w_6 = 100            # roll weight 100
        self.w_7 = 50000           # vddot weight 500
        self.w_8 = 10000          # pitchddot weight 100
        self.w_9 = 10000           # rollddot weight 10000
 

        ### Constraints ###
        self.d_0 = 5
        self.z_des = 15

        self.v_min = v_min
        self.v_max = v_max

        self.vdot_max = vdot_max
        self.vdot_min = vdot_min

        self.vddot_max = vddot_max
        self.vddot_min = vddot_min

        self.pitch_max = pitch_max
        self.pitch_min = pitch_min

        self.pitch_dot_max = pitchdot_max
        self.pitch_dot_min = pitchdot_min

        self.pitchddot_max = pitchddot_max
        self.pitchddot_min = pitchddot_min

        self.roll_max = roll_max
        self.roll_min = roll_min

        self.rolldot_max = rolldot_max
        self.rolldot_min = rolldot_min

        self.rollddot_max = rollddot_max
        self.rollddot_min = rollddot_min

        self.g = 9.81
        self.num_batch = num_batch
        self.num = num
        self.t = 0.2

        # Initialization of mean and cov
        mean_v = jnp.zeros(self.num) 
        mean_pitch = jnp.zeros(self.num)
        mean_roll = jnp.zeros(self.num) 
        self.mean = jnp.hstack(( mean_v, mean_pitch, mean_roll ))

        ### Covariance ###

        vel_coef_cov = 0.008
        pitch_coef_cov = 0.006
        roll_coef_cov = 0.0015

        cov_v_control = jnp.identity(self.num)*vel_coef_cov
        cov_control_pitch = jnp.identity(self.num)*pitch_coef_cov
        cov_control_roll = jnp.identity(self.num)*roll_coef_cov

        self.cov = jax.scipy.linalg.block_diag(cov_v_control, cov_control_pitch,cov_control_roll)
        self.sigma = jnp.diag(jnp.array([vel_coef_cov,pitch_coef_cov,roll_coef_cov]))

        # Dot value init
        self.vdot_init = 0
        self.pitchdot_init = 0
        self.rolldot_init = 0

        self.beta = 5

        ### MPPI Parameters ###
        self.param_exploration = 0.0  # constant parameter of mppi
        self.param_lambda = 50  # constant parameter of mppi
        self.param_alpha = 1.0 # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.stage_cost_weight = 1
        self.terminal_cost_weight = 1

        ############################### Terrain initialisation #####################
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.width = width
        self.height = height

        self.altitude_scale = altitude_scale
        self.altitude_offset = altitude_offset

        self.terrain = terrain_height
        self.x_range = x_range
        self.y_range = y_range
       
        # initialization
        self.compute_cost_mppi_batch = jit(vmap(self.compute_cost_mppi,in_axes = (1,None,None,None,1,1,1,1,1,1,1,1,1,1)))
        self.compute_weights_batch = jit(vmap(self._compute_weights, in_axes = ( 0, None, None )  ))
        self.compute_epsilon_batch = jit(vmap(self.compute_epsilon, in_axes = ( 1, None )  ))
        self.moving_average_filter_batch = jit(vmap(self.moving_average_filter, in_axes = ( 1, 1 ), out_axes= (1)  ))



    @partial(jit, static_argnums=(0,))
    def compute_states(self,v_samples,pitch_samples,pitchdot_samples,roll_samples,psi_init, pitch_init, x_init,y_init,z_init):
        
        R = ( self.g /v_samples)*jnp.sin(roll_samples)*jnp.cos(pitch_samples)
        Q = (pitchdot_samples-jnp.sin(roll_samples)*R)/jnp.cos(roll_samples)
        psidot_samples = (jnp.sin(roll_samples)/jnp.cos(pitch_samples)*Q + jnp.cos(roll_samples)/jnp.cos(pitch_samples)*R)
        psi_samples = psi_init+jnp.cumsum(psidot_samples*self.t, axis = 1)
        psi_samples = jnp.hstack(( psi_init*jnp.ones(( self.num_batch, 1 )), psi_samples[:, 0:-1]    ))

        v_x =  v_samples*jnp.cos(psi_samples)*jnp.cos(pitch_samples)
        v_y =  v_samples*jnp.sin(psi_samples)*jnp.cos(pitch_samples)
        v_z =  v_samples*jnp.sin(pitch_samples)

        x = x_init+jnp.cumsum(v_x*self.t, axis = 1)
        y = y_init+jnp.cumsum(v_y*self.t, axis = 1)
        z = z_init+jnp.cumsum(v_z*self.t, axis = 1)

        x = jnp.hstack(( x_init*jnp.ones(( self.num_batch, 1  )), x[:, 0:-1]     ))
        y = jnp.hstack(( y_init*jnp.ones(( self.num_batch, 1  )), y[:, 0:-1]     ))
        z = jnp.hstack(( z_init*jnp.ones(( self.num_batch, 1  )), z[:, 0:-1]     ))

        return x, y, z, psi_samples
    

    @partial(jit, static_argnums=(0,))
    def compute_rollouts_mppi(self, v_samples,pitch_samples,pitchdot_samples,roll_samples,psi_init, pitch_init, x_init,y_init,z_init):

        R = ( self.g /v_samples)*jnp.sin(roll_samples)*jnp.cos(pitch_samples)
        Q = (pitchdot_samples-jnp.sin(roll_samples)*R)/jnp.cos(roll_samples)
        psidot_samples = (jnp.sin(roll_samples)/jnp.cos(pitch_samples)*Q + jnp.cos(roll_samples)/jnp.cos(pitch_samples)*R)
        psi_samples = psi_init+jnp.cumsum(psidot_samples*self.t)

        v_x =  v_samples*jnp.cos(psi_samples)*jnp.cos(pitch_samples)
        v_y =  v_samples*jnp.sin(psi_samples)*jnp.cos(pitch_samples)
        v_z =  v_samples*jnp.sin(pitch_samples)

        x = x_init+jnp.cumsum(v_x*self.t)
        y = y_init+jnp.cumsum(v_y*self.t)
        z = z_init+jnp.cumsum(v_z*self.t)

        return x, y, z,psi_samples
		

    @partial(jit, static_argnums=(0,))
    def compute_noise_samples(self, key):
        
        key, subkey = random.split(key)
        control_samples = jax.random.multivariate_normal(key, self.mean, self.cov, (self.num_batch,))
        
        epsilon_v = control_samples[:, 0: self.num]
        epsilon_pitchdot = control_samples[:, self.num : 2*self.num]
        epsilon_roll = control_samples[:, 2*self.num : 3*self.num]

        return epsilon_v, epsilon_pitchdot, epsilon_roll, key

    @partial(jit, static_argnums=0)
    def get_height_at(self,x, y):

        i = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.width - 1)
        j = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.height - 1)
    
        return jnp.clip(map_coordinates(self.terrain, [[i], [j]], order=1, mode='nearest')[0],0,None)
    
    
    @partial(jit, static_argnums=(0,))
    def compute_cost_mppi(self,u,
                          x_goal,y_goal,z_goal,
                          x,y,z,controls_stack,
                          v_dot,pitch_dot,roll_dot,
                            v_ddot,pitch_ddot, roll_ddot):

        def cost_lax(carry,idx): 

            cost = carry

            # Goal reaching cost
            cost_goal = ((x[idx]-x_goal)**2+(y[idx]-y_goal)**2) * self.w_1 

            # Terrain altitude at the x[idx] y[idx]
            z_terrain = self.get_height_at(x[idx],y[idx]	)

            # Terrain cost
            f_z = ((z[idx]-(z_terrain+self.d_0)))
            cost_min_alt = jnp.maximum(0,-f_z) * self.w_3_1
            
            # Desired altitude cost
            f_z_max = ((z[idx]-(z_terrain+self.z_des)))
            cost_z = jnp.maximum(0,f_z_max) * self.w_3_2


            # MPPI cost
            #u_mppi = jnp.stack((u[idx],u[idx+self.num],u[idx+self.num*2]))
            mppi = (self.param_gamma * u[idx] @ jnp.linalg.inv(self.sigma) @ controls_stack[idx]) * self.w_2

            # Dot cost
            cost_v_dot = (jnp.maximum(0,(jnp.abs(v_dot[idx]) - self.v_max))) * self.w_4
            cost_pitch_dot = (jnp.maximum(0,(jnp.abs(pitch_dot[idx]) - self.pitch_max))) * self.w_5
            cost_roll_dot = (jnp.maximum(0,(jnp.abs(roll_dot[idx]) - self.roll_max))) * self.w_6

            # DDot cost

            cost_v_ddot = (jnp.maximum(0,(jnp.abs(v_ddot[idx]) - self.vddot_max))) * self.w_7
            cost_pitch_ddot = (jnp.maximum(0,(jnp.abs(pitch_ddot[idx]) - self.pitchddot_max))) * self.w_8
            cost_roll_ddot = (jnp.maximum(0,(jnp.abs(roll_ddot[idx]) - self.rollddot_max))) * self.w_9

            cost = cost_goal + cost_min_alt + cost_z + mppi + cost_v_dot + cost_pitch_dot + cost_roll_dot + cost_v_ddot + cost_pitch_ddot + cost_roll_ddot

            return (cost),(cost)

        carry_init = 0
        carry_final, result = lax.scan(cost_lax, carry_init, jnp.arange(self.num_batch))
        cost = result

        return cost
    
    @partial(jit, static_argnums=(0,))
    def _compute_weights(self, S, rho, eta):

        w = (1.0 / eta) * jnp.exp( (-1.0/self.param_lambda) * (S-rho) )

        return w
    
    @partial(jit, static_argnums=(0,))
    def compute_epsilon(self, epsilon, w): 
        w_epsilon_init = jnp.zeros((3))

        def lax_eps(carry,idx):

            w_epsilon = carry
            w_epsilon = w_epsilon + w[idx] * epsilon[idx]
            return (w_epsilon),(0)

        carry_init = (w_epsilon_init)
        carry_final,result = jax.lax.scan(lax_eps,carry_init,jnp.arange(self.num_batch))
        w_epsilon = carry_final

        return w_epsilon
    
    @partial(jit, static_argnums=(0,))
    def g_(self, v, pitch, roll):

        v = jnp.clip(v, self.v_min, self.v_max)
        pitch = jnp.clip(pitch, self.pitch_min, self.pitch_max)
        roll = jnp.clip(roll, self.roll_min, self.roll_max)

        return v, pitch, roll
    
    @partial(jit, static_argnums=(0,))
    def moving_average_filter(self, xx_mean, xx):

        b = jnp.ones(self.window_size)/self.window_size
        n_conv = int(self.window_size/2)
        
        xx_mean = jnp.convolve(xx, b, mode="same")
        xx_mean = xx_mean.at[0].set(xx_mean[0] * self.window_size/n_conv)

        xx_mean_init = xx_mean
        def lax_maf(carry,idx):
            xx_mean = carry
            xx_mean = xx_mean.at[idx].set(xx_mean[idx] * self.window_size/(idx+n_conv))
            xx_mean = xx_mean.at[-idx].set(xx_mean[-idx] * self.window_size/(idx + n_conv - (self.window_size % 2)) )
            return (xx_mean),(0)

        carry_init = (xx_mean_init)
        carry_final,result = jax.lax.scan(lax_maf,carry_init,jnp.arange(1,n_conv))
        xx_mean = carry_final

        return xx_mean
    
    @partial(jit, static_argnums=(0,))
    def dot_controls(self,v,pitch,roll, v_init,pitch_init,roll_init):

        # Vdot
        v_d = jnp.zeros((self.num_batch,self.num+1))
        v_d = v_d.at[:,0].set(v_init)
        v_d = v_d.at[:,1:].set(v)
        v_dot = jnp.diff(v_d,axis=1)/self.t
        
        # Vddot
        v_dd = jnp.zeros((self.num_batch,self.num+1))
        v_dd = v_d.at[:,0].set(self.vdot_init)
        v_dd = v_d.at[:,1:].set(v_dot)
        v_ddot = jnp.diff(v_dd,axis=1)/self.t

        # Pitchdot
        pitch_d = jnp.zeros((self.num_batch,self.num+1))
        pitch_d = pitch_d.at[:,0].set(pitch_init)
        pitch_d = pitch_d.at[:,1:].set(pitch)
        pitch_dot = jnp.diff(pitch_d,axis=1)/self.t

        # Pitchddot
        pitch_dd = jnp.zeros((self.num_batch,self.num+1))
        pitch_dd = pitch_dd.at[:,0].set(self.pitchdot_init)
        pitch_dd = pitch_dd.at[:,1:].set(pitch_dot)
        pitch_ddot = jnp.diff(pitch_d,axis=1)/self.t

        # Rolldot
        roll_d = jnp.zeros((self.num_batch,self.num+1))
        roll_d = roll_d.at[:,0].set(roll_init)
        roll_d = roll_d.at[:,1:].set(roll)
        roll_dot = jnp.diff(roll_d,axis=1)/self.t

        # Rollddot
        roll_dd = jnp.zeros((self.num_batch,self.num+1))
        roll_dd = roll_dd.at[:,0].set(self.rolldot_init)
        roll_dd = roll_dd.at[:,1:].set(roll_dot)
        roll_ddot = jnp.diff(roll_dd,axis=1)/self.t

        return v_dot, v_ddot, pitch_dot, pitch_ddot, roll_dot, roll_ddot

    
    @partial(jit, static_argnums=(0,))
    def mppi_base_main(self,u_prev, key,
                   x_init, y_init, z_init, psi_init,
                   v_init, pitch_init, roll_init,
                   x_goal,y_goal,z_goal,
                   x_global,y_global,z_global
                   ):
        
        # Defining control sequence

        u = u_prev
        v_prev = u[0:self.num]
        pitch_prev = u[self.num:self.num*2]
        roll_prev = u[self.num*2:self.num*3]
        
        # Sampling noise

        epsilon_v, epsilon_pitch, epsilon_roll, key = self.compute_noise_samples(key)

        # Calculating control sequence - Raw samples

        uu = jnp.tile(u,(self.num_batch,1))
        v_raw = uu[:,0:self.num] + epsilon_v
        pitch_raw = uu[:,self.num:self.num*2] + epsilon_pitch
        roll_raw = uu[:,self.num*2:self.num*3] + epsilon_roll
        uu_mppi_cost = jnp.stack((uu[:,0:self.num], uu[:,self.num:self.num*2] , uu[:,self.num*2:self.num*3]),axis=-1)
        # Clipping contols

        v, pitch, roll = self.g_(v_raw, pitch_raw, roll_raw)

        # Compute dot values (velocity, pitch, roll)

        v_dot, v_ddot, pitch_dot, pitch_ddot, roll_dot, roll_ddot = self.dot_controls(v,pitch,roll, v_init,pitch_init,roll_init)

        # Computing states/rollouts

        x, y, z, psi = self.compute_states(v,pitch,pitch_dot,roll,psi_init, pitch_init, x_init,y_init,z_init)
      		
        x_traj_global = x + x_global
        y_traj_global = y + y_global
        z_traj_global = z + z_global
        # Compute cost

        controls_stack = jnp.stack((v, pitch, roll),axis=-1) #reshaping for mppi cost ()


        S_mat = self.compute_cost_mppi_batch(uu_mppi_cost,                             # None
                                             x_goal,y_goal,z_goal,          # None, None, None
                                             x_traj_global,y_traj_global,z_traj_global,controls_stack,          # 4   
                                             v_dot,pitch_dot,roll_dot,      # 3
                                             v_ddot,pitch_ddot, roll_ddot)  # 3
    
        S = jnp.sum(S_mat,axis = 0)

        # Compute rho

        rho = S.min()

        # Calculate eta

        eta = jnp.sum(jnp.exp( (-1.0/self.param_lambda) * (S-rho) ))

        # Compute weights

        w = self.compute_weights_batch(S,rho,eta)

        # Compute weighted epsilom

        epsilon_stack = jnp.stack((epsilon_v, epsilon_pitch, epsilon_roll),axis=-1) #reshaping for mppi 
        w_epsilon = self.compute_epsilon_batch(epsilon_stack,w)

        # Smoothening of epsilon

        xx_mean = jnp.zeros(w_epsilon.shape)
        w_epsilon = self.moving_average_filter_batch(xx_mean,w_epsilon)
        
        # New control sequence

        v_new = u[0:self.num]+w_epsilon[:,0]
        pitch_new = u[self.num:self.num*2]+w_epsilon[:,1]
        roll_new = u[self.num*2:self.num*3]+w_epsilon[:,2]

        # Clipping new control sequence

        v_new, pitch_new, roll_new = self.g_(v_new, pitch_new, roll_new) 

        # Calculating new dot controls
        v_dot_new, v_ddot_new, pitch_dot_new, pitch_ddot_new, roll_dot_new, roll_ddot_new = self.dot_controls(v_new,pitch_new,roll_new, v_init,
                                                                                      pitch_init,roll_init)
        
        # Calculating optimal trajectory

        x_opt, y_opt, z_opt, psi_opt= self.compute_rollouts_mppi(v_new,pitch_new,pitch_dot_new[0],roll_new,psi_init, pitch_init, x_init,y_init,z_init)

        # Updating states for next iteration

        init_states_upd = jnp.array([x_opt[0],y_opt[0],z_opt[0],psi_opt[0]])

        # Updating new init controls

        init_controls_upd = jnp.array([v_new[1],pitch_new[1],roll_new[1]])

        # Updating contol sequence
        
        v_prev = jnp.zeros((self.num))
        v_prev = v_prev.at[:-1].set(v_new[1:])
        v_prev = v_prev.at[-1]. set(v_new[-1])

        pitch_prev = jnp.zeros((self.num))
        pitch_prev = pitch_prev.at[:-1].set(pitch_new[1:])
        pitch_prev = pitch_prev.at[-1].set(pitch_new[-1])

        roll_prev = jnp.zeros((self.num))
        roll_prev = roll_prev.at[:-1].set(roll_new[1:])
        roll_prev = roll_prev.at[-1].set(roll_new[-1])

        u_prev = jnp.hstack((v_prev,pitch_prev,roll_prev))

        ### returning dot values 

        dot_new = jnp.array([v_dot_new[0][2],pitch_dot_new[0][2],roll_dot_new[0][2]])

        ### returning ddot values

        ddot_new = jnp.array([v_ddot_new[0][3],pitch_ddot_new[0][3],roll_ddot_new[0][3]])

        return key, init_states_upd, init_controls_upd, dot_new, ddot_new, u_prev






        











    




        


