
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt 

import bernstein_coeff_order10_arbitinterval
from functools import partial
from jax import jit, random,vmap
import jax
import jax.lax as lax
from jax.scipy.ndimage import map_coordinates

class learned_pi_mppi():

    def __init__(self,v_max, v_min, vdot_max, vdot_min, vddot_max,vddot_min,
                    pitch_max,pitch_min, pitchdot_max, pitchdot_min, pitchddot_max, pitchddot_min,
                        roll_max, roll_min, rolldot_max, rolldot_min, rollddot_max, rollddot_min,
                        scale, octaves, persistence, lacunarity, width, height, altitude_scale, altitude_offset,
                        terrain,x_range,y_range,num_batch,proj_iterations):


        # Weights
        self.w_1 = 2.35				#goal reaching cost 0.01
        self.w_2 = .0000001				#mppi 0.0001
        self.w_3 = 150000			#min altitude 1 
        self.w_4 = 150000	

        ############
        self.d_0 = 5
        self.z_des = 15
        
        self.v_min = v_min
        self.v_max = v_max

        self.v_dot_max = vdot_max
        self.v_dot_min = vdot_min

        self.v_ddot_max = vddot_max
        self.v_ddot_min = vddot_min

        self.pitch_max = pitch_max
        self.pitch_min = pitch_min

        self.pitch_dot_max = pitchdot_max
        self.pitch_dot_min = pitchdot_min

        self.pitch_ddot_max = pitchddot_max
        self.pitch_ddot_min = pitchddot_min

        self.roll_max = roll_max
        self.roll_min = roll_min

        self.roll_dot_max = rolldot_max
        self.roll_dot_min = rolldot_min

        self.roll_ddot_max = rollddot_max
        self.roll_ddot_min = rollddot_min


        self.t_fin = 20
        self.num = 100
        self.t = self.t_fin/self.num
        self.num_batch = num_batch
        # self.ellite_num = 200
        # self.ellite_num_projection = 150


        ############################ waypoint parameterization for control inputs
        
        tot_time = np.linspace(0, self.t_fin, self.num)
        self.tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

        self.nvar = jnp.shape(self.P_jax)[1]


        # self.P_jax = jnp.identity(self.num)
        # self.Pdot_jax = jnp.diff(self.P_jax, axis = 0)/self.t 
        # self.Pddot_jax = jnp.diff(self.Pdot_jax, axis = 0)/self.t

        # self.nvar = jnp.shape(self.P_jax)[0]
        
        self.num_dot = self.num
        self.num_ddot = self.num_dot

        ###############################3
        self.A_projection = jnp.identity(self.nvar)

        
        self.A = jnp.vstack(( self.P_jax, -self.P_jax  ))
        self.A_dot = jnp.vstack(( self.Pdot_jax, -self.Pdot_jax  ))
        self.A_ddot = jnp.vstack(( self.Pddot_jax, -self.Pddot_jax  ))
        self.A_control = jnp.vstack(( self.A, self.A_dot, self.A_ddot ))
        
        self.A_eq_control = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0] ))

        self.rho_projection = 1.0
        self.rho_ineq = 1.0
        
        self.beta = 5

        ####################################
        
        ###################### parameters

        self.rho_ineq = 1.0
        
        #################################################
        self.maxiter = 1
        self.maxiter_cem = 1
        self.maxiter_projection = proj_iterations
        # Normalization
        self.e = 10**(-3)
        
        self.alpha_mean = 0.2
        self.alpha_cov = 0.2
        
        self.lamda = 0.9
        self.g = 9.81
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

        self.compute_cross_track_batch = jit(jax.vmap(self.compute_cross_track, in_axes = (0, 0, 0, None, None, None) )   )

        self.compute_cost_mppi_batch = jit(vmap(self.compute_cost_mppi,in_axes = (None,None,None,1,1,1,1)))
        self.compute_cost_batch = jit(vmap(self.compute_cost,in_axes=((None,None,None,0,0,0,0,None,None))))

        self.compute_weights_batch = jit(vmap(self._compute_weights, in_axes = ( 0, None, None )  ))

        self.compute_epsilon_batch = jit(vmap(self.compute_epsilon, in_axes = ( 1, None )  ))
        self.compute_w_epsilon_batch = jit(vmap(self.compute_w_epsilon,in_axes = (0,0)))


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

        self.terrain = terrain
        self.x_range = x_range
        self.y_range = y_range
        
        ######################################################################################## matrices for computing initial guess based on neural parameters


    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init):

        v_init_vec = v_init*jnp.ones((self.num_batch, 1))
        v_dot_init_vec = v_dot_init*jnp.ones((self.num_batch, 1)) 

        pitch_init_vec = pitch_init*jnp.ones((self.num_batch, 1))
        pitch_dot_init_vec = pitch_dot_init*jnp.ones((self.num_batch, 1))

        roll_init_vec = roll_init*jnp.ones((self.num_batch, 1))
        roll_dot_init_vec = roll_dot_init*jnp.ones((self.num_batch, 1))

        b_eq_v = jnp.hstack(( v_init_vec, v_dot_init_vec))
        b_eq_pitch = jnp.hstack(( pitch_init_vec, pitch_dot_init_vec ))
        b_eq_roll = jnp.hstack(( roll_init_vec, roll_dot_init_vec ))
            
        return b_eq_v, b_eq_pitch, b_eq_roll

    @partial(jit, static_argnums=(0,))
    def compute_feasible_control(self, control_samples, lamda_control, b_eq_control, s_control, control_max, control_min, control_dot_max, control_dot_min, control_ddot_max, control_ddot_min):
        
        b_control = jnp.hstack(( control_max*jnp.ones(( self.num_batch, self.num  )), -control_min*jnp.ones(( self.num_batch, self.num  ))     ))
        b_control_dot = jnp.hstack(( control_dot_max*jnp.ones(( self.num_batch, self.num_dot  )), -control_dot_min*jnp.ones(( self.num_batch, self.num_dot  ))     ))
        b_control_ddot = jnp.hstack(( control_ddot_max*jnp.ones(( self.num_batch, self.num_ddot  )), -control_ddot_min*jnp.ones(( self.num_batch, self.num_ddot  ))     ))		
        b_control_comb = jnp.hstack(( b_control, b_control_dot, b_control_ddot  ))

        b_control_aug = b_control_comb-s_control
        
        cost = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_control.T, self.A_control)

        cost_mat = jnp.vstack((  jnp.hstack(( cost, self.A_eq_control.T )), jnp.hstack(( self.A_eq_control, jnp.zeros(( jnp.shape(self.A_eq_control)[0], jnp.shape(self.A_eq_control)[0] )) )) ))
        lincost = -lamda_control-self.rho_projection*jnp.dot(self.A_projection.T, control_samples.T).T-self.rho_ineq*jnp.dot(self.A_control.T, b_control_aug.T).T

        sol = jnp.linalg.solve(cost_mat, jnp.hstack(( -lincost, b_eq_control )).T).T
        
        control_projected = sol[:, 0: self.nvar]

        s_control = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot )), -jnp.dot(self.A_control, control_projected.T).T+b_control_comb  )

        res_control_vec = jnp.dot(self.A_control, control_projected.T).T-b_control_comb+s_control

        res_control = jnp.linalg.norm(jnp.dot(self.A_control, control_projected.T).T-b_control_comb+s_control, axis = 1)

        lamda_control = lamda_control-self.rho_ineq*jnp.dot(self.A_control.T, res_control_vec.T).T

        return control_projected, s_control, res_control, lamda_control

    # @partial(jit, static_argnums=(0,))
    # def compute_projection(self, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll, c_v_samples, c_pitch_samples, c_roll_samples, b_eq_v, b_eq_pitch, b_eq_roll):
            
    #     c_v_samples_init = c_v_samples 
    #     c_pitch_samples_init = c_pitch_samples 
    #     c_roll_samples_init = c_roll_samples 
    #     lamda_v_init = lamda_v 
    #     lamda_pitch_init = lamda_pitch 
    #     lamda_roll_init = lamda_roll 
        
    #     s_v_init = s_v 
    #     s_pitch_init = s_pitch 
    #     s_roll_init = s_roll		


    #     def lax_custom_projection(carry, idx):
        
    #         c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry
        
    #         c_v_samples, s_v, res_v, lamda_v = self.compute_feasible_control(c_v_samples, lamda_v, b_eq_v, s_v, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)

    #         c_pitch_samples, s_pitch, res_pitch, lamda_pitch = self.compute_feasible_control(c_pitch_samples, lamda_pitch, b_eq_pitch, s_pitch, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)

    #         c_roll_samples, s_roll, res_roll, lamda_roll = self.compute_feasible_control(c_roll_samples, lamda_roll, b_eq_roll, s_roll, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min)

    #         return (c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll), (res_v, res_pitch, res_roll)		

    #     carry_init = (c_v_samples_init, c_pitch_samples_init, c_roll_samples_init, lamda_v_init, lamda_pitch_init, lamda_roll_init, s_v_init, s_pitch_init, s_roll_init )
    #     carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter_projection))

    #     c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry_final
        
    #     res_v, res_pitch, res_roll = res_tot

    #     return c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll


    @partial(jit, static_argnums=(0,))
    def compute_projection(self, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input, b_eq_v, b_eq_pitch, b_eq_roll):
            
        c_v_samples_init = c_v_samples_input 
        c_pitch_samples_init = c_pitch_samples_input 
        c_roll_samples_init = c_roll_samples_input 
        lamda_v_init = lamda_v 
        lamda_pitch_init = lamda_pitch 
        lamda_roll_init = lamda_roll 
        
        s_v_init = s_v 
        s_pitch_init = s_pitch 
        s_roll_init = s_roll		


        def lax_custom_projection(carry, idx):
        
            c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry
        
            c_v_samples, s_v, res_v, lamda_v = self.compute_feasible_control(c_v_samples_input, lamda_v, b_eq_v, s_v, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)

            c_pitch_samples, s_pitch, res_pitch, lamda_pitch = self.compute_feasible_control(c_pitch_samples_input, lamda_pitch, b_eq_pitch, s_pitch, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)

            c_roll_samples, s_roll, res_roll, lamda_roll = self.compute_feasible_control(c_roll_samples_input, lamda_roll, b_eq_roll, s_roll, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min)

            return (c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll), (res_v, res_pitch, res_roll)		

        carry_init = (c_v_samples_init, c_pitch_samples_init, c_roll_samples_init, lamda_v_init, lamda_pitch_init, lamda_roll_init, s_v_init, s_pitch_init, s_roll_init )
        carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter_projection))

        c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry_final
        
        res_v, res_pitch, res_roll = res_tot

        return c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll

    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec_single(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init):

        v_init_vec = v_init*jnp.ones((1, 1))
        v_dot_init_vec = v_dot_init*jnp.ones((1, 1)) 

        pitch_init_vec = pitch_init*jnp.ones((1, 1))
        pitch_dot_init_vec = pitch_dot_init*jnp.ones((1, 1))

        roll_init_vec = roll_init*jnp.ones((1, 1))
        roll_dot_init_vec = roll_dot_init*jnp.ones((1, 1))

        b_eq_v = jnp.hstack(( v_init_vec, v_dot_init_vec))
        b_eq_pitch = jnp.hstack(( pitch_init_vec, pitch_dot_init_vec ))
        b_eq_roll = jnp.hstack(( roll_init_vec, roll_dot_init_vec ))
            
        return b_eq_v, b_eq_pitch, b_eq_roll

    @partial(jit, static_argnums=(0,))
    def compute_projection_single(self, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input, b_eq_v, b_eq_pitch, b_eq_roll):
            
        c_v_samples_init = c_v_samples_input 
        c_pitch_samples_init = c_pitch_samples_input 
        c_roll_samples_init = c_roll_samples_input 
        lamda_v_init = lamda_v 
        lamda_pitch_init = lamda_pitch 
        lamda_roll_init = lamda_roll 
        
        s_v_init = s_v 
        s_pitch_init = s_pitch 
        s_roll_init = s_roll		


        def lax_custom_projection(carry, idx):
        
            c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry
            
            c_v_samples, s_v, res_v, lamda_v = self.compute_feasible_control_single(c_v_samples_input, lamda_v, b_eq_v, s_v, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)
            
            c_pitch_samples, s_pitch, res_pitch, lamda_pitch = self.compute_feasible_control_single(c_pitch_samples_input, lamda_pitch, b_eq_pitch, s_pitch, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)
            
            c_roll_samples, s_roll, res_roll, lamda_roll = self.compute_feasible_control_single(c_roll_samples_input, lamda_roll, b_eq_roll, s_roll, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min)

            return (c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll), (res_v, res_pitch, res_roll)		

        carry_init = (c_v_samples_init, c_pitch_samples_init, c_roll_samples_init, lamda_v_init, lamda_pitch_init, lamda_roll_init, s_v_init, s_pitch_init, s_roll_init )
        carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter_projection))

        c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry_final
        
        res_v, res_pitch, res_roll = res_tot

        return c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll
    # @partial(jit, static_argnums=(0,))
    # def compute_projection_single(self, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll, c_v_samples, c_pitch_samples, c_roll_samples, b_eq_v, b_eq_pitch, b_eq_roll):
            
    #     c_v_samples_init = c_v_samples 
    #     c_pitch_samples_init = c_pitch_samples 
    #     c_roll_samples_init = c_roll_samples 
    #     lamda_v_init = lamda_v 
    #     lamda_pitch_init = lamda_pitch 
    #     lamda_roll_init = lamda_roll 
        
    #     s_v_init = s_v 
    #     s_pitch_init = s_pitch 
    #     s_roll_init = s_roll		


    #     def lax_custom_projection(carry, idx):
        
    #         c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry
            
    #         c_v_samples, s_v, res_v, lamda_v = self.compute_feasible_control_single(c_v_samples, lamda_v, b_eq_v, s_v, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)
            
    #         c_pitch_samples, s_pitch, res_pitch, lamda_pitch = self.compute_feasible_control_single(c_pitch_samples, lamda_pitch, b_eq_pitch, s_pitch, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)
            
    #         c_roll_samples, s_roll, res_roll, lamda_roll = self.compute_feasible_control_single(c_roll_samples, lamda_roll, b_eq_roll, s_roll, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min)

    #         return (c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll), (res_v, res_pitch, res_roll)		

    #     carry_init = (c_v_samples_init, c_pitch_samples_init, c_roll_samples_init, lamda_v_init, lamda_pitch_init, lamda_roll_init, s_v_init, s_pitch_init, s_roll_init )
    #     carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter_projection))

    #     c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry_final
        
    #     res_v, res_pitch, res_roll = res_tot

    #     return c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll

    @partial(jit, static_argnums=(0,))
    def compute_feasible_control_single(self, control_samples, lamda_control, b_eq_control, s_control, control_max, control_min, control_dot_max, control_dot_min, control_ddot_max, control_ddot_min):
        
        b_control = jnp.hstack(( control_max*jnp.ones(( 1, self.num  )), -control_min*jnp.ones(( 1, self.num  ))     ))
        b_control_dot = jnp.hstack(( control_dot_max*jnp.ones(( 1, self.num_dot  )), -control_dot_min*jnp.ones(( 1, self.num_dot  ))     ))
        b_control_ddot = jnp.hstack(( control_ddot_max*jnp.ones(( 1, self.num_ddot  )), -control_ddot_min*jnp.ones(( 1, self.num_ddot  ))     ))		
        b_control_comb = jnp.hstack(( b_control, b_control_dot, b_control_ddot  ))

        b_control_aug = b_control_comb-s_control
        
        cost = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_control.T, self.A_control)

        cost_mat = jnp.vstack((  jnp.hstack(( cost, self.A_eq_control.T )), jnp.hstack(( self.A_eq_control, jnp.zeros(( jnp.shape(self.A_eq_control)[0], jnp.shape(self.A_eq_control)[0] )) )) ))
        lincost = -lamda_control-self.rho_projection*jnp.dot(self.A_projection.T, control_samples.T).T-self.rho_ineq*jnp.dot(self.A_control.T, b_control_aug.T).T

        sol = jnp.linalg.solve(cost_mat, jnp.hstack(( -lincost, b_eq_control )).T).T
        
        control_projected = sol[:,0: self.nvar]

        s_control = jnp.maximum( jnp.zeros(( 1, 2*self.num+2*self.num_dot+2*self.num_ddot )), -jnp.dot(self.A_control, control_projected.T).T+b_control_comb  )

        res_control_vec = jnp.dot(self.A_control, control_projected.T).T-b_control_comb+s_control

        res_control = jnp.linalg.norm(jnp.dot(self.A_control, control_projected.T).T-b_control_comb+s_control, axis = 1)

        lamda_control = lamda_control-self.rho_ineq*jnp.dot(self.A_control.T, res_control_vec.T).T

        return control_projected.squeeze(axis=0), s_control, res_control.squeeze(axis=0), lamda_control

    @partial(jit, static_argnums=(0,))
    def compute_control_samples(self, key, mean_control, cov_control):
        
        key, subkey = random.split(key)
        control_samples = jax.random.multivariate_normal(key, mean_control, cov_control, (self.num_batch, ))

        c_v_samples = control_samples[:, 0: self.nvar]
        c_pitch_samples = control_samples[:, self.nvar : 2*self.nvar]
        c_roll_samples = control_samples[:, 2*self.nvar : 3*self.nvar]

        return c_v_samples, c_pitch_samples, c_roll_samples, key



    @partial(jit, static_argnums=(0,))
    def compute_rollouts(self,  x_init, y_init, z_init, psi_init, v_samples, roll_samples, pitch_samples, pitchdot_samples):
        
        R = ( self.g /v_samples)*jnp.sin(roll_samples)*jnp.cos(pitch_samples)
        Q = (pitchdot_samples+jnp.sin(roll_samples)*R)/jnp.cos(roll_samples)
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

        return x, y, z, psi_samples, psidot_samples

    @partial(jit, static_argnums=(0,))
    def compute_rollouts_mppi(self,  x_init, y_init, z_init, psi_init, v_samples, roll_samples, pitchdot_samples,pitch_samples):
        
        # pitch_samples = pitch_init+jnp.cumsum(pitchdot_samples*self.t)
        R = ( self.g /v_samples)*jnp.sin(roll_samples)*jnp.cos(pitch_samples)
        Q = (pitchdot_samples+jnp.sin(roll_samples)*R)/jnp.cos(roll_samples)
        psidot_samples = (jnp.sin(roll_samples)/jnp.cos(pitch_samples)*Q + jnp.cos(roll_samples)/jnp.cos(pitch_samples)*R)
        psi_samples = psi_init+jnp.cumsum(psidot_samples*self.t)

        v_x =  v_samples*jnp.cos(psi_samples)*jnp.cos(pitch_samples)
        v_y =  v_samples*jnp.sin(psi_samples)*jnp.cos(pitch_samples)
        v_z =  v_samples*jnp.sin(pitch_samples)

        x = x_init+jnp.cumsum(v_x*self.t)
        y = y_init+jnp.cumsum(v_y*self.t)
        z = z_init+jnp.cumsum(v_z*self.t)



        return x, y, z,psi_samples,pitch_samples

    # @partial(jit, static_argnums=0)
    # def get_height_at(self,x, y):
    #     # Convert input data to jax.numpy arrays
    #     # terrain = jnp.array(self.terrain)
        
    #     # Create grid coordinates
    #     x_grid = jnp.arange(self.terrain.shape[0])
    #     y_grid = jnp.arange(self.terrain.shape[1])
        
    #     # Normalize coordinates to be between 0 and the size of the dimension - 1
    #     x_coords = (x / (x_grid[-1] - x_grid[0])) * (self.terrain.shape[0] - 1)
    #     y_coords = (y / (y_grid[-1] - y_grid[0])) * (self.terrain.shape[1] - 1)
        
    #     # Create a grid of coordinates
    #     coords = jnp.array([x_coords, y_coords])
        
    #     # Perform interpolation using map_coordinates
    #     return jnp.clip(map_coordinates(self.terrain, coords, order=1, mode='nearest'),0,None)


    @partial(jit, static_argnums=0)
    def get_height_at(self,x, y):

        # Map (x, y) to (i, j) in grid coordinates
        i = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.width - 1)
        j = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.height - 1)
        

        return jnp.clip(map_coordinates(self.terrain, [[i], [j]], order=1, mode='nearest')[0],0,None)



    # @partial(jit, static_argnums=(0,))
    # def compute_cost_mppi(self,controls_stack,x,y,z,x_fin,y_fin,z_fin,):

    #     u_mean = jnp.mean(controls_stack,axis = 0)
    #     sigma = jnp.cov((controls_stack - u_mean).T)

    #     def cost_lax(carry,idx):
    #         cost = carry
    #         cost_goal = ((x[idx]-x_fin)**2+(y[idx]-y_fin)**2)

    #         # z_terrain = 100*(0.1*jnp.cos(0.01*x[idx]) + 0.1*jnp.sin(0.01*y[idx]))

    #         # z_terrain = 40*(0.7*jnp.cos(0.016*x[idx]) + 0.4*jnp.sin(0.016*y[idx])) #(75*(jnp.cos(0.02*x[idx]) + jnp.sin(0.02*y[idx])))
    #         # z_terrain = jnp.clip(z_terrain,0,40)

    #         z_terrain = self.get_height_at(x[idx],y[idx]	)

    #         f_z = ((z[idx]-(z_terrain+self.d_0)))

    #         # cost_alt = jnp.log(1+jnp.exp(-self.beta*f_z))/self.beta
    #         cost_alt = jnp.maximum(0,-f_z)
            
    #         # cost_z = ((z_terrain+self.z_des) - z[idx])**2

    #         f_z_max = ((z[idx]-(z_terrain+self.z_des)))
    #         cost_z = jnp.maximum(0,f_z_max)

    #         cost_mppi = self.param_gamma * u_mean.T @ jnp.linalg.inv(sigma) @ controls_stack[idx]

    #         # cost_array = jnp.array([cost_goal,cost_alt,cost_z,cost_mppi])

    #         # cost_min = jnp.min(cost_array)
    #         # cost_max = jnp.max(cost_array)
            
            
    #         # c_1 = ((cost_goal - cost_min)/(cost_max - cost_min + 0.0001)) * self.w_1 
    #         # c_3 = ((cost_alt - cost_min)/(cost_max - cost_min+ 0.0001)) * self.w_3
    #         # c_4 = ((cost_z - cost_min)/(cost_max - cost_min+ 0.0001)) * self.w_4
    #         # c_2 = ((cost_mppi- cost_min)/(cost_max - cost_min+ 0.0001)) * self.w_2


    #         cost = cost_goal * self.w_1 + cost_mppi * self.w_2 + cost_alt * self.w_3 + cost_z * self.w_4

    #         return(cost),(cost)
        
    #     carry_init = 0
    #     carry_final, result = lax.scan(cost_lax, carry_init, jnp.arange(self.num_batch))
    #     cost = result

    #     return cost


    @partial(jit, static_argnums=(0,))
    def compute_cost(self,x_goal,y_goal,z_goal,
                    x,y,z,controls_stack,u_mean,sigma
                          ):

        # Goal reaching cost
        cost_goal = ((x-x_goal)**2+(y-y_goal)**2) * self.w_1 

        # Terrain altitude at the x[idx] y[idx]
        z_terrain = self.get_height_at(x,y	)

        # Terrain cost
        f_z = ((z-(z_terrain+self.d_0)))
        cost_min_alt = jnp.maximum(0,-f_z) * self.w_3
        
        # Desired altitude cost
        f_z_max = ((z-(z_terrain+self.z_des)))
        cost_z = jnp.maximum(0,f_z_max) * self.w_4

        mppi = (self.param_gamma * u_mean @ jnp.linalg.inv(sigma) @ controls_stack) * self.w_2

        return cost_goal, cost_min_alt, cost_z, mppi
    
    @partial(jit, static_argnums=(0,))
    def compute_cost_mppi(self,
						x_goal,y_goal,z_goal,
						x,y,z,controls_stack,):
        u_mean = jnp.mean(controls_stack,axis = 0)

        sigma = jnp.cov((controls_stack - u_mean).T)
        cost_goal, cost_min_alt, cost_z, mppi = self.compute_cost_batch(x_goal,y_goal,z_goal,
																x,y,z,controls_stack,u_mean,sigma)
        cost = cost_goal + cost_min_alt + cost_z + mppi

        return cost
    

    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):
        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)
        prods = d * jnp.outer(term_1,term_2)
        # prods = d*jnp.outer(diffs,diffs)
        return prods	

    @partial(jit, static_argnums=(0,))
    def comp_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, samples_ellite):
        
        w = cost_ellite
        w_min = jnp.min(cost_ellite)
        w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
        sum_w = jnp.sum(w, axis = 0)
        mean_control = (1-self.alpha_mean)*mean_control_prev + self.alpha_mean*(jnp.sum( (samples_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
        diffs = (samples_ellite - mean_control)
        prod_result = self.vec_product(diffs, w)
        cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.0001*jnp.identity(self.nvar*3)

        
        return mean_control, cov_control

    @partial(jit, static_argnums=(0,))
    def compute_cross_track(self, x, y, z, x_ref, y_ref, z_ref):
        
        dist = jnp.sqrt( (x-x_ref)**2+(y-y_ref)**2 +(z-z_ref)**2  )
        idx = jnp.argmin(dist)
        x_closest = x_ref[idx]
        y_closest = y_ref[idx]
        z_closest = z_ref[idx]
        cross_track = (x-x_closest)**2+(y-y_closest)**2 +(z-z_closest)**2
        return (cross_track)
        

    @partial(jit, static_argnums=(0,))
    def _compute_weights(self, S, rho, eta):

        # calculate weight
        w = (1.0 / eta) * jnp.exp( (-1.0/self.param_lambda) * (S-rho) )

        return w

    # @partial(jit, static_argnums=(0,))
    # def compute_epsilon(self, epsilon, w): 
    #     w_epsilon_init = jnp.zeros((3))

    #     def lax_eps(carry,idx):

    #         w_epsilon = carry
    #         w_epsilon = w_epsilon + w[idx] * epsilon[idx]
    #         return (w_epsilon),(0)

    #     carry_init = (w_epsilon_init)
    #     carry_final,result = jax.lax.scan(lax_eps,carry_init,jnp.arange(self.num_batch))
    #     w_epsilon = carry_final

    #     return w_epsilon

    @partial(jit, static_argnums=(0,))
    def compute_epsilon(self, epsilon, w): 
        we = self.compute_w_epsilon_batch(epsilon,w)
        w_epsilon = jnp.sum(we,axis = 0)

        return w_epsilon
    
    @partial(jit, static_argnums=(0,))
    def compute_w_epsilon(self,epsilon,w):
        return (w*epsilon)

    @partial(jit, static_argnums=(0,))
    def state_update(self,vdot,v_init,rolldot,roll_init,pitchdot,pitch_init, x_init,y_init,z_init,psi_init):
        vdot_new = jnp.mean(vdot[0:3])
        v = v_init + vdot_new*self.t/2


        rolldot_new = jnp.mean(rolldot[0:3])
        roll = roll_init + rolldot_new*self.t/2

        pitchdot_new = jnp.mean(pitchdot[0:3])
        pitch = pitch_init + pitchdot_new*self.t/2

        R = ( self.g /v)*jnp.sin(roll)*jnp.cos(pitch)
        Q = (pitchdot_new+jnp.sin(roll)*R)/jnp.cos(roll)
        psidot = (jnp.sin(roll)/jnp.cos(pitch)*Q + jnp.cos(roll)/jnp.cos(pitch)*R)
        psi = psi_init+(psidot*self.t/2)

        v_x =  v*jnp.cos(psi)*jnp.cos(pitch)
        v_y =  v*jnp.sin(psi)*jnp.cos(pitch)
        v_z = -v*jnp.sin(pitch)

        x = x_init+(v_x*self.t/2)
        y = y_init+(v_y*self.t/2)
        z = z_init+(v_z*self.t/2)

        states_upd = jnp.hstack((v,roll,pitch,x,y,z,psi))

        return states_upd

    @partial(jit, static_argnums=(0,))
    def dot_controls(self,v,pitch,roll, v_init,pitch_init,roll_init,v_dot_init,pitch_dot_init,roll_dot_init):

        # Vdot
        v_d = jnp.zeros(self.num+1)
        v_d = v_d.at[0].set(v_init)
        v_d = v_d.at[1:].set(v)
        v_dot = jnp.diff(v_d)/self.t

        # Vddot
        v_dd = jnp.zeros(self.num+1)
        v_dd = v_d.at[0].set(v_dot_init)
        v_dd = v_d.at[1:].set(v_dot)
        v_ddot = jnp.diff(v_dd)/self.t

        # Pitchdot
        pitch_d = jnp.zeros(self.num+1)
        pitch_d = pitch_d.at[0].set(pitch_init)
        pitch_d = pitch_d.at[1:].set(pitch)
        pitch_dot = jnp.diff(pitch_d)/self.t

        # Pitchddot
        pitch_dd = jnp.zeros(self.num+1)
        pitch_dd = pitch_dd.at[0].set(pitch_dot_init)
        pitch_dd = pitch_dd.at[1:].set(pitch_dot)
        pitch_ddot = jnp.diff(pitch_d)/self.t

        # Rolldot
        roll_d = jnp.zeros(self.num+1)
        roll_d = roll_d.at[0].set(roll_init)
        roll_d = roll_d.at[1:].set(roll)
        roll_dot = jnp.diff(roll_d)/self.t

        # Rollddot
        roll_dd = jnp.zeros(self.num+1)
        roll_dd = roll_dd.at[0].set(roll_dot_init)
        roll_dd = roll_dd.at[1:].set(roll_dot)
        roll_ddot = jnp.diff(roll_dd)/self.t

        return v_dot, v_ddot, pitch_dot, pitch_ddot, roll_dot, roll_ddot

    @partial(jit, static_argnums=(0,))
    def compute_s_init(self, control_samples, control_max, control_min, control_dot_max, control_dot_min, control_ddot_max, control_ddot_min):
        
        b_control = jnp.hstack([ control_max*jnp.ones(( self.num_batch, self.num  )), -control_min*jnp.ones(( self.num_batch, self.num  ))     ])
        b_control_dot = jnp.hstack([ control_dot_max*jnp.ones(( self.num_batch, self.num_dot  )), -control_dot_min*jnp.ones(( self.num_batch, self.num_dot  ))     ])
        b_control_ddot = jnp.hstack([ control_ddot_max*jnp.ones(( self.num_batch, self.num_ddot  )), -control_ddot_min*jnp.ones(( self.num_batch, self.num_ddot  ))     ])		
        b_control_comb = jnp.hstack([ b_control, b_control_dot, b_control_ddot  ])

        # b_control_aug = b_control_comb-s_control

        s_control = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot )), -jnp.matmul(self.A_control, control_samples.T).T+b_control_comb  )

        
        return s_control

    #@partial(jit, static_argnums=(0,))
    def compute_cem(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init, psi_init, x_init, y_init, z_init,
                     x_fin, y_fin, z_fin, mean,key,x_global,y_global,z_global,lamda_v_init,lamda_pitch_init,lamda_roll_init,
                    c_v_samples_nn,c_pitch_samples_nn,c_roll_samples_nn, c_v_samples,c_pitch_samples,c_roll_samples):

        s_v_init = self.compute_s_init(c_v_samples_nn, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)
        s_pitch_init = self.compute_s_init(c_pitch_samples_nn, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)
        s_roll_init = self.compute_s_init(c_roll_samples_nn, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min) 

        # lamda_v_init = jnp.zeros((self.num_batch, self.nvar)) 
        # lamda_pitch_init = jnp.zeros((self.num_batch, self.nvar))
        # lamda_roll_init = jnp.zeros((self.num_batch, self.nvar))

        # s_v_init = jnp.zeros((self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot))
        # s_pitch_init = jnp.zeros((self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot))
        # s_roll_init = jnp.zeros((self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot)) 

        # MPPI
        # c_v_raw_samples = c_v_samples
        # c_pitch_raw_samples = c_pitch_samples 
        # c_roll_raw_samples = c_roll_samples

        # raw_samples = jnp.hstack((c_v_raw_samples,c_pitch_raw_samples,c_roll_raw_samples))

        # print(jnp.shape(c_t_samples)) 
        # kk	

        # v_raw_samples = jnp.dot(self.P_jax, c_v_samples.T).T 
        # pitch_raw_samples = jnp.dot(self.P_jax, c_pitch_samples.T).T
        # roll_raw_samples = jnp.dot(self.P_jax, c_roll_samples.T).T
        # pitchdot_raw_samples = jnp.dot(self.Pdot_jax, c_pitch_samples.T).T

        #x_traj_raw, y_traj_raw, z_traj_raw, psi_samples_raw, psidot_samples_raw = self.compute_rollouts(x_init, y_init, z_init, psi_init, v_raw_samples, roll_raw_samples, pitch_raw_samples, pitchdot_raw_samples)

        #raw_states = jnp.array([x_traj_raw,y_traj_raw,z_traj_raw])
        # plt.figure(1)

        # plt.plot(x_traj_raw.T, y_traj_raw.T)
        # plt.scatter(x_fin,y_fin)
        # plt.axis('equal')
        # plt.show()
        ################ some projection parameters
        b_eq_v, b_eq_pitch, b_eq_roll = self.compute_boundary_vec(v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init)

        c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = self.compute_projection(lamda_v_init, lamda_pitch_init, lamda_roll_init, s_v_init, s_pitch_init, s_roll_init, c_v_samples, c_pitch_samples, c_roll_samples, b_eq_v, b_eq_pitch, b_eq_roll)

        # plt.figure()
        # plt.plot(np.asarray(res_roll))
        # plt.show()

        # plt.figure()
        # plt.plot(np.asarray(res_pitch))
        # plt.show()

        # plt.figure()
        # plt.plot(np.asarray(res_v))
        # plt.show()

        # proj_samples = jnp.hstack((c_v_samples,c_pitch_samples,c_roll_samples))

        # res_v_cost = res_v.T[:, -1]
        # res_pitch_cost = res_pitch.T[:, -1]
        # res_roll_cost = res_roll.T[:, -1]

        v_samples = jnp.dot(self.P_jax, c_v_samples.T).T 
        vdot_samples = jnp.dot(self.Pdot_jax, c_v_samples.T).T
        pitch_samples = jnp.dot(self.P_jax, c_pitch_samples.T).T
        pitchdot_samples = jnp.dot(self.Pdot_jax, c_pitch_samples.T).T
        roll_samples = jnp.dot(self.P_jax, c_roll_samples.T).T		
        rolldot_samples = jnp.dot(self.Pdot_jax, c_roll_samples.T).T	

        ##### trajectory rollouts
        x_traj, y_traj, z_traj, psi_samples, psidot_samples = self.compute_rollouts(x_init, y_init, z_init, psi_init, v_samples, roll_samples, pitch_samples, pitchdot_samples)

        x_traj_global = x_traj + x_global
        y_traj_global = y_traj + y_global
        z_traj_global = z_traj + z_global

        controls_stack = jnp.stack((v_samples,pitch_samples,roll_samples),axis=-1)


        S_mat = self.compute_cost_mppi_batch(x_fin,y_fin,z_fin,
					x_traj_global,y_traj_global,z_traj_global,controls_stack)
        S = jnp.sum(S_mat,axis = 0)


        rho = S.min()

        eta = jnp.sum(jnp.exp( (-1.0/self.param_lambda) * (S-rho) ))

        w = self.compute_weights_batch(S,rho,eta)


        epsilon = controls_stack - jnp.mean(controls_stack,axis=0)

        w_epsilon = self.compute_epsilon_batch(epsilon,w)

        u_new = jnp.mean(controls_stack,axis=0) + w_epsilon

        v_new = u_new[:,0]
        pitch_new = u_new[:,1]
        roll_new = u_new[:,2]

        c_v_mppi = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(11)) @ self.P_jax.T @ v_new
        c_pitch_mppi = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(11)) @ self.P_jax.T @ pitch_new
        c_roll_mppi = jnp.linalg.inv(self.P_jax.T @ self.P_jax+0.001*jnp.identity(11)) @ self.P_jax.T @ roll_new

        pitchdot_new = jnp.dot(self.Pdot_jax, c_pitch_mppi.T).T

        x_traj_mppi, y_traj_mppi, z_traj_mppi,psi_mppi,pitch_mppi =	self.compute_rollouts_mppi( x_init, y_init, z_init, psi_init, v_new, roll_new, pitchdot_new,pitch_new)
        
        # plt.figure()
        # plt.plot(np.asarray(x_traj_global.T),np.asarray(y_traj_global.T),linewidth=0.5)
        # x_traj_mppi_global = x_traj_mppi + x_global
        # y_traj_mppi_global = y_traj_mppi + y_global
        # plt.plot(np.asarray(x_traj_mppi_global),np.asarray(y_traj_mppi_global),linewidth=3)
        # plt.scatter(x_fin, y_fin)
        # plt.show()

        mean = jnp.hstack((c_v_mppi,c_pitch_mppi,c_roll_mppi))

        new_init_states = jnp.array([x_traj_mppi[0],y_traj_mppi[0],z_traj_mppi[0],psi_mppi[0]])

        new_init_controls = jnp.array([v_new[1],pitch_new[1],roll_new[1]])


        # Calculating new dot controls
        v_dot_mppi, v_ddot_mppi, pitch_dot_mppi, pitch_ddot_mppi, roll_dot_mppi, roll_ddot_mppi = self.dot_controls(v_new,pitch_new,roll_new, v_init,
                                                                                        pitch_init,roll_init,v_dot_init,pitch_dot_init,roll_dot_init)

        ### dot
        # pitch_dot_mppi = jnp.dot(self.P,c_pitchdot_mppi.T) #check this 
        # v_dot_mppi = jnp.dot(self.Pdot_jax,c_v_mppi.T)
        # roll_dot_mppi = jnp.dot(self.Pdot_jax,c_roll_mppi.T)

        dot_values = jnp.hstack((v_dot_mppi[2],pitchdot_new[2],roll_dot_mppi[2]))

        ### ddot
        # v_ddot_mppi = jnp.dot(self.Pddot_jax,c_v_mppi.T)
        # roll_ddot_mppi = jnp.dot(self.Pddot_jax,c_roll_mppi.T)
        # pitch_ddot_mppi = jnp.dot(self.Pdot,c_pitchdot_mppi.T)

        ddot_values = jnp.hstack((v_ddot_mppi[3],pitch_ddot_mppi[3],roll_ddot_mppi[3]))

        return mean,key,new_init_states,new_init_controls,dot_values,ddot_values