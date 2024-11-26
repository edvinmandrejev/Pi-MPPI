

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



class MLP(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(MLP, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),


			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out


class mlp_projection_filter(nn.Module):
	
    def __init__(self, P, Pdot, Pddot, mlp, num_batch, inp_mean, inp_std, t_fin):
        super(mlp_projection_filter, self).__init__()
        
        # BayesMLP
        self.mlp = mlp
        
        # Normalizing Constants
        self.inp_mean = inp_mean
        self.inp_std = inp_std


        self.v_min = 12.0
        self.v_max = 20.0

        self.v_dot_max = 3
        self.v_dot_min = -3

        self.v_ddot_max = 3.0
        self.v_ddot_min = -3.0

        self.pitch_max = 0.2
        self.pitch_min = -0.2

        self.pitch_dot_max = 0.25
        self.pitch_dot_min = -0.25

        self.pitch_ddot_max = 0.15
        self.pitch_ddot_min = -0.15

        self.roll_max = 0.25
        self.roll_min = -0.25

        self.roll_dot_max = 0.25
        self.roll_dot_min = -0.25

        self.roll_ddot_max = 0.15
        self.roll_ddot_min = -0.15


        # P Matrices
        self.P = P.to(device)
        self.Pdot = Pdot.to(device)
        self.Pddot = Pddot.to(device)


        self.A_eq_x = torch.vstack([self.P[0], self.Pdot[0]    ]  )
        self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0]   ]  )

                
        # No. of Variables
        self.nvar = P.size(dim = 1)
        self.num = P.size(dim = 0)
        self.num_batch = num_batch

        self.A_projection = torch.eye(self.nvar, device = device)

        self.rho_projection = 1
        self.rho_ineq = 1
        
        
        self.A_projection = torch.eye(self.nvar, device = device)

        self.maxiter =  15

        self.t_fin = t_fin		

        self.tot_time = torch.linspace(0, t_fin, self.num, device=device)

        self.A = torch.vstack([ self.P, -self.P  ])
        self.A_dot = torch.vstack([ self.Pdot, -self.Pdot  ])
        self.A_ddot = torch.vstack([self.Pddot, -self.Pddot  ])
        self.A_control = torch.vstack([self.A, self.A_dot, self.A_ddot ])
        
        self.A_eq_control = torch.vstack([self.P[0], self.Pdot[0] ])

        self.num_dot = self.num
        self.num_ddot = self.num_dot

        self.num_constraint = 2*self.num+2*self.num_dot+2*self.num_ddot

        ########################################
        
        # RCL Loss
        self.rcl_loss = nn.MSELoss()
  
  
    def compute_boundary_vec(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init):

        v_init_vec = v_init.reshape(self.num_batch, 1)
        v_dot_init_vec = v_dot_init.reshape(self.num_batch, 1) 

        pitch_init_vec = pitch_init.reshape(self.num_batch, 1)
        pitch_dot_init_vec = pitch_dot_init.reshape(self.num_batch, 1)

        roll_init_vec = roll_init.reshape(self.num_batch, 1)
        roll_dot_init_vec = roll_dot_init.reshape(self.num_batch, 1)

        b_eq_v = torch.hstack([ v_init_vec, v_dot_init_vec])
        b_eq_pitch = torch.hstack([pitch_init_vec, pitch_dot_init_vec ])
        b_eq_roll = torch.hstack([roll_init_vec, roll_dot_init_vec ])
            
        return b_eq_v, b_eq_pitch, b_eq_roll


    def compute_mat_inv(self):
        
        cost = self.rho_projection*torch.mm(self.A_projection.T, self.A_projection)+self.rho_ineq*torch.mm(self.A_control.T, self.A_control)

        cost_mat = torch.vstack([  torch.hstack([ cost, self.A_eq_control.T ]), torch.hstack(( self.A_eq_control, torch.zeros(( self.A_eq_control.size(dim = 0), self.A_eq_control.size(dim = 0) ), device = device ) ))  ])

        cost_mat_inv = torch.linalg.inv(cost_mat)

        return cost_mat_inv

    def compute_feasible_control(self, cost_mat_inv_control, control_samples, lamda_control, b_eq_control, s_control, control_max, control_min, control_dot_max, control_dot_min, control_ddot_max, control_ddot_min):
        
        b_control = torch.hstack([ control_max*torch.ones(( self.num_batch, self.num  ), device = device), -control_min*torch.ones(( self.num_batch, self.num  ), device = device)     ])
        b_control_dot = torch.hstack([ control_dot_max*torch.ones(( self.num_batch, self.num_dot  ), device = device), -control_dot_min*torch.ones(( self.num_batch, self.num_dot  ), device = device)     ])
        b_control_ddot = torch.hstack([ control_ddot_max*torch.ones(( self.num_batch, self.num_ddot  ), device = device), -control_ddot_min*torch.ones(( self.num_batch, self.num_ddot  ), device = device)     ])		
        b_control_comb = torch.hstack([ b_control, b_control_dot, b_control_ddot  ])

        b_control_aug = b_control_comb-s_control
        
        lincost = -lamda_control-self.rho_projection*torch.mm(self.A_projection.T, control_samples.T).T-self.rho_ineq*torch.mm(self.A_control.T, b_control_aug.T).T

        # print(lincost.size())	


        sol = torch.mm(cost_mat_inv_control, torch.hstack(( -lincost, b_eq_control )).T).T
        
        control_projected = sol[:, 0: self.nvar]

        s_control = torch.maximum( torch.zeros(( self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot ), device = device), -torch.mm(self.A_control, control_projected.T).T+b_control_comb  )

        res_control_vec = torch.mm(self.A_control, control_projected.T).T-b_control_comb+s_control

        res_control = torch.linalg.norm(torch.mm(self.A_control, control_projected.T).T-b_control_comb+s_control, dim = 1)

        lamda_control = lamda_control-self.rho_ineq*torch.mm(self.A_control.T, res_control_vec.T).T

        return control_projected, s_control, res_control, lamda_control


    def compute_s_init(self, control_samples, lamda_control, b_eq_control, control_max, control_min, control_dot_max, control_dot_min, control_ddot_max, control_ddot_min):
        
        b_control = torch.hstack([ control_max*torch.ones(( self.num_batch, self.num  ), device = device), -control_min*torch.ones(( self.num_batch, self.num  ), device = device)     ])
        b_control_dot = torch.hstack([ control_dot_max*torch.ones(( self.num_batch, self.num_dot  ), device = device), -control_dot_min*torch.ones(( self.num_batch, self.num_dot  ), device = device)     ])
        b_control_ddot = torch.hstack([ control_ddot_max*torch.ones(( self.num_batch, self.num_ddot  ), device = device), -control_ddot_min*torch.ones(( self.num_batch, self.num_ddot  ), device = device)     ])		
        b_control_comb = torch.hstack([ b_control, b_control_dot, b_control_ddot  ])

        # b_control_aug = b_control_comb-s_control

        s_control = torch.maximum( torch.zeros(( self.num_batch, 2*self.num+2*self.num_dot+2*self.num_ddot ), device = device), -torch.mm(self.A_control, control_samples.T).T+b_control_comb  )

        
        return s_control



        # torch.linalg.norm(c_v_samples-c_v_samples_prev, dim = 1) + \
        # 					  torch.linalg.norm(c_pitch_samples-c_pitch_samples_prev, dim = 1) +\
        #                       torch.linalg.norm(c_roll_samples-c_roll_samples_prev, dim = 1) +\
                                    
                                    


    def custom_forward(self, lamda_v, lamda_pitch, lamda_roll, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input, b_eq_v, b_eq_pitch, b_eq_roll, c_v_samples, c_pitch_samples, c_roll_samples):
        

        cost_mat_inv_control = 	self.compute_mat_inv()


        s_v = self.compute_s_init(c_v_samples, lamda_v, b_eq_v, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)

        s_pitch = self.compute_s_init(c_pitch_samples, lamda_pitch, b_eq_pitch, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)

        s_roll = self.compute_s_init(c_roll_samples, lamda_roll, b_eq_roll, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min)

        # print(s_v)    
        
        accumulated_res_primal = []

        accumulated_res_fixed_point = []

        for i in range(0, self.maxiter):
        
            c_v_samples_prev = c_v_samples.clone()
            c_pitch_samples_prev = c_pitch_samples.clone()
            c_roll_samples_prev = c_roll_samples.clone()
            lamda_v_prev = lamda_v.clone() 
            lamda_pitch_prev = lamda_pitch.clone()
            lamda_roll_prev = lamda_roll.clone()
            s_pitch_prev = s_pitch.clone() 
            s_roll_prev = s_roll.clone()
            s_v_prev = s_v.clone()

            # print("in ",c_v_samples_input)

            c_v_samples, s_v, res_v, lamda_v =  self.compute_feasible_control(cost_mat_inv_control, c_v_samples_input, lamda_v, b_eq_v, s_v, self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)
            
            # print("out ",c_v_samples)

            c_pitch_samples, s_pitch, res_pitch, lamda_pitch =  self.compute_feasible_control(cost_mat_inv_control, c_pitch_samples_input, lamda_pitch, b_eq_pitch, s_pitch, self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min, self.pitch_ddot_max, self.pitch_ddot_min)
            
            c_roll_samples, s_roll, res_roll, lamda_roll =  self.compute_feasible_control(cost_mat_inv_control, c_roll_samples_input, lamda_roll, b_eq_roll, s_roll, self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min, self.roll_ddot_max, self.roll_ddot_min)
            
            accumulated_res_primal.append(res_v+res_pitch+res_roll)

            fixed_point_res = torch.linalg.norm(lamda_v-lamda_v_prev, dim = 1) +\
                                torch.linalg.norm(lamda_pitch-lamda_pitch_prev, dim = 1) + \
                                torch.linalg.norm(lamda_roll-lamda_roll_prev, dim = 1) + \
                                torch.linalg.norm(s_pitch-s_pitch_prev, dim = 1) + \
                                torch.linalg.norm(s_roll-s_roll_prev, dim = 1)  + \
                                torch.linalg.norm(s_v-s_v_prev, dim = 1)+\
                                torch.linalg.norm(c_v_samples-c_v_samples_prev, dim = 1) + \
                                torch.linalg.norm(c_pitch_samples-c_pitch_samples_prev, dim = 1) +\
                                torch.linalg.norm(c_roll_samples-c_roll_samples_prev, dim = 1)		
                                
                                
            accumulated_res_fixed_point.append(fixed_point_res)


        accumulated_res_primal_temp = accumulated_res_primal
        accumulated_res_fixed_point_temp = accumulated_res_fixed_point
        
        res_primal_stack = torch.stack(accumulated_res_primal )
        res_fixed_stack = torch.stack(accumulated_res_fixed_point )

        accumulated_res_primal = torch.sum(res_primal_stack, axis = 0)/self.maxiter
        accumulated_res_fixed_point = torch.sum(res_fixed_stack, axis = 0)/self.maxiter

        return c_v_samples, c_pitch_samples, c_roll_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp


    def decoder_function(self, inp_norm, init_state, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input):
        
        # v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init  = init_stat

        v_init = init_state[:, 0]
        v_dot_init = init_state[:, 1]
        pitch_init = init_state[:, 2]
        pitch_dot_init = init_state[:, 3]
        roll_init = init_state[:, 4]
        roll_dot_init = init_state[:, 5]
        

        neural_output_batch = self.mlp(inp_norm)

        
        lamda_samples = neural_output_batch[:, 0 : 3*self.nvar  ]
        c_samples = neural_output_batch[:, 3*self.nvar : 6*self.nvar]

        # s_samples = torch.maximum( torch.zeros(( self.num_batch, 3*self.num_constraint  ), device = device),   neural_output_batch[:, 3*self.nvar: 3*self.nvar+3*self.num_constraint])

        c_v_samples = c_samples[:, 0 : self.nvar ]
        c_pitch_samples = c_samples[:, self.nvar : 2*self.nvar ]
        c_roll_samples = c_samples[:, 2*self.nvar : 3*self.nvar ]

        lamda_v = lamda_samples[:, 0 : self.nvar ]
        lamda_pitch = lamda_samples[:, self.nvar : 2*self.nvar ]
        lamda_roll = lamda_samples[:, 2*self.nvar : 3*self.nvar ]

        # print(lamda_v)
        # print(c_v_samples)

        # s_v = s_samples[:, 0 : self.num_constraint ]
        # s_pitch = s_samples[:, self.num_constraint : 2*self.num_constraint ]
        # s_roll = s_samples[:, 2*self.num_constraint : 3*self.num_constraint ]

        b_eq_v, b_eq_pitch, b_eq_roll = self.compute_boundary_vec(v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init)

        # print(b_eq_v)

        c_v_samples, c_pitch_samples, c_roll_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp = self.custom_forward(lamda_v, lamda_pitch, lamda_roll, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input, b_eq_v, b_eq_pitch, b_eq_roll, c_v_samples, c_pitch_samples, c_roll_samples)
        
        return c_v_samples, c_pitch_samples, c_roll_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp


    def mlp_loss(self, accumulated_res_primal, accumulated_res_fixed_point, c_v_samples, c_v_samples_input, c_pitch_samples, c_pitch_samples_input, c_roll_samples, c_roll_samples_input):		
        # Aug loss
        primal_loss = 0.5 * (torch.mean(accumulated_res_primal))
        fixed_point_loss = 0.5 * (torch.mean(accumulated_res_fixed_point  ))

        proj_loss_v = self.rcl_loss(c_v_samples, c_v_samples_input)
        proj_loss_pitch = self.rcl_loss(c_pitch_samples, c_pitch_samples_input)
        proj_loss_roll = self.rcl_loss(c_roll_samples, c_roll_samples_input)
        
        proj_loss = proj_loss_v+proj_loss_pitch+proj_loss_roll

        # acc_loss = 0.5 * (torch.mean(predict_acc))

        loss = primal_loss+fixed_point_loss+0.1*proj_loss

        # loss = fixed_point_loss+primal_loss

        return primal_loss, fixed_point_loss, loss


    def forward(self,  inp, init_state, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input):
        
        
        # Normalize input
        inp_norm = (inp - self.inp_mean) / self.inp_std

        # Decode y
        c_v_samples, c_pitch_samples, c_roll_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp = self.decoder_function( inp_norm, init_state, c_v_samples_input, c_pitch_samples_input, c_roll_samples_input)
        
            
        return c_v_samples, c_pitch_samples, c_roll_samples, accumulated_res_fixed_point, accumulated_res_primal, accumulated_res_primal_temp, accumulated_res_fixed_point_temp

	
		
  
							   
								
   
   
		

	 
	

  
