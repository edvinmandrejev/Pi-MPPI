# π-MPPI: A Projection-based Model Predictive Path Integral Scheme for Smooth Optimal Control of Fixed-Wing Aerial Vehicles

## Getting Started
1. Clone the repository
```
git clone https://github.com/edvinmandrejev/Pi-MPPI.git
```
2. Install dependencies
```
pip install -r requirements.txt
```
## Obstacle Avoidance Scenario
1. Comparison: π-MPPI and MPPIwSGF
   Obstacle avoidanance scenario for baseline MPPIwSGF and π-MPPI can be run using [Obstacle avoidance (MPPIwSGF and π-MPPI)](https://github.com/edvinmandrejev/Pi-MPPI/blob/main/Obstacle%20Avoidance/Comparison%3A%20Pi-MPPI%20and%20Baseline-MPPIwSGF/obstacle_avoidance.ipynb).
2. Comparison: π-MPPI and π-MPPI learned
   Obstacle avoidanance scenario for baseline π-MPPI and π-MPPI learned can be run using [Obstacle avoidance (π-MPPI learned)](https://github.com/edvinmandrejev/Pi-MPPI/blob/main/Obstacle%20Avoidance/Comparison%3A%20PI-MPPI%20learned/obst_avoidance_comparison_learned_vs_proj.ipynb).
## Terrain Following example
1. Comparison: π-MPPI and MPPIwSGF
  Terrain following scenario for baseline MPPIwSGF and π-MPPI can be run using [Terrain following(MPPIwSGF and π-MPPI)](https://github.com/edvinmandrejev/Pi-MPPI/blob/main/Terrain%20Following/Comparison%3A%20Pi-MPPI%20and%20Baseline-MPPIwSGF/terrain_following.ipynb)
2. Comparison: π-MPPI and π-MPPI learned
  Terrain following scenario for baseline π-MPPI and π-MPPI learned can be run using [Terrain following(π-MPPI learned)](https://github.com/edvinmandrejev/Pi-MPPI/blob/main/Terrain%20Following/Comparison%3A%20PI-MPPI%20learned/terrain_baseline_learned_comparison.ipynb)

## Visualisation
Visualization was developed with the help from the following repos.
```
https://github.com/eyler94/EE674LQR.git
http://gitlab.hh.se/jendav/Gazebo_Harbour_Models.git
https://github.com/andreasBihlmaier/pysdf.git
```
### Obstacle avoidance scenario:
```
python3 visualisation_obstacle.py
```
### Terrain Following scenario:
Launch the environment
```
roslaunch cessna_plotter model_disp.launch
```
Initialise the point cloud (terrain)
```
python3 point_cloud.py 
```
Initialise the trajectory (path)
```
python3 path.py
```
Run the visualisation
```
python3 move.py
```

