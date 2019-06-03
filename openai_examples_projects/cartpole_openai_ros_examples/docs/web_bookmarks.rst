Wiki for Cartpole-v0:
---------------------
https://github.com/openai/gym/wiki/CartPole-v0


Algorithms for Solution Made by users
-------------------------------------

https://gym.openai.com/envs/CartPole-v0/


Whe Cartpole ends at 200 iterations
-----------------------------------
https://github.com/openai/gym/blob/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/envs/__init__.py


Cartpole Env definition
-----------------------
https://github.com/openai/gym/blob/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/envs/classic_control/cartpole.py


Actions and Observations in Demos 2d
------------------------
actions:
********
force = self.force_mag if action==1 else -self.force_mag
Go LEFT
Go Right

observations
************
self.state = (x,x_dot,theta,theta_dot)
x: position of base
x_dot: speed of base
theta: angle of pole
theta_dot: angular speed of pole joint.


Actions and Observations in Demos 3D
------------------------
actions:
********
If you select only 2 actions, the first two will only be done.
Go LEFT
Go Right
Go LEFT * 10
Go Right * 10


observations
************
self.state = (x,x_dot,theta,theta_dot)
x: position of base
x_dot: speed of base
theta: angle of pole
theta_dot: angular speed of pole joint.



