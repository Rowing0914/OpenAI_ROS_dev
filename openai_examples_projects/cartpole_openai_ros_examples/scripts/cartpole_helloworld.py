#!/usr/bin/env python
import rospy
import gym

rospy.init_node('cartpole_helloworld', anonymous=True, log_level=rospy.INFO)

env = gym.make('CartPole-v0')
env.reset()

rate = rospy.Rate(30)

for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    rate.sleep()
