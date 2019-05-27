#!/usr/bin/env python
import rospy
import gym

rospy.init_node('cartpole_helloworld', anonymous=True, log_level=rospy.INFO)

env = gym.make('CartPole-v0')
env.reset()

rate = rospy.Rate(30)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        rate.sleep()