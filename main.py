import gym
import gym_hypercube
import numpy as np

id = gym_hypercube.dynamic_register(n_dimensions=2,
                                    env_description={'high_reward_value': 1,
                                                     'low_reward_value': 0.1,
                                                     'nb_target': 1,
                                                     'mode': 'random',
                                                     'agent_starting': 'fixed',
                                                     'speed_limit_mode': 'vector_norm'},
                                    continuous=True,
                                    acceleration=True,
                                    reset_radius=None)

env = gym.make(id)
s = env.reset()
print(s)
d = False
total_reward = 0
goal = np.asarray([-0.6, -0.6, 0, 0])
while not d:
    env.render()
    obs, r, d, info = env.step(env.action_space.sample())
    total_reward += r
env.close()
print(total_reward)
