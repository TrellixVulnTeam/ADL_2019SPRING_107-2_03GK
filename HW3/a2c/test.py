from model import ActorCritic
import torch
import gym
from PIL import Image
import numpy as np
def test(n_episodes=30, name='LunarLander_TWO.pth'):
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()
    rewards = []
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        rewards.append(running_reward)
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    print('Mean:', np.mean(rewards))
    env.close()
            
if __name__ == '__main__':
    test()
