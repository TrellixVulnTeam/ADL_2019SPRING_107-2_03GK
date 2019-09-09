import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" )#if torch.cuda.is_available() else "cpu"

class AgentMario:
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.9
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 16
        self.seed = 7122
        self.max_steps = 1e7
        self.grad_norm = 0.5
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = True # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 4000
        self.save_freq = 100000
        self.save_dir = './checkpoints/'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 
        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, 
                eps=1e-5)

        self.hidden = None
        self.init_game_setting()

    ####
    def calc_actual_state_values(self, rewards, dones):
        R = []
        rewards.reverse()

        # If we happen to end the set on a terminal state, set next return to zero
        if dones[-1] == True:
            next_return = 0

        # If not terminal state, bootstrap v(s) using our critic
        # TODO: don't need to estimate again, just take from last value of v(s) estimates
        else:
            s = torch.from_numpy(self.rollouts.obs[-1]).float().unsqueeze(0)#states
            next_return = self.model.get_state_value(Variable(s)).data[0][0]

            # Backup from last state to calculate "true" returns for each state in the set
        R.append(next_return)
        dones.reverse()
        for r in range(1, len(rewards)):
            if not dones[r]:
                this_return = rewards[r] + next_return * self.gamma
            else:
                this_return = 0
            R.append(this_return)
            next_return = this_return

        R.reverse()
        state_values_true = Variable(torch.FloatTensor(R)).unsqueeze(1)

        return state_values_true
   ####

    def _update(self):
        # TODO: Compute returns
        # R_t = reward_t + gamma * R_{t+1}
        state_values_true = self.calc_actual_state_values(self.rollouts.rewards, self.rollouts.dones)#(rewards, dones)#from storage: obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy()); obs =state?

        # TODO:
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)

        s = Variable(torch.FloatTensor(self.rollouts.obs))
        action_probs, state_values_est, hiddens = self.model(s)#action_probs, state_values_est
        action_log_probs = action_probs.log()
        a = Variable(torch.LongTensor(self.rollouts.actions).view(-1, 1))
        chosen_action_log_probs = action_log_probs.gather(1, a)
        # This is also the TD error
        advantages = state_values_true - state_values_est
        entropy = (action_probs * action_log_probs).sum(1).mean()
        action_loss = (chosen_action_log_probs * advantages).mean()
        value_loss = advantages.pow(2).mean()


        loss = value_loss + action_loss -  0.0001 * entropy#entropy_weight = 0.0001
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())
        RolloutStorage.reset()##

        return loss.item()

    def _step(self, obs, hiddens, masks):

        with torch.no_grad():
            pass
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            actions, values, hiddens = self.make_action(obs, hiddens, masks)
        #print("##################################*****************",actions.cpu().numpy(),type(actions.cpu().numpy()),actions.cpu().numpy().shape)
        #print("##################################*****************",actions.max(1)[0].item())
        obs, rewards, dones, infos = self.envs.step(actions.max(1)[0])#.numpy().max(0)[0].item())

        # TODO:
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)

        self.rollouts.to(device)
        masks = 1 - dones
        self.rollouts.insert(obs, hiddens, actions, values, rewards, masks)
        self.rollouts.to(device)

    def train(self):

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        self.rollouts.obs[0].copy_(obs)#torch.Size([16, 4, 84, 84])
        self.rollouts.to(self.device)
        
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                print("# ******************step***********************", step)
                #print("self.rollouts.actions[step]", self.rollouts.actions[step])
                # print("self.rollouts.obs[step]", self.rollouts.hiddens[step])
                # print("self.rollouts.obs[step]", self.rollouts.masks[step])
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                    if m == 0:
                        running_reward.append(r.item())
                episode_rewards *= self.rollouts.masks[step + 1]

            loss = self._update()
            total_steps += self.update_freq * self.n_processes

            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
            
            if total_steps % self.save_freq == 0:
                self.save_model('model.pt')
            
            if total_steps >= self.max_steps:
                break

    def save_model(self, filename):
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        self.model = torch.load(path)

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, hiddens, masks, test=False):
        # TODO: Use you model to choose an action
        # if test == True:
        #     observation = torch.from_numpy(observation).permute(2, 0, 1).unsqueeze(0).to(device)
        # print("!!!!!!!!!!!!!!",observation.shape)
        # state = torch.from_numpy(observation).float().unsqueeze(0)
        values, action_probs, hiddens = self.model(observation, hiddens, masks)

        # m = Categorical(action_probs)
        # action = m.sample()
        # #self.saved_actions.append(m.log_prob(action))

        return action_probs, values, hiddens
