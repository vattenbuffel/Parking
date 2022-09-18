from network import FeedForwardNN
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import time

class PPO:
    def __init__(self, env) -> None:
        self.init_hyperparameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim) # Figures out the action
        self.critic = FeedForwardNN(self.obs_dim, 1) # Figures out the value of the action
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)


        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.3)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.render_i = 0
        self.save_i = 0

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def init_hyperparameters(self):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 200
        self.gamma = 0.99
        self.n_updates_per_iteration = 10
        self.clip = 0.2 # epsilon
        self.learning_rate = 3e-4
        self.render_every_i = 25
        self.save_every_i = 100

    def get_action(self, obs):
        # Query the actor network for a mean action
        mean = self.actor.forward(obs)

        # Create multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # sample an action from the distribtuion and get it's log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_batch_rewards_to_go(self, batch_rewards):
        # The rewards to go per episode per batch to return
        # the shape vill be (num timesteps per episode)
        # Noa: Compute a discounted value of all future rewards
        batch_rewards_to_go = []

        # Iterate through each episode backwards to maintain same order in batch_rewards_to_go
        for ep_rewards in reversed(batch_rewards): # Why does this need to be reversed?
            discounted_reward = 0

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        return batch_rewards_to_go

    def evaluate(self, batch_observations, batch_actions):
        # Query critic network for a value v for each obs in batch_obs
        V = self.critic(batch_observations).squeeze()

        # Calvulate the log probabilities of batch actions using most reset actor netwrok
        mean = self.actor.forward(batch_observations)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)

        return V, log_probs

    def rollout(self):
        # Batch data
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_lens = [] # episodic lengths in batch

        t = 0
        ep_rewards = []

        while t < self.timesteps_per_batch:
            ep_rewards = [] # Rewards this episode

            if self.render_i == self.render_every_i:
                self.render_i = 0
                render = True
            else:
                self.render_i += 1
                render = False

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                if render:
                    self.env.render()
                
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)

                ep_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if done: 
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        # Convert to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        batch_rewards_to_go = self.compute_batch_rewards_to_go(batch_rewards)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rewards
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rewards_to_go, batch_lens

    def learn(self, total_timesteps):
        t_so_far = 0 # Number of simulated timesteps
        batch_rewards = []
        i_so_far = 0

        # Wrap up your game into a class noa

        while t_so_far < total_timesteps:
            batch_observations, batch_actions, batch_log_probs, batch_rewards_new, batch_rewards_to_go, batch_lens = self.rollout()
            batch_rewards.extend(batch_rewards_new)

            t_so_far += np.sum(batch_lens)

            i_so_far += 1
			# Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_observations, batch_actions)

            A_k = batch_rewards_to_go - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # calculate v_phi and pi_heta(a_t | s_t)
                V, cur_log_probs = self.evaluate(batch_observations, batch_actions)

                # Calculate ratios
                ratios = torch.exp(cur_log_probs - batch_log_probs)

                # Calcualte surrogate losses
                sur1 = ratios * A_k
                sur2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(sur1, sur2)).mean()

                # Backward propagation on actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)

                # Backward propagation on critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
                self.logger['actor_losses'].append(actor_loss.detach())
            
            # Print a summary of our training so far
            self._log_summary()

            if self.save_i % self.save_every_i == 0:
                torch.save(self.actor.state_dict(), f'./ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
                self.save_i  = 0
            else:
                self.save_i += 1


        return batch_rewards

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []



if __name__ == '__main__':
    import gym
    env = gym.make('Pendulum-v0')
    model = PPO(env)
    while True:
        model.learn(100_000)


