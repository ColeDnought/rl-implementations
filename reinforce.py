from gymnasium import Env
from torch import nn
import torch
from tqdm.auto import trange
from torch.utils.tensorboard import SummaryWriter

class REINFORCE:
    def __init__(
            self,
            env: Env,
            policy: nn.Module,
            continuous_actions = False,
            discount = 0.9,
            optimizer = torch.optim.AdamW,
            lr = 1e-3,
            value_net: nn.Module = None,
            value_lr = 1e-3,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.policy = policy.to(self.device)
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.discount = discount
        self.continuous = continuous_actions
        self.value_net = value_net
        if value_net is not None:
            self.value_net = value_net.to(self.device)
            self.value_optimizer = optimizer(self.value_net.parameters(), lr=value_lr)

    def run_episode(self, max_steps):
        """Run one episode, return lists of log-probs, rewards, and state values."""
        obs, _ = self.env.reset()
        log_probs, rewards, values = [], [], []

        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            logits = self.policy(obs_tensor)
            if self.continuous:
                dist = torch.distributions.Normal(logits, 1.0)  # stddev=1.0 (change ?)
            else:
                dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()

            if self.continuous:
                log_probs.append(dist.log_prob(action).sum())
                env_action = action.cpu().numpy()
            else:
                log_probs.append(dist.log_prob(action))
                env_action = action.item()

            if self.value_net is not None:
                values.append(self.value_net(obs_tensor).squeeze())

            obs, reward, terminated, truncated, _ = self.env.step(env_action)
            rewards.append(reward)

            if terminated or truncated:
                break

        return log_probs, rewards, values

    def returns(self, rewards): # Higher gamma working better for Acrobot
        """Compute discounted returns G_t for each timestep."""
        G, running = [], 0.0
        for r in reversed(rewards):
            running = r + self.discount * running
            G.insert(0, running)
        return torch.tensor(G, dtype=torch.float32, device=self.device)

    def learn(self, num_episodes = 1000, max_steps = 1000, reporter = SummaryWriter(), batch_size=1):
        steps = trange(num_episodes)

        for episode in steps:
            all_log_probs, all_G, all_values, all_rewards = [], [], [], []

            for _ in range(batch_size):
                log_probs, rewards, values = self.run_episode(max_steps)
                G = self.returns(rewards)
                all_log_probs.extend(log_probs)
                all_G.append(G)
                all_values.extend(values)
                all_rewards.append(sum(rewards))

            G_cat = torch.cat(all_G)

            if self.value_net is not None:
                V = torch.stack(all_values)
                advantages = G_cat - V.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                value_loss = nn.functional.mse_loss(V, G_cat)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                reporter.add_scalar('Value Loss', value_loss.item(), episode)
                reporter.add_scalar('Average Advantage', advantages.mean().item(), episode)
            else:
                advantages = (G_cat - G_cat.mean()) / (G_cat.std() + 1e-8)
                reporter.add_scalar('Average Return', G_cat.mean().item(), episode)

            # -mean( A_t * log policy(a_t|s_t) ) across all timesteps in batch
            loss = -torch.stack([lp * a for lp, a in zip(all_log_probs, advantages)]).mean()
            reporter.add_scalar('Policy Loss', loss.item(), episode)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_reward = sum(all_rewards) / batch_size
            steps.set_postfix({'mean reward': mean_reward})
            reporter.add_scalar('Episode Reward', mean_reward, episode)
