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
        self.env = env
        self.policy = policy
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.discount = discount
        self.continuous = continuous_actions
        self.value_net = value_net
        if value_net is not None:
            self.value_optimizer = optimizer(self.value_net.parameters(), lr=value_lr)

    def run_episode(self, max_steps):
        """Run one episode, return lists of log-probs, rewards, and state values."""
        obs, _ = self.env.reset()
        log_probs, rewards, values = [], [], []

        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = self.policy(obs_tensor)
            if self.continuous:
                dist = torch.distributions.Normal(logits, 1.0)  # stddev=1.0 (change ?)
            else:
                dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()

            if self.continuous:
                log_probs.append(dist.log_prob(action).sum())
                env_action = action.numpy()
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
        return torch.tensor(G, dtype=torch.float32)

    def learn(self, num_episodes = 1000, max_steps = 1000, reporter = SummaryWriter()):
        steps = trange(num_episodes)

        for _ in steps:
            log_probs, rewards, values = self.run_episode(max_steps)
            G = self.returns(rewards)

            if self.value_net is not None:
                V = torch.stack(values)
                advantages = (G - V.detach())
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                value_loss = nn.functional.mse_loss(V, G)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                reporter.add_scalar('Value Loss', value_loss.item(), steps.n)
                reporter.add_scalar('Average Advantage', advantages.mean().item(), steps.n)
            else:
                advantages = (G - G.mean()) / (G.std() + 1e-8)
                reporter.add_scalar('Average Return', G.mean().item(), steps.n)

            # -sum( A_t * log policy(a_t|s_t) )
            loss = -torch.stack([lp * a for lp, a in zip(log_probs, advantages)]).sum()
            reporter.add_scalar('Policy Loss', loss.item(), steps.n)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            steps.set_postfix({'total reward': sum(rewards)})
            reporter.add_scalar('Episode Reward', sum(rewards), steps.n)
