from gymnasium.vector import AsyncVectorEnv
from torch import nn
import torch
from tqdm.auto import trange
from torch.utils.tensorboard import SummaryWriter

class REINFORCE:
    def __init__(
            self,
            env: AsyncVectorEnv,
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
        self.num_envs = env.num_envs
        self.policy = policy.to(self.device)
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.discount = discount
        self.continuous = continuous_actions
        self.value_net = value_net
        if value_net is not None:
            self.value_net = value_net.to(self.device)
            self.value_optimizer = optimizer(self.value_net.parameters(), lr=value_lr)

    def run_episode(self, max_steps):
        """Run one episode across all envs in parallel, return per-env trajectories."""
        obs, _ = self.env.reset()

        n = self.num_envs
        ep_log_probs = [[] for _ in range(n)]
        ep_rewards   = [[] for _ in range(n)]
        ep_values    = [[] for _ in range(n)]
        done         = [False] * n

        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            logits = self.policy(obs_tensor)

            if self.continuous:
                dist = torch.distributions.Normal(logits, 1.0)
            else:
                dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            if self.continuous:
                log_prob = log_prob.sum(dim=-1)

            if self.value_net is not None:
                v = self.value_net(obs_tensor).squeeze(-1)

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())

            for i in range(n):
                if not done[i]:
                    ep_log_probs[i].append(log_prob[i])
                    ep_rewards[i].append(reward[i])
                    if self.value_net is not None:
                        ep_values[i].append(v[i])
                    if terminated[i] or truncated[i]:
                        done[i] = True

            if all(done):
                break

        return ep_log_probs, ep_rewards, ep_values

    def returns(self, rewards):
        """Compute discounted returns G_t for each timestep."""
        G, running = [], 0.0
        for r in reversed(rewards):
            running = r + self.discount * running
            G.insert(0, running)
        return torch.tensor(G, dtype=torch.float32, device=self.device)

    def learn(self, num_episodes=1000, max_steps=1000, reporter=SummaryWriter()):
        steps = trange(num_episodes)

        for episode in steps:
            all_log_probs, all_G, all_values, all_rewards = [], [], [], []

            ep_log_probs, ep_rewards, ep_values = self.run_episode(max_steps)
            for i in range(self.num_envs):
                if not ep_rewards[i]:
                    continue
                G = self.returns(ep_rewards[i])
                all_log_probs.extend(ep_log_probs[i])
                all_G.append(G)
                all_values.extend(ep_values[i])
                all_rewards.append(sum(ep_rewards[i]))

            if not all_G:
                continue

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

            loss = -torch.stack([lp * a for lp, a in zip(all_log_probs, advantages)]).mean()
            reporter.add_scalar('Policy Loss', loss.item(), episode)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_reward = sum(all_rewards) / len(all_rewards)
            steps.set_postfix({'mean reward': mean_reward})
            reporter.add_scalar('Episode Reward', mean_reward, episode)
