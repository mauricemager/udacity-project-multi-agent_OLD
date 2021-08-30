# individual network settings for each actor + critic pair
# see networkforall for details

import torch, random
import numpy as np
import torch.nn.functional as F

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
from collections import deque, namedtuple

# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent:
    def __init__(self, config):
                 
#                  in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
#         super(DDPGAgent, self).__init__()
    
        self.config = config
        self.seed   = self.config.seed
        
    
        self.actor  = Network(self.config.in_actor, self.config.hidden_in_actor, self.config.hidden_out_actor, 
                              self.config.out_actor, actor=True).to(device)
        self.critic = Network(self.config.in_critic, self.config.hidden_in_critic, 
                              self.config.hidden_out_critic, 1).to(device)
    
        self.target_actor  = Network(self.config.in_actor, self.config.hidden_in_actor, self.config.hidden_out_actor, 
                                     self.config.out_actor, actor=True).to(device)
        self.target_critic = Network(self.config.in_critic, self.config.hidden_in_critic,      
                                     self.config.hidden_out_critic,1).to(device)        

        self.noise = OUNoise(self.config.action_size, scale=1.0)

        
        # initialize targets same as original networks (tau = 1.0 > hard update)
        self.soft_update(self.actor,  self.target_actor,  1.0)
        self.soft_update(self.critic, self.target_critic, 1.0)
        
#         hard_update(self.target_actor, self.actor)
#         hard_update(self.target_critic, self.critic)

        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=config.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.lr_critic, weight_decay=1.e-5)
        

    def act(self, obs, noise=0.0):
        obs = torch.from_numpy(obs).float().to(device) 
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train()
#         print(f'action is of type: {type(action)}')
        
        
        
        
#         action = self.target_actor(obs) + noise * self.noise.noise()
        return action
        
        
#         print(f' obs if type: {type(obs)}')
# #         obs = obs.to(device)
#         action = self.actor(obs) + noise*self.noise.noise()
#         return np.clip(action, -1, 1)

# #     def target_act(self, obs, noise=0.0):
# #         obs = obs.to(device)
# #         action = self.target_actor(obs) + noise*self.noise.noise()
# #         return np.clip(action, -1, 1)
    

    def update(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states)
        print(f" next_states shape =  {next_states.shape}")
#         print(f' type states = {type(next_states)}')
        print(f" next_actions shape =  {actions_next.shape}")
#         target_input1 = torch.cat((next_states, actions_next))
        target_input2 = torch.cat((next_states, actions_next), dim=1)
#         print(f"len target_input1 = {len(target_input1)}")
        print(f"len target_input2 = {target_input2.shape}")

        
        Q_targets_next = self.target_critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss   = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)
        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    
    def reset(self):
        self.noise.reset()

    
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)