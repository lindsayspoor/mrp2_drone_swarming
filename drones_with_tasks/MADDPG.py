import torch as T
import torch.nn.functional as F
from RL_Agent import RLAgent
import numpy as np

class MADDPG:
    def __init__(self, dims_actor, dims_critic, N, N_actions, task='task1', alpha1=0.01, alpha2=0.01, layer1_nodes=64, layer2_nodes=64, gamma=0.99, tau=0.01, directory = 'checkpoints/'):

        self.agents = []
        self.N = N
        self.N_actions = N_actions
        directory += task

        for i in range(self.N):
            # print(f"{dims_actor[i]=}")
            self.agents.append(RLAgent(N, N_actions, dims_actor[i], dims_critic, i, directory, alpha1, alpha2))

    def save(self):
        print("...saving checkpoint...")
        for agent in self.agents:
            agent.save_models()

    def load(self):
        print('...loading checkpoint...')
        for agent in self.agents:
            agent.load_models()

    def select_action(self, observations):

        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(observations[i])
            action = np.argmax(action)
            actions.append(action)

        return actions
    
    def learn(self, buffer):
        
        if not buffer.is_ready_to_sample():
            return
        print(f"ready to sample")
        actor_states, states, actions, rewards, actor_new_states, new_states, dones = buffer.sample()

        device = self.agents[0].actor.device

        states = T.tensor(np.array(states), dtype=T.float).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float).to(device)
        rewards = T.tensor(np.array(rewards), dtype=T.float).to(device)
        new_states = T.tensor(np.array(new_states), dtype=T.float).to(device)
        dones = T.tensor(np.array(dones)).to(device)

        all_new_actions = []
        all_new_mu_actions = []
        all_old_actions = []

        for i, agent in enumerate(self.agents):
            new_actor_states = T.tensor(actor_new_states[i], dtype=T.float).to(device)

            new_action = agent.target_actor.forward(new_actor_states)
            print(f"did it, {i}")
            all_new_actions.append(new_action)
            mu_states = T.tensor(actor_states[i], dtype=T.float).to(device)

            mu_action = agent.actor.forward(mu_states)

            all_new_mu_actions.append(mu_action)

            all_old_actions.append(actions[i])

        new_actions = T.cat([a for a in all_new_actions], dim=1)
        mu = T.cat([a for a in all_new_mu_actions], dim=1)
        old_actions = T.cat([a for a in all_old_actions], dim=1)

        for i, agent in enumerate(self.agents):
            
            new_critic_value = agent.target_critic.forward(new_states, new_actions).flatten()
            print(f"did the target critic forward, {i}")
            new_critic_value[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()
            print(f"critic forward {i}")
            target = rewards[:,i] + agent.gamma * new_critic_value
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            
            critic_loss.backward(retain_graph=True)
            print(f" did the critic loss {i}")
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_params()
            print("end of loop")
