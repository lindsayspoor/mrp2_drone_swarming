#### based on tutorial from philtabor on GitHub ####
import torch as T
from ActorCriticNetworks import Actor, Critic
import numpy as np

class RLAgent:
    def __init__(self, N, N_actions, dims_actor, dims_critic, i, directory, alpha1, alpha2, layer1_nodes=64, layer2_nodes=64, gamma=0.99, tau=0.01):

        self.gamma = gamma
        self.tau = tau
        self.N_actions = N_actions
        self.agent = f"agent_{i}"

        self.actor = Actor(alpha1, dims_actor, layer1_nodes, layer2_nodes, N_actions, directory=directory, filename=f"{self.agent}_actor")
        self.critic = Critic(alpha2, dims_critic, layer1_nodes, layer2_nodes, N, N_actions, directory=directory, filename=f"{self.agent}_critic")

        self.target_actor = Actor(alpha1, dims_actor, layer1_nodes, layer2_nodes, N_actions, directory=directory, filename=f"{self.agent}_target_actor")
        self.target_critic = Critic(alpha2, dims_critic, layer1_nodes, layer2_nodes, N, N_actions, directory=directory, filename=f"{self.agent}_target_critic")

        self.update_params(tau=1)


    def select_action(self, observation):

        obs = T.tensor(np.array(observation), dtype=T.float).to(self.actor.device)

        actions = self.actor.forward(obs)

        noise = T.rand(self.N_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()
    
    def update_params(self, tau=None):

        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for k in actor_state_dict:
            actor_state_dict[k] = tau * actor_state_dict[k].clone() + (1-tau) * target_actor_state_dict[k].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        for k in critic_state_dict:
            critic_state_dict[k] = tau * critic_state_dict[k].clone() + (1-tau) * target_critic_state_dict[k].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    
    def save_models(self):
        self.actor.save()
        self.target_actor.save()
        self.critic.save()
        self.target_critic.save()

    
    def load_models(self):
        self.actor.load()
        self.target_actor.load()
        self.critic.load()
        self.target_critic.load()






        

