"""
    File contains the Agent class as used in the report
"""

import numpy as np
from torch import nn, optim ,tensor, argmax, save, load, no_grad, from_numpy, sum
import torch
from datetime import datetime
import os
from copy import deepcopy

from replay_mem import Replay_Memory
from neural_nets import Actor_NN,Critic_NN
from noise import OUNoise, GaussNoise


class Agent:
    def __init__(self,state_dim,
                action_dim = 1,
                action_min = 0,
                action_max = 100,
                repl_mem_size = 50000,
                repl_batch_size=128,
                noise_gen = "OUNoise",
                noise_mu = 0.,
                noise_sigma = 0.01,
                noise_decay = None,
                gamma = 0.99,
                actor_learning_rate = 0.0001,
                critic_learning_rate = 0.0001,
                target_update_rate = 0.001,
                optimizer_L2 = 0.0001
                ):
        """ initializes and constructs the agent object

        Args:
            state_dim (int): dimensionality of the statespace
            action_dim (int, optional): dimensionality of the action space. Defaults to 1.
            action_min (int, optional): minimum value of the action. Defaults to 0.
            action_max (int, optional): maximum value of the action. Defaults to 100.
            repl_mem_size (int, optional): total number of samples to be stored. Defaults to 50000.
            repl_batch_size (int, optional): size of sample batch from the replay memory used for each training. Defaults to 128.
            noise_gen (str, optional): noise generator used for exploration. Defaults to "OUNoise".
            noise_mu ([type], optional): mu value for the noise generator. Defaults to 0..
            noise_sigma (float, optional): sigma value for the noise generator. Defaults to 0.01.
            noise_decay ([type], optional): decay value for the noise generator. Defaults to None.
            gamma (float, optional): gamma hyperparameter. Defaults to 0.99.
            actor_learning_rate (float, optional): hyperparameter used for the neural net optimizer. Defaults to 0.0001.
            critic_learning_rate (float, optional): hyperparameter used for the neural net optimizer. Defaults to 0.0001.
            target_update_rate (float, optional): rate to update target networks. Defaults to 0.001.
            optimizer_L2 (float, optional): L2 value as used for the optimizer. Defaults to 0.0001.
        """
        assert isinstance(state_dim, int) and state_dim >= 2 
        assert isinstance(action_dim, int) and action_dim >= 1
        assert isinstance(repl_mem_size,int)
        assert isinstance(repl_batch_size,int) and repl_batch_size <= repl_mem_size
        assert noise_gen == "OUNoise" or noise_gen == "GaussNoise"
        assert gamma >= 0 and gamma <= 1
        assert actor_learning_rate >= 0 and actor_learning_rate <= 1
        assert critic_learning_rate >= 0 and critic_learning_rate <= 1
        assert target_update_rate > 0 and target_update_rate <= 1
        assert optimizer_L2 >= 0 and optimizer_L2 <= 1
        
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.action_min = torch.tensor([action_min])
        self.action_max = torch.tensor([action_max])
        
        # store the parameters
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.noise_decay = noise_decay
        self.noise_gen_name = noise_gen
        if noise_gen == "OUNoise":
            self.noise_gen = OUNoise(self.noise_mu,theta=0.15,sigma=self.noise_sigma,decay=noise_decay)
        elif noise_gen == "GaussNoise":
            self.noise_gen = GaussNoise(self.noise_mu,sigma=self.noise_sigma,decay=noise_decay)
        self.gamma = gamma
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.optimizer_L2 = optimizer_L2
        self.target_update_rate = target_update_rate
        
        #initialize the Replay memory
        self.repl_mem = Replay_Memory(repl_mem_size,repl_batch_size,state_dim)

        # initialize the policy and target network
        self.actor_nn = Actor_NN(state_dim,self.action_dim)
        self.actor_target_nn = Actor_NN(state_dim,self.action_dim)
        self.critic_nn = Critic_NN(state_dim,self.action_dim)
        self.critic_target_nn = Critic_NN(state_dim,self.action_dim)
        self.critic_loss = nn.MSELoss()
        
        # make sure networks and target networks are identical at start
        self.actor_target_nn.load_state_dict(self.actor_nn.state_dict())
        self.critic_target_nn.load_state_dict(self.critic_nn.state_dict())

        # initialize the optimizers for both actor and critic
        self.actor_optimizer = optim.Adam(self.actor_nn.parameters(),
                                          lr = self.actor_lr,weight_decay=self.optimizer_L2)
        self.critic_optimizer = optim.Adam(self.critic_nn.parameters(),
                                          lr = self.critic_lr,weight_decay=self.optimizer_L2)
        
        self.actor_params_grads = {}
        for name, param in self.actor_nn.named_parameters():
            self.actor_params_grads[name] = torch.zeros_like(param.data)

        # rest the logging for the training
        self.log_samples = None
        self.log_episodes = None
        self.log_rewards = None
        self.log_avg_rewards = None

        
    def __str__(self):
        """ gives a summary of the agent settings as a string

        Returns:
            [String]: summary of the agent configuration
        """
        return  f"mem: {self.repl_mem.memory_size}, batch: {self.repl_mem.batch_size}, " + \
                f"noise: {self.noise_gen_name}, noise_mu: {self.noise_mu}, noise_sigma: {self.noise_sigma}, noise_decay {self.noise_decay}, " + \
                f"gamma: {self.gamma}, LR_A: {self.actor_lr}, LR_C: {self.critic_lr}, " + \
                f"TN_upd: {self.target_update_rate}, optim_L2: {self.optimizer_L2}"

    def init_logging(self,samples):
        """initializes the logs and sets the size in accordance to the number of expected samples given

        Args:
            samples (int): number of samples in total to log (can differ from episodes if not all episodes ar logged)
        """
        self.log_samples = samples
        self.log_sample = 0
        self.log_episodes = np.zeros(samples,dtype=int)
        self.log_rewards = np.zeros(samples,dtype=float)
        self.log_avg_rewards = np.zeros(samples,dtype=float)

    def log(self,episode,reward,avg_reward):
        """adds a sample to the log

        Args:
            episode (int): which episode the logged values are from
            reward (float): the actual reward for that episode
            avg_reward (float): the average reward in the specific episode
        """
        assert self.log_samples is not None
        
        if self.log_sample >= self.log_samples:
            print("ERROR: sample logs full") 
        else:
            self.log_episodes[self.log_sample] = episode
            self.log_rewards[self.log_sample] = reward
            self.log_avg_rewards[self.log_sample] = avg_reward
        self.log_sample +=1

    
    def select_action(self,state,episode=1,greedy=False):
        """ select action with noise if greedy is set to false, without noise otherwise

        Args:
            state (np.Array): the actual state of the agent
            episode (int): the episode of the game 
            greedy (bool, optional): if set to True the agent works greedy (always selects optimal action). 
                                     Defaults to False.
        
        Returns:
            [int]: the value of the action to take
        """
        with no_grad():
            self.actor_nn.eval()
            orig_action = self.actor_nn(state)
        
        noise = 0
        if not greedy:
            noise = self.noise_gen.noise(episode)
        action = orig_action + noise
   
        action = torch.clamp(action,self.action_min,self.action_max)
        return action


    def invert_gradients(self,gradients, actions):
        """inverts the gradients in order to keep actor action values betweens limits

        Args:
            gradients (tensor): the gradients as provided by the loss function
            actions (tensor): the action in the sample space

        Returns:
            [type]: [description]
        """
        
        action_range = (self.action_max-self.action_min) 
        index = gradients>0

        gradients[index] *=  (index.float() * (self.action_max - actions)/action_range)[index]
        gradients[~index] *= ((~index).float() * (actions- self.action_min)/action_range)[~index]

        return gradients	

        
    def target_update(self,source_nn, target_nn, update_rate):
        """ updates the target weights with a given update rate

        Args:
            source_nn (nn.Module): the source neural net
            target_nn (nn.Module): the target neural net
            update_rate (int): the rate for which to update the target weights
        """
        for source_param, target_param in zip(source_nn.parameters(), target_nn.parameters()):
                target_param.data.copy_(update_rate * source_param.data + (1.0 - update_rate) * target_param.data)


    def update_networks(self,state,action,reward,next_state):
        """adds sample to replay memory, 
            perform Q-learning on online network if minimal number of samples in replay memory is reached 
            and update target network if interval is passed

        Args:
            state (tensor): the state in which the agent performed the action
            action (int): the value of the action taken between state and next_state
            reward (float): the reward received for arriving in next_state
            next_state (np.Array): the state in which the agent arrived

        """
        self.repl_mem.add_timestep(state,action,reward,next_state)
        
        # updating networks is only allowed if sufficient samples are in the replay memory
        if self.repl_mem.is_batch_size_reached():
            
            # collect a batch from the replay memory
            states,actions,rewards,rewards,next_states = self.repl_mem.get_new_batch()
            
            self.actor_nn.train()
            self.critic_nn.train()
            self.actor_target_nn.eval()
            self.critic_target_nn.eval()
            
            # determine critic loss
            with torch.no_grad():
                next_state_actions = self.actor_target_nn(next_states)
                next_state_q_values = self.critic_target_nn(next_states,next_state_actions)
                expected_q_values = rewards.unsqueeze(1) + (self.gamma * next_state_q_values)
            q_values = self.critic_nn(states,actions)
            critic_loss = self.critic_loss(expected_q_values,q_values)  

            # optimize critic
            self.critic_optimizer.zero_grad()  
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # determine actor loss
            actor_actions = torch.autograd.Variable(self.actor_nn(states))
            actor_actions.requires_grad = True
            actor_loss = self.critic_nn(states, actor_actions).mean()
            self.critic_nn.zero_grad()
            actor_loss.backward()
            actor_gradients = deepcopy(actor_actions.grad.data)
            
            # invert gradients
            actor_gradients = self.invert_gradients(actor_gradients,actor_actions)
            actor_actions = self.actor_nn(states)
            out = -torch.mul(actor_gradients, actor_actions)
            self.actor_nn.zero_grad()
            out.backward(torch.ones(out.shape))

            # optimize actor
            self.actor_optimizer.step()

            # update target networks
            self.target_update(self.critic_nn, self.critic_target_nn,self.target_update_rate)
            self.target_update(self.actor_nn, self.actor_target_nn, self.target_update_rate)
            
            
###################################

    def get_filename(self):
        """aux function that composes the filename for the saved agent

        Returns:
            [String]: filename for the stored agent
        """
        now = datetime.now()
        
        time_string = now.strftime("%Y%m%d %H%M%S")
        
        return  f"statedim{self.state_dim}_actiondim{self.action_dim}_" + \
                f"mem{self.repl_mem.memory_size}_batch{self.repl_mem.batch_size}_" + \
                f"nmu{self.noise_mu}_nsigma{self.noise_sigma}_" + \
                f"g{self.gamma}_LRA{self.actor_lr}_LRC{self.critic_lr}_" + \
                f"TNupd{self.target_update_rate}_optimL{self.optimizer_L2}_" + \
                time_string + ".pt"  

    def save(self, checkpoint_name=None):
        """saves the (trained) agent together with the logging of the training 

        Args:
            checkpoint_name ([String], optional): an optional filename can be passed. Defaults to None.
                                                  if none is given, standard filename will be used from get_filename
        """
        path = os.getcwd()
        
        if checkpoint_name is None:
            checkpoint_name = self.get_filename()
        
        checkpoint = {
            'agent_name': self.__str__(),
            'state_dim': self.state_dim,
            'action_dim':  self.action_dim,
            'action_min': self.action_min,
            'action_max': self.action_max,
            'repl_mem_size': self.repl_mem.memory_size,
            'repl_batch_size': self.repl_mem.batch_size,
            'noise_gen': self.noise_gen.name,
            'noise_mu': self.noise_mu,
            'noise_sigma': self.noise_sigma,
            'noise_decay': self.noise_decay,
            'gamma': self.gamma,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'target_update_rate': self.target_update_rate,
            'optimizer_L2': self.optimizer_L2,
    
            'actor_nn': self.actor_nn.state_dict(),
            'actor_target_nn': self.actor_target_nn.state_dict(),
            'critic_nn': self.critic_nn.state_dict(),
            'critic_target_nn': self.critic_target_nn.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_episodes': self.log_episodes,
            'log_reward': self.log_rewards,
            'log_avg_reward': self.log_avg_rewards
        }
        save(checkpoint, os.path.join(path,"saved_agents", checkpoint_name))

    @classmethod
    def load(cls,filename):
        """loads an existing agent from file

        Args:
            filename (String): the name of the file containing the agent

        Raises:
            OSError: error is raised when file could not get loaded

        Returns:
            [Agent]: the actual agent from the file
        """
        if os.path.isfile(filename):
            
            checkpoint = load(filename)

            agent = Agent(  state_dim = checkpoint['state_dim'],
                            action_dim = checkpoint['action_dim'],
                            action_min = checkpoint['action_min'],
                            action_max = checkpoint['action_max'],  
                            repl_mem_size = checkpoint['repl_mem_size'],
                            repl_batch_size = checkpoint['repl_batch_size'],
                            noise_gen = checkpoint['noise_gen'],
                            noise_mu = checkpoint['noise_mu'],
                            noise_sigma = checkpoint['noise_sigma'],
                            noise_decay = checkpoint['noise_decay'],
                            gamma = checkpoint['gamma'],
                            actor_learning_rate = checkpoint['actor_lr'],
                            critic_learning_rate = checkpoint['critic_lr'],
                            target_update_rate = checkpoint['target_update_rate'],
                            optimizer_L2 = checkpoint['optimizer_L2'])
            
            agent.actor_nn.load_state_dict(checkpoint['actor_nn'])
            agent.actor_target_nn.load_state_dict(checkpoint['actor_target_nn'])
            agent.critic_nn.load_state_dict(checkpoint['critic_nn'])
            agent.critic_target_nn.load_state_dict(checkpoint['critic_target_nn'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
            agent.log_episodes = checkpoint['log_episodes']
            agent.log_rewards = checkpoint['log_reward']
            agent.log_avg_rewards = checkpoint['log_avg_reward']

            print('checkpoint loaded at {}'.format(filename))
            return agent
        else:
            raise OSError("Checkpoint file not found.")    
