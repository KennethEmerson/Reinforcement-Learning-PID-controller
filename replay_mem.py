"""
    file contains the Replay Memory Class which is used by the agent
"""
import torch
import random

class Replay_Memory:
    def __init__(self,memory_size,batch_size,state_size,action_size=1):
        """initialize the replay memory object

        Args:
            memory_size (int): total size of the memory in number of samples
            batch_size (int): number of samples in each batch for the network update
            state_size (int): dimensionality of the statespace
            action_size (int): dimensionality of the actionspace
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.samples_in_memory = 0
        self.reset()


    def reset(self):
        """ resets the replay memory
        """
        self.mem_pointer = 0
        self.states = torch.zeros(self.memory_size,self.state_size).float()
        self.actions = torch.zeros(self.memory_size,self.action_size).float()
        self.rewards = torch.zeros(self.memory_size).float()
        self.next_states = torch.zeros((self.memory_size,self.state_size)).float()


    def add_timestep(self,state,action,reward,next_state):
        """adds a timestep/sample to the memory

        Args:
            state (np.array): the actual state from which the agent moved
            action (int): the action taken by the agent in "state"
            reward (float): the reward received by entering next_state
            next_state (np.array): the state in which the agent arrived after performing action in state
        """
        assert(self.state_size  == state.size()[0])
        assert(self.state_size == next_state.size()[0])
       
        self.states[self.mem_pointer] = state
        self.actions[self.mem_pointer] = action
        self.rewards[self.mem_pointer] = reward
        self.next_states[self.mem_pointer] = next_state
        self.mem_pointer += 1
        if self.mem_pointer >= self.memory_size:
            self.mem_pointer = 0

        if self.samples_in_memory < self.memory_size:
            self.samples_in_memory += 1


    def is_batch_size_reached(self):
        """checks if the minimum number of samples is in the memory to create a batch

        Returns:
            [bool]: true is the minimal number of samples is available to create a batch
        """
        return self.samples_in_memory >= self.batch_size

    
    def get_new_batch(self):
        """creates a new batch from the replay memory

        Returns:
            tuple: tuple containing the tensors with states,actions,rewards,next_states
        """
        assert self.samples_in_memory >= self.batch_size

        #only takes actual samples into account for sampling minibatch hence "samples_in_memory"
        samples = random.sample(range(self.samples_in_memory), self.batch_size) 
        indexes = torch.tensor(samples)

        states = self.states.index_select(0,indexes)
        actions = self.actions.index_select(0,indexes)
        rewards = self.rewards.index_select(0,indexes)
        next_states = self.next_states.index_select(0,indexes)

        return states,actions,rewards,rewards,next_states