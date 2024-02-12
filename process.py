"""
    File contains the process class which allows the manipulation of the process
"""
import torch
import numpy as np

class Process_SISO:
    def __init__(self,
                 name,
                 diff_function,
                 setpoint_min,
                 setpoint_max,
                 process_signal_min,
                 process_signal_max,
                 actuator_signal_min,
                 actuator_signal_max,
                 disturbance_signal_min,
                 disturbance_signal_max,
                 episode_size = 200, 
                 reward_function = "L1"):
        """initialises the process object

        Args:
            name ([String]): The name of the process
            diff_function ([function]): the differential equation to use
            setpoint_min ([float]): the minimal setpoint value
            setpoint_max ([float]): the maximal setpoint value
            process_signal_min ([float]): the minimal process signal value
            process_signal_max ([float]): the maximal process signal value
            actuator_signal_min ([float]): the minimal actuator signal value
            actuator_signal_max ([float]): the maximal actuator signal value
            disturbance_signal_min ([float]): the minimal disturbance signal value
            disturbance_signal_max ([float]): the maximal disturbance signal value
            episode_size (int, optional): the number of timesteps in one episode. Defaults to 200.
            reward_function (str, optional): the reward function to use. Defaults to "L1".

        Raises:
            ValueError: gives value error if the reward function is unknown
        """

        # set parameters
        self.name = name
        self.diff_function = diff_function
        self.setpoint_min = setpoint_min
        self.setpoint_max = setpoint_max
        self.process_signal_min = process_signal_min
        self.process_signal_max = process_signal_max
        self.actuator_signal_min = actuator_signal_min
        self.actuator_signal_max = actuator_signal_max
        self.disturbance_signal_min = disturbance_signal_min
        self.disturbance_signal_max = disturbance_signal_max
        self.previous_process_signal = None
        self.process_signal = None
        self.previous_setpoint = None
        self.setpoint = None
         
        # initialize logging arrays
        self.episode_size = episode_size
        self.timesteps = np.arange(0,episode_size) 
        self.actuator_signals = np.zeros(episode_size)
        self.process_signals = np.zeros(episode_size)
        self.setpoints = np.zeros(episode_size)
        self.rewards = np.zeros(episode_size)
        
        # select correct reward function
        if reward_function == "monotonic":
            self.reward_function = self.reward_monotonic
        elif reward_function == "monotonic_improved":
            self.reward_function = self.reward_monotonic_improved
        elif reward_function == "L1":
            self.reward_function = self.reward_L1
        else:
            raise ValueError("reward function not defined")

        
    def __str__(self):
        """returns the name of the process as a string

        Returns:
            [String]: name of the process
        """
        return self.name
    
             
    def reward_L1(self):
        """reward function based on L1

        Returns:
            [float]: the actual reward
        """
        return -abs(self.process_signal - self.setpoint)
    

    def reward_monotonic(self):
        """reward function based on monotomic decrease of error signal

        Returns:
            [float]: the actual reward
        """
        reward = -1
        if abs(self.previous_process_signal-self.previous_setpoint) > abs(self.process_signal-self.setpoint):
            reward = 1
        return reward
    
    def reward_monotonic_improved(self): 
        """reward function based on monotomic decrease of error signal and small enough error signal

        Returns:
            [float]: the actual reward
        """
        reward = -1
        if abs(self.previous_process_signal-self.previous_setpoint) > abs(self.process_signal-self.setpoint):
            reward = 0
        if 0.01 > (abs(self.process_signal-self.setpoint)/ (self.process_signal_max-self.process_signal_min)):
            reward = 1
        return reward
    

    def new_setpoint(self,setpoint):
        """updates the setpoint

        Args:
            setpoint ([Float]): the new setpoint value
        """
        self.setpoint = setpoint
 

    def get_episode(self):
        """get the loggin of the complete episode

        Returns:
            [np.array,np.array,np.array,np.array,np.array,np.array]: logging of timesteps, actuator signals,
                                                                      process signals, setpoints, disturbance signals
        """
        return self.timesteps, self.actuator_signals,self.process_signals,self.setpoints,self.rewards,self.disturbances


    def get_state(self):
        """returns the actual state as a tensor with:
            - actual process_signal value
            - change in process_signal value in respect to previous process_signal
            - deviation from setpoint
        Returns:
            [Tensor]: tensor containing the state
        """
        process_signal = self.process_signal
        process_signal_change = self.process_signal-self.previous_process_signal
        setpoint_deviation = self.setpoint-self.process_signal
        return torch.tensor([process_signal,process_signal_change,setpoint_deviation])
    

    def get_state_size(self):
        """returns the state size

        Returns:
            [int]: in this case state size is 3
        """
        return 3

    
    def initialize(self,initial_actuator_signal,initial_process_signal,initial_disturbance,setpoint):
        """initializes the process

        Args:
            initial_actuator_signal ([Float]): the initial actuator value
            initial_process_signal ([Float]): the initial process signal value
            initial_disturbance ([Float]): the initial disturbance signal value
            setpoint ([function]): the function that describes the evolution of the setpoint

        Returns:
            [Tensor]: returns the initial state
        """
        self.timesteps = np.arange(0,self.episode_size) 
        self.actuator_signals = np.zeros(self.episode_size)
        self.process_signals = np.zeros(self.episode_size)
        self.setpoints = np.zeros(self.episode_size)
        self.disturbances = np.zeros(self.episode_size)
        self.rewards = np.zeros(self.episode_size)

        self.timestep = 0
        self.previous_process_signal = initial_process_signal
        self.process_signal = initial_process_signal
        self.previous_setpoint = setpoint
        self.setpoint = setpoint
        
        return self.get_state() 


    def step(self,actuator_signal,disturbance, setpoint):
        """returns the next state and reward based on the actual state, disturbance and action taken

        Args:
            actuator_signal ([Float]): the actual actuator signal hence action
            disturbance ([Float]): the actual disturbance signal
            setpoint ([Float]): the actual setpoint

        Returns:
            [Tensor,float]: returns the next state and the reward
        """
        self.previous_process_signal = self.process_signal
        self.previous_setpoint = self.setpoint
        self.setpoint = setpoint

        if isinstance(actuator_signal,torch.Tensor):
            self.process_signal = self.previous_process_signal + self.diff_function(actuator_signal.item(),self.previous_process_signal,disturbance)
        else:
            self.process_signal = self.previous_process_signal + self.diff_function(actuator_signal,self.previous_process_signal,disturbance)
        reward = self.reward_function()
        
        self.actuator_signals[self.timestep] = actuator_signal
        self.process_signals[self.timestep] = self.process_signal
        self.setpoints[self.timestep] = setpoint
        self.disturbances[self.timestep] = disturbance

        self.rewards[self.timestep] = reward
        
        self.timestep += 1
        return self.get_state(),reward