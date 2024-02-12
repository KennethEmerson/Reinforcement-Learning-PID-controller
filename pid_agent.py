"""
    file contains the implementation of a PID controller. 
    Class contains identical methods as the RL agent for compatibility
"""

import torch
import numpy as np

PID_INTERNAL_MIN = -100
PID_INTERNAL_MAX = 100

class PID_Agent:
    def __init__(self, kp, ki, kd, actuator_signal_min,actuator_signal_max,invertor=False):
        """Initializes the PID agent object

        Args:
            kp (float): The proportional gain
            ki (float): The integrating gain
            kd (float): The derivative gain
            actuator_signal_min (float): the minimal actuator signal
            actuator_signal_max (float): the maximal actuator signal
            invertor (bool, optional): will invert the working of the controller (if higher input shluld decrease output). Defaults to False.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.actuator_signal_min = actuator_signal_min
        self.actuator_signal_max = actuator_signal_max
        self.invertor = invertor

        self.timestep = 0
        self.initialize()


    def initialize(self):
        """sets all parameters to initial stage
        """
        self.setpoint = 0.0
        self.last_error = 0.0
        self.int_term = 0.0


    def __str__(self):
        """gives the settings of the pid controller as a string

        Returns:
            [String]: PID controller settings
        """
        return  f"PID controller: kp: {self.kp}, ki: {self.ki}, " + \
                    f"kd: {self.kd}, anti-windup min: {self.actuator_signal_min}, anti-windup min: {self.actuator_signal_max}"


    def init_logging(self,samples):
        """initializes the logging for the training (is not actually used)

        Args:
            samples (int): number of samples and thus space to provide for logging
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

    
    def select_action(self, state,episode=1,greedy=True):
        """selects action given the actual state of the process (PID controller only uses actual process value)

        Args:
            state ([Tensor/float]): if Tensor is assumed to be process state, if float is assumed to be process actuator_signal value
            episode (int, optional): [this value is not used. only present to be compatible with RL-agent
            greedy (bool, optional): this value is not used. only present to be compatible with RL-agent

        Returns:
            [float]: action value selected by the PID controller
        """
        if isinstance(state,torch.Tensor):
            error = state[2].item()
        else:
            error = self.setpoint - state
        #ic(error)
        prop_term = self.kp * error
        self.int_term += error 

        #integral anti windup
        if (self.int_term < PID_INTERNAL_MIN):
            self.int_term = PID_INTERNAL_MIN
        elif (self.int_term > PID_INTERNAL_MAX):
            self.int_term = PID_INTERNAL_MAX

        diff_term = error - self.last_error 
        self.last_error = error
        actuator_signal =  prop_term + (self.ki * self.int_term) + (self.kd * diff_term)
        
        # inverts signal if required
        if self.invertor:
            actuator_signal = -actuator_signal
        
        #integral anti windup
        if (actuator_signal < PID_INTERNAL_MIN):
            actuator_signal = PID_INTERNAL_MIN
        elif (actuator_signal > PID_INTERNAL_MAX):
            actuator_signal = PID_INTERNAL_MAX

        # create offset for actuator_signal
        internal_range = PID_INTERNAL_MAX - PID_INTERNAL_MIN
        actuator_signal_range = self.actuator_signal_max-self.actuator_signal_min
        actuator_signal = (actuator_signal_range * (actuator_signal - PID_INTERNAL_MIN ) / internal_range) + self.actuator_signal_min
        return actuator_signal