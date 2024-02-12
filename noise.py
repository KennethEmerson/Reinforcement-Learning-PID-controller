"""
    File contains the noise generator as used by the agent
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def noise_decay(epsilon_min,epsilon_max,epsilon_decay,episode):
    """function to support the decay of the noise functions

    Args:
        epsilon_min (float): the minimum epsilon value 
        epsilon_max (float): the maximale epsilon value
        epsilon_decay (int): parameter that controls the ratio of decay (higher is slowe decay)
        episode (int): the actual episode for which the decay is calculated

    Returns:
        [float]: the actual value after considering the decay
    """
    return epsilon_min + (epsilon_max - epsilon_min) * \
                         math.exp(-1. * episode / epsilon_decay) 


class OUNoise:
    def __init__(self,mu=0, theta=0.15, sigma=0.3,decay=None):
        """initializes the Ornstein–Uhlenbeck noise generator

        Args:
            mu (int, optional): the mu value for the generator. Defaults to 0.
            theta (float, optional): the theta value for the generator. Defaults to 0.15.
            sigma (float, optional): the sigma value for the generator. Defaults to 0.3.
            decay ([type], optional): the decay ratio. Defaults to None.
        """

        self.name = "OUNoise"
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu
        self.decay = decay
        if self.decay is None:
            self.noise = self.non_decaying_noise
        else:
            self.noise = self.decaying_noise
        self.reset()


    def reset(self):
        """rests the generator to its initial state
        """
        self.state = self.mu


    def non_decaying_noise(self,episode=None):
        """calculates the noise value for that given state without decay

        Args:
            episode (int, optional): the actual episode for which to calculate the noise. Defaults to None.

        Returns:
            [float]: random noise value generated conform the Ornstein–Uhlenbeck process
        """
        self.state += self.theta * (self.mu - self.state) + self.sigma * random.gauss(0,1)
        return self.state


class GaussNoise:
    def __init__(self,mu=0,sigma=0.3,decay=None):
        """initializes the Gaussian noise generator

        Args:
            mu (int, optional): the mu value for the generator. Defaults to 0.
            sigma (float, optional): the sigma value for the generator. Defaults to 0.3.
            decay ([type], optional): the decay ratio. Defaults to None.
        """
        self.name = "GaussNoise"
        self.mu = mu
        self.sigma = sigma
        self.decay = decay
        if self.decay is None:
            self.noise = self.non_decaying_noise
        else:
            self.noise = self.decaying_noise
            

    def non_decaying_noise(self,episode=None):
        """calculates the noise value for that given state without decay

        Args:
            episode (int, optional): the actual episode for which to calculate the noise. Defaults to None.

        Returns:
            [float]: random noise value generated conform the Gaussian process
        """
        return random.gauss(self.mu,self.sigma)


    def decaying_noise(self,episode):
        """calculates the noise value for that given state with decay

        Args:
            episode (int, optional): the actual episode for which to calculate the noise.

        Returns:
            [float]: random noise value generated conform the Gaussian process
        """
        sigma = noise_decay(0,self.sigma,self.decay,episode)
        
        return random.gauss(self.mu,sigma)
    


if __name__ == '__main__':
    ou1 = OUNoise(mu=0,theta=0.1,sigma=3)
    ou2 = OUNoise(mu=0,theta=0.1,sigma=3,decay=500)
    states1 = []
    states2 = []
    for i in range(1000):
        states1.append(ou1.noise(i))
        states2.append(ou2.noise(i))
    
    plt.figure(figsize=(5,5))
    plt.plot(states1,label="Non decaying OUNoise: mu=0 theta=0.1 sigma=0.1",alpha=0.4,color="black")
    plt.plot(states2,label="Decaying OUNoise: mu=0 theta=0.1 sigma=0.1 decay=200",alpha=1,color="black")
    plt.legend()
    plt.show()
    
    MU = 0
    SIGMA = 10
    DECAY = 350

    ou1 = GaussNoise(mu=MU,sigma=SIGMA)
    ou2 = GaussNoise(mu=MU,sigma=SIGMA,decay=DECAY)
    states1 = []
    states2 = []
    for i in range(1000):
        states1.append(ou1.noise(i))
        states2.append(ou2.noise(i))
    
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.plot(states1,label=f"Non decaying Gaussian noise: mu: {MU} sigma: {SIGMA}")
    plt.plot(states2,label=f"Decaying Gaussian noise: mu: {MU} sigma: {SIGMA} decay={DECAY}")
    plt.legend()
    
    plt.show()