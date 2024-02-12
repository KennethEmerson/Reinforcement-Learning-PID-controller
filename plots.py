"""
 contains aux functions for plotting the results   
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

markers = [".","o","v","^","<",">","s","p","*"]
colors = ["black","red","blue","green"]

def plot_process_episode(   plotname,
                            timesteps,
                            process_min,
                            process_max, 
                            actuator_signal,
                            process_signal,
                            setpoints,
                            disturbance_signal=None):
    """plots the process results during one episode

    Args:
        plotname ([String]): name of the plot
        timesteps ([int]): number of timesteps in episode
        process_min ([float]): minmimum value for the process value
        process_max ([float]): maximum value for the process value
        actuator_signal ([np.array]): array of actuator signals during the episode
        process_signal ([np.array]): array of process signals during the episode
        setpoints ([np.array]): array of setpoint signals during the episode
        disturbance_signal ([np.array], optional): array of disturbance signals during the episode. Defaults to None.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,4))
    
    ax1.plot(timesteps,process_signal,label='process signal (y)',linewidth=0.7,alpha=1,color="black")
    ax1.plot(timesteps,setpoints,label='reference signal (r)',linewidth=0.7,linestyle='dashed',alpha=1,color="black")
    ax1.set_ylim(process_min, process_max)
    ax1.set_xlabel('timesteps')
    ax1.set_title("process")
    ax1.legend()

    ax2.plot(timesteps,actuator_signal,label='actuator signal (u)',linewidth=0.7,alpha=1,color="black")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('timesteps')
    ax2.set_title("actuator signal (u)")
    
    if disturbance_signal is not None: 
        ax3.plot(timesteps,disturbance_signal,label='disturbance (d)',linewidth=0.7,alpha=1,color="black")
        ax3.set_xlabel('timesteps')
        ax3.set_title("disturbance signal (d)")
        
    else:
        ax3.remove()

    if plotname == 'tank':
        ax1.set_ylabel('level (m)')
        ax2.set_ylabel('position (%)')
        ax3.set_ylabel('position (%)')
        ax3.set_ylim(0, 100)

    if plotname == 'heater':
        ax1.set_ylabel('temperature (°C)')
        ax2.set_ylabel('voltage (V)')
        ax3.set_ylabel('tempreature (°C)')
        ax3.set_ylim(process_min, process_max)

    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.9, wspace=0.2, hspace=0.1)
    plt.show()


def plot_rewards(agent):
    """plots the rewards and average reward of one agent

    Args:
        agent (Agent): the trained agent for which the log needs to be plotted
    """
    fig, ax = plt.subplots()

    episodes_x = agent.log_episodes

    plt.xlabel('episodes')
    plt.ylabel('reward')

    ax.plot(agent.log_episodes, agent.log_rewards,label="reward",linewidth=0.6,alpha=0.6) 
    ax.plot(agent.log_episodes, agent.log_avg_rewards,label="average reward",linewidth=0.6,alpha=1,c="red") 
    ax.legend()
    plt.title(agent,fontsize=10)
    plt.show()


def plot_comparison(plot_name,agent_list,save_plot=True):
    """plots the average rewards of all agents in the list

    Args:
        plot_name ([String]): name of the plot
        agent_list ([Agent]): List of agents for which to plot the average reward logging
        save_plot (bool, optional): if true the plot is saved. Defaults to True.
    """

    fig, ax = plt.subplots(figsize=(15,8))

    plt.xlabel('episodes')
    plt.ylabel('avg rewards')
    plt.title(plot_name)

    for index, agent in enumerate(agent_list):
        marker_space = len(agent.log_episodes)
        ax.plot(agent.log_episodes, agent.log_avg_rewards,label=agent.__str__(),linewidth=0.7,alpha=0.8,
                        color=colors[index]) 
    ax.legend(fontsize=8)
    
    # save plot if requested
    if save_plot:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H_%M_%S")  
        plt.savefig(f".//plots//{dt_string}.png",format="png",dpi=300)
    
    plt.show()

