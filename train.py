"""
    File contains the general training procedure
"""
import numpy as np
import random
import torch
import datetime

from setpoint import static_function

def run_train_episode(
                        process,
                        agent,
                        episode,
                        init_actuator_signal,
                        init_process_signal,
                        init_disturbance_signal,
                        setpoint_evolution,
                        episode_size):
    
    """procedure to run one training episode

    Args:
        process ([Process]): the process object to use
        agent ([Agent]): the agent object to train
        episode ([int]): number of the episode
        init_actuator_signal ([float]): the intial actuator signal value at start of episode
        init_process_signal ([float]): the intial process signal value at start of episode
        init_disturbance_signal ([float]): the intial disturbance signal value at start of episode
        setpoint_evolution ([function]): the function that describes the setpoint evolution
        episode_size ([int]): number of timesteps in the episode

    Returns:
        [Float,int]: returns the cumulative reward and the duration of the episode
    """
    cumulative_reward = 0.0
    timestep = 0
    non_terminate = True
    state = process.initialize(init_actuator_signal,init_process_signal,init_disturbance_signal,setpoint_evolution(0))

    while non_terminate:
    
        action = agent.select_action(state,episode,greedy=False)
        next_state,reward = process.step(action,init_disturbance_signal,setpoint_evolution(timestep))
        agent.update_networks(state,action,reward,next_state)
        
        # Update statistics
        cumulative_reward += reward
        state = next_state
        timestep += 1

        if episode_size <= timestep:
            non_terminate = False
    
    return cumulative_reward, timestep
  


def run_trial(process,
              agent_list,
              episodes,
              episode_size,
              episode_print=None,
              logging=True,
              save_agents=False):
    """runs a training trial of multiple episodes on one or more agents

    Args:
        process ([Process]): the process object to use
        agent_list ([list[Agent]]): the list of agent objects to train
        episodes ([int]): total number of episodes to train
        episode_size ([int]): number of timesteps in the episode
        episode_print (int, optional): interval of episodes to print results. Defaults to None.
        logging (bool, optional): if true the results will be logged in the agnet object. Defaults to True.
        save_agents (bool, optional): if true the agent will be saved after the trial. Defaults to False..
    """
    # set seeds
    torch.manual_seed(0)
    random.seed(0)	
    np.random.seed(0)

    # perform the trial for each agent in the agent list
    for index, agent in enumerate(agent_list):
    
        # if logging is requested: initialize logs
        if logging:
            agent.init_logging(episodes)
            avg_cumul_reward = 0.0
        
        # print the results of the episode
        if episode_print is not None:
            print("\n" + "-" *90)
            print(agent)
            print("-" * 90 + "\n")
            print(  f"\n{'epi.':>10}{'duration':>10}{'init_output':>14}{'init_dist':>14}{'setp.':>10}" +
                    f"{'reward':>10}{'avg rew.':>10}{'t-steps':>10}{'mem':>10}\n")
            
        
        # Loop over episodes
        for episode in range(episodes): 
            begin_time = datetime.datetime.now()

            # choose random input
            setpoint = static_function(random.uniform(process.setpoint_min, process.setpoint_max))
          
            init_actuator_signal = random.uniform(process.actuator_signal_min, process.actuator_signal_max)
            init_process_signal = random.uniform(process.setpoint_min, process.setpoint_max)
            init_disturbance_signal = random.uniform(process.disturbance_signal_min, process.disturbance_signal_max)
            
            cumul_reward, timesteps= run_train_episode( process,
                                                        agent,
                                                        episode,
                                                        init_actuator_signal = init_actuator_signal,
                                                        init_process_signal = init_process_signal,
                                                        init_disturbance_signal = init_disturbance_signal,
                                                        setpoint_evolution = setpoint,
                                                        episode_size = episode_size)
                                                         
            # Per-episode statistics
            avg_cumul_reward *= 0.95
            avg_cumul_reward += 0.05 * cumul_reward

            # log the results in the agent
            if logging:
                agent.log(episode,cumul_reward,avg_cumul_reward)
            
            #print the results per episode if requested
            if episode_print is not None and episode % episode_print==0:
                duration = datetime.datetime.now() - begin_time
                print(f"{episode:>10}{round(duration.total_seconds(),0):>10}{round(init_process_signal,2):>14}" + 
                    f"{round(init_disturbance_signal,2):>14}{round(setpoint(0),2):>10}{round(cumul_reward,2):>10}" + 
                    f"{round(avg_cumul_reward,2):>10}{timesteps:>10}{agent.repl_mem.samples_in_memory:>10}")
                
        
        # if requested, save the agent
        if save_agents:
            agent.save()