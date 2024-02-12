"""
    File contains the generic evaluation procedure
"""
from agent import Agent
from diff_functions import TANK_DIFF_FUNCTION
from process import Process_SISO

def run_eval_episode(
                        process,
                        agent,
                        init_actuator_signal,
                        init_process_signal,
                        disturbance_evolution,
                        setpoint_evolution,
                        episode_size):
    
    """ runs one evaluation episode for a given process and controller

    Args:
        process (Process): the process object representing the actual process to be used
        agent (Agent): the Agent or Controller object representing the actual agent to be used
        init_actuator_signal (float): the initial value of the actuator signal when starting the episode
        init_process_signal (float): the initial value of the process signal when starting the episode
        disturbance_evolution (function): the function describing the disturbance signal evolution per timestep
        setpoint_evolution (function): the function describing the setpoint signal evolution per timestep
        episode_size (int): number of timesteps in the episode

    Returns:
        [int,int]: the cumulative reward obtained during the eval episode, and the last timstep (latter is not used anywhere)
    """
    
    cumulative_reward = 0.0
    timestep = 0
    non_terminate = True
    state = process.initialize(init_actuator_signal,init_process_signal,disturbance_evolution(0),setpoint_evolution(0))
    
    episode = 1
    while non_terminate:
        action = agent.select_action(state,episode,greedy=True)
        next_state,reward = process.step(action,disturbance_evolution(timestep),setpoint_evolution(timestep))
        
        cumulative_reward += reward
        state = next_state
        timestep += 1
        #ic(timestep, state)
        if episode_size <= timestep:
            non_terminate = False
    
    return cumulative_reward, timestep

