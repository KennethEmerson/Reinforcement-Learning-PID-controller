"""
    File contains the main training of the agents for the tank level process
"""

from diff_functions import tank_control_diff_function
from process import Process_SISO
from agent import Agent
from plots import plot_comparison
from train import run_trial


# set constants for the differential equation
TANK_DIFF_FUNCTION = tank_control_diff_function( diam_tank = 2,
                                            control_valve_diam = 0.2,
                                            dist_valve_diam = 0.2,
                                            dist_pressure = 10000,
                                            static_p_diff=0,
                                            rho_liquid=1000)

# set constants for the training
DIFF_FUNCTION = TANK_DIFF_FUNCTION
EPISODES = 800
EPISODE_SIZE = 600
SETPOINT_MIN = 1
SETPOINT_MAX = 4
ACTUATOR_SIGNAL_MIN = 0
ACTUATOR_SIGNAL_MAX = 100
DISTURBANCE_SIGNAL_MIN = 25
DISTURBANCE_SIGNAL_MAX = 75
PROCESS_SIGNAL_MIN = 0
PROCESS_SIGNAL_MAX = 5

SAVE_AGENTS = False 

# define the scenario name
SCENARIO_NAME = f"TANK SCENARIO, ep_size:{EPISODE_SIZE}, r_min: {SETPOINT_MIN}, r_max: {SETPOINT_MAX}, " + \
                f"u_min: {ACTUATOR_SIGNAL_MIN}, u_max: {ACTUATOR_SIGNAL_MAX}, d_min: {DISTURBANCE_SIGNAL_MIN}, " + \
                f"d_max: {DISTURBANCE_SIGNAL_MAX}"


# define the process
process = Process_SISO( "TANK",
                        DIFF_FUNCTION,
                        setpoint_min = SETPOINT_MIN,
                        setpoint_max = SETPOINT_MAX,
                        process_signal_min = PROCESS_SIGNAL_MIN,
                        process_signal_max = PROCESS_SIGNAL_MAX,
                        actuator_signal_min = ACTUATOR_SIGNAL_MIN,
                        actuator_signal_max = ACTUATOR_SIGNAL_MAX,
                        disturbance_signal_min = DISTURBANCE_SIGNAL_MIN,
                        disturbance_signal_max = DISTURBANCE_SIGNAL_MAX,
                        episode_size = EPISODE_SIZE,
                        reward_function = "monotonic_improved")


# define the agent
agent1 = Agent(state_dim = process.get_state_size(),
                action_dim = 1,
                action_min = ACTUATOR_SIGNAL_MIN,
                action_max = ACTUATOR_SIGNAL_MAX,
                repl_mem_size = 80000,
                repl_batch_size=128,
                noise_gen = "GaussNoise",
                noise_mu = 0.,
                noise_sigma = 10,
                noise_decay = 350,
                gamma = 0.99,
                actor_learning_rate = 0.0001,
                critic_learning_rate = 0.0001,
                target_update_rate = 0.00001,
                optimizer_L2 = 0.0001
            )



## run trial
agent_list = [agent1]

run_trial(process,agent_list,
            episodes = EPISODES,
            episode_size = EPISODE_SIZE,
            episode_print=1,
            logging=True,
            save_agents=SAVE_AGENTS)
            
#plot_rewards(agent)
plot_comparison(SCENARIO_NAME,agent_list,save_plot=True)