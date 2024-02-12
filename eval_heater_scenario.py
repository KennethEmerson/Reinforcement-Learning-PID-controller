"""
    File contains the main evaluation of the agents for the heater process
"""

from agent import Agent
from pid_agent import PID_Agent
from diff_functions import HEATER_DIFF_FUNCTION
from process import Process_SISO
from eval import run_eval_episode

from plots import plot_process_episode, plot_rewards
from setpoint import static_function, step_function, ramp_function
from noise import OUNoise

# set the correct differential equation
DIFF_FUNCTION = HEATER_DIFF_FUNCTION

# set the constants for the process
EPISODE_SIZE = 1000
SETPOINT = 15.
SETPOINT_MIN = 0
SETPOINT_MAX = 40
PROCESS_SIGNAL_INIT = 10.
PROCESS_SIGNAL_MIN = 0
PROCESS_SIGNAL_MAX = 40
ACTUATOR_SIGNAL_INIT = 0.
ACTUATOR_SIGNAL_MIN = 0
ACTUATOR_SIGNAL_MAX = 100
DISTURBANCE_SIGNAL= 10.
DISTURBANCE_SIGNAL_MIN = 0
DISTURBANCE_SIGNAL_MAX = 40
SCENARIO_NAME = 'heater'

################################################################

# create Ornstein–Uhlenbeck noise generator for the setpoint
noise_generator = OUNoise(mu=15,theta=0.01,sigma=0.1,decay=None)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# select one of the setpoint evolutions below:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SETPOINT_EVOLUTION = static_function(SETPOINT)
#SETPOINT_EVOLUTION = step_function(SETPOINT,SETPOINT+10,300)
#SETPOINT_EVOLUTION = ramp_function(15.,25.,0.05,200)
#SETPOINT_EVOLUTION = noise_generator.non_decaying_noise

################################################################

# create Ornstein–Uhlenbeck noise generator for the disturbance
noise_generator = OUNoise(mu=12,theta=0.01,sigma=0.1,decay=None)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# select one of the disturbance evolutions below:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

DISTURBANCE_EVOLUTION = static_function(DISTURBANCE_SIGNAL)
#DISTURBANCE_EVOLUTION = step_function(DISTURBANCE_SIGNAL,DISTURBANCE_SIGNAL+4,700)
#DISTURBANCE_EVOLUTION = ramp_function(12,20,0.01,700)
#DISTURBANCE_EVOLUTION = noise_generator.non_decaying_noise

################################################################

# set the PID agent
pid_agent = PID_Agent(  kp = 0.9,
                        ki = 0.8,
                        kd = 0.,
                        actuator_signal_min = ACTUATOR_SIGNAL_MIN,
                        actuator_signal_max = ACTUATOR_SIGNAL_MAX,
                        invertor = False)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# select the correct agent below:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#agent = pid_agent 
agent = Agent.load('saved_agents/heater_statedim3_actiondim1_mem80000_batch128_nmu0.0_nsigma10_g0.99_LRA0.0001_LRC0.0001_TNupd1e-05_optimL0.0001_20210829 185119.pt')


################################################################

# set the process parameters
process = Process_SISO( "heater",
                        DIFF_FUNCTION,
                        setpoint_min = SETPOINT_MIN,
                        setpoint_max = SETPOINT_MAX,
                        process_signal_min = PROCESS_SIGNAL_MIN,
                        process_signal_max = PROCESS_SIGNAL_MAX,
                        actuator_signal_min= ACTUATOR_SIGNAL_MIN,
                        actuator_signal_max= ACTUATOR_SIGNAL_MAX,
                        disturbance_signal_min= DISTURBANCE_SIGNAL_MIN,
                        disturbance_signal_max= DISTURBANCE_SIGNAL_MAX,
                        episode_size = EPISODE_SIZE, 
                        reward_function = "monotonic_improved")

################################################################

# start evaluation and plot results
cumulative_reward, timestep = run_eval_episode( process,
                                                agent,
                                                init_actuator_signal = ACTUATOR_SIGNAL_INIT,
                                                init_process_signal = PROCESS_SIGNAL_INIT,
                                                disturbance_evolution= DISTURBANCE_EVOLUTION,
                                                setpoint_evolution = SETPOINT_EVOLUTION,
                                                episode_size = EPISODE_SIZE)

timesteps, actuator_signals, process_signals, setpoints, rewards, disturbances = process.get_episode()
plot_process_episode(   SCENARIO_NAME,
                        timesteps,
                        PROCESS_SIGNAL_MIN,
                        PROCESS_SIGNAL_MAX,
                        actuator_signals,
                        process_signals,
                        setpoints,
                        disturbances)