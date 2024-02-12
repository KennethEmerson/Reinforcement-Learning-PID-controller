"""
    file contains the different functions for evolving setpoints and disturbances
    note: the noise generator used in the evaluation can be found in "noise.py"
"""
import math
import random

def static_function(initial_value):
    """creates a function which will keep a value a predefined value during all timesteps

    Args:
        initial_value ([function]): function that keeps a value a predefined value during all timesteps
    """
    def function(timestep):
        return initial_value
    return function


#def sinus_function(mean,amplitude,period):

 #   def function(timestep):
 #       return mean + amplitude*math.sin(timestep * 2 * math.pi/ (period))
 #   return function


def step_function(initial_value,final_value,trigger_timestep):
    """ creates a function which will create a step function at a given trigger timestep

    Args:
        initial_value ([Float]): initial value before the trigger
        final_value ([Float]): value after the trigger
        trigger_timestep ([int]): the timestep for which to trigger the step function
    """
    def function(timestep):
        setpoint = initial_value
        if timestep >= trigger_timestep:
            setpoint = final_value
        return setpoint
    return function

#def block_function(min_value,max_value,period):
#    def function(timestep):
#        trigger = math.sin(timestep / (period * 2 * math.pi))
#        return min_value if trigger <= 0 else max_value
#    return function

def ramp_function(min_value,max_value,increment,trigger_timestep):
    """creates a function which will create a increasing ramp function from a given trigger timestep

    Args:
        min_value ([float]): minimum value of the ramp
        max_value ([float]): maximum value of the ramp
        increment ([float]): speed of the ramp
        trigger_timestep ([int]): the timestep for which to trigger the ramp function
    """
    def function(timestep):
        if timestep >= trigger_timestep:
            output = min_value + ((timestep-trigger_timestep)*increment)
            output = output if output <= max_value else max_value
        else: 
            output = min_value
        return output
    return function