"""
    File contains the differential equation generator for both scenarios
    each generated differential equations must take as parameters:
    - the actuator signal
    - the actual process signal that needs to be controlled
    - the disturbance signal
"""

import math

def tank_control_diff_function(diam_tank,control_valve_diam,dist_valve_diam,dist_pressure,static_p_diff=0,rho_liquid=1000):
    """creates the differential equation for the tank scenario

    Args:
        diam_tank (float): the diameter of the tank
        control_valve_diam (float): the diameter of the control valve opening
        dist_valve_diam (float): the diameter of the disturbance valve opening
        dist_pressure (float): the static pressure of the incoming liquid before the disturbance valve
        static_p_diff (int, optional): the static pressure difference between tanktop and after control valve. Defaults to 0.
        rho_liquid (int, optional): the mass density of the liquid. Defaults to 1000.

    Returns:
        [function]: the differential equation for the tank scenario
    """
    control_valve_area = math.pi * (control_valve_diam**2) / 4
    dist_valve_area = math.pi * (dist_valve_diam**2) / 4
    radius = diam_tank/2

    def diff_function(control_signal,process_signal,disturbance_signal):
        """calculates the change in tank level due to control signal from the agent/controller
           and the disturbance signal hence the position of the input valve

        Args:
            control_signal (Float): the actual control valve position as given by controller/agent
            process_signal (Float): the actual process process_signal value thus the height of the tank
            disturbance_signal (Float): the actual disturbance signal hence the input valve position

        Returns:
            [Float]: the change in tank level
        """
        assert control_signal >= 0 and control_signal <= 100
        assert disturbance_signal >= 0 and disturbance_signal <= 100

        dist_valve_opening = dist_valve_area * disturbance_signal/100
        dist_speed = math.sqrt(dist_pressure * 2 / rho_liquid)
        dist_flow = dist_valve_opening * dist_speed

        control_valve_opening = control_valve_area * control_signal/100
        control_speed = math.sqrt((2*9.81*process_signal)+(2*static_p_diff/rho_liquid))
        control_flow = control_valve_opening * control_speed

        return (dist_flow - control_flow) / (math.pi*(radius**2))
    return diff_function



def heater_control_diff_function(theta_t,heather_gain):
    """creates the differential equation for the heater scenario

    Args:
        theta_t (float): the time constant for the air heater
        heather_gain (float): the output/input gain
    """
    
    def diff_function(control_signal,process_signal,disturbance_signal):
        """calculates the change in air temperature due to the heater control signal and the disturbance signal

        Args:
            control_signal (Float): the actual voltage over the heater as given by controller/agent
            process_signal (Float): the actual process process_signal value thus the outgoing air temperature
            disturbance_signal (Float): the actual disturbance signal hence the input air temperature

        Returns:
            [Float]: the change in air temperature
        """
        assert control_signal >= 0 and control_signal <= 100
        return (1/theta_t) * (-process_signal + heather_gain*control_signal + disturbance_signal) 
    return diff_function


# This is the actual diff equation for the tank scenario as used in the report
TANK_DIFF_FUNCTION = tank_control_diff_function( diam_tank = 2,
                                            control_valve_diam = 0.2,
                                            dist_valve_diam = 0.2,
                                            dist_pressure = 10000,
                                            static_p_diff=0,
                                            rho_liquid=1000)

# This is the actual diff equation for the air heater scenario as used in the report
HEATER_DIFF_FUNCTION = heater_control_diff_function(theta_t= 22,
                                                heather_gain= 0.2)