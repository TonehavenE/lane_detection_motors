from pymavlink import mavutil
import sys
import signal
from pid import PID
import numpy as np

def press_to_depth(pressure):
    """Convert pressure to depth
    Args:
        pressure (float): Pressure in hPa
    Returns:
        float: Depth in water in meters
    """
    rho = 1029  # density of fresh water in kg/m^3
    g = 9.81  # gravity in m/s^2
    pressure_at_sea_level = 1013.25  # pressure at sea level in hPa
    # multiply by 100 to convert hPa to Pa
    return (pressure - pressure_at_sea_level) * 100 / (rho * g)

def get_depth_error(mav, desired_depth=0.5):
     # get pressure from the vehicle
    msg = mav.recv_match(type="SCALED_PRESSURE2", blocking=True)
    press_abs = msg.press_abs  # in hPa

    # calculate depth
    current_depth = press_to_depth(press_abs)

    # calculate error
    error = desired_depth - current_depth
    return error