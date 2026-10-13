import numpy as np
from dataclasses import dataclass

@dataclass
class MotorParams:
    arm_length: float = 0.2
    thrust_coeff: float = 1.8e-5 
    yaw_torque_coeff: float = 2.5e-7
    motor_time_const: float = 0.05
    max_speed: float = 700. # 700 rad/s

#motor order:
# 0 = front_left (FL), 1 = front_right (FR), 2 = rear_right (RR), 3 = rear_left (RL) - imagine looking top-down fiew at a quadrotor with FL top left.

def motor_step(current_speeds, commanded_speeds, dt, p):
    #update motor speeds using a first-order response model.

    if dt <= 0 or p.motor_time_const <= 0 or p.max_speed <= 0:
        raise ValueError("Time step, time constant and max speed must be positive")

    current = np.asarray(current_speeds, dtype=float)
    commanded = np.asarray(commanded_speeds, dtype=float)

    if current.shape != (4,) or commanded.shape != (4,):
        raise ValueError("Expected four motor speeds")

    target = np.clip(commanded, 0.0, p.max_speed) # commanded speed set to range 0-700rad/s

    response = 1.0 - np.exp(-dt / p.motor_time_const) # 0.05 is the motor time constant

    new_speeds = current + response * (target - current)

    return np.clip(new_speeds, 0.0, p.max_speed)


def motor_forces_and_torques(speeds, p):
    # Convert four rotor speeds into body force and torque

    speeds = np.asarray(speeds, dtype=float)

    if speeds.shape != (4,):
        raise ValueError("Expected four motor speeds")

    if np.any(speeds < 0) or np.any(speeds > p.max_speed):
        raise ValueError("Motor speeds outside allowed range")

    # Each rotor's upward thrust F=kT*w^2
    thrusts = p.thrust_coeff * speeds**2

    # X-configuration geometry; each motor sits along a 0.20m arm, 45 degree angle from the body axes
    a = p.arm_length / np.sqrt(2.0) 

    x = np.array([a, a, -a, -a])
    y = np.array([a, -a, -a, a])

    # reaction-torque signs of alternating rotors
    yaw_signs = np.array([-1.0, 1.0, -1.0, 1.0])

    # Net upward force in the body frame
    force_body = np.array([
        0.0,
        0.0,
        np.sum(thrusts)
    ]) # all rotor thrusts are along body +Z, if the drone is tilted, dynamics module transforms the force into world coordinates using quaternion

    # Torque from r cross F plus the rotor reaction torque
    torque_body = np.array([
        np.dot(y, thrusts),
        -np.dot(x, thrusts),
        p.yaw_torque_coeff * np.dot(yaw_signs, speeds**2)
    ])

    return force_body, torque_body


# Test :)

if __name__ == "__main__":

    p = MotorParams()

    # Speed required for four motors to support a 1 kg drone
    hover_speed = np.sqrt(
        9.81 / (4 * p.thrust_coeff)
    )

    speeds = np.full(4, hover_speed)

    force, torque = motor_forces_and_torques(speeds, p)

    print("Hover motor speed:", round(hover_speed, 2), "rad/s")
    print("Total body force:", np.round(force, 4))
    print("Total body torque:", np.round(torque, 4))