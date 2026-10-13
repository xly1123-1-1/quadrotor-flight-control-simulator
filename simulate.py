
import numpy as np

from dynamics import DroneState, DroneParameters, step
from actuators import MotorParams, motor_step, motor_forces_and_torques

drone = DroneParameters(
    mass=1.0,
    inertia=np.diag([0.008, 0.008, 0.016])
)

motors = MotorParams()

state = DroneState(
    position=np.array([0., 0., 1.]),
    velocity=np.zeros(3),
    orientation=np.array([1., 0., 0., 0.]),
    angular_velocity=np.zeros(3)
) # initial state

dt = 0.01

hover_speed = np.sqrt(
    drone.mass * drone.gravity / (4 * motors.thrust_coeff)
)

# motors already spinning at hover speed
actual_speeds = np.full(4, hover_speed)

def advance(state, actual_speeds, commands, steps):
    for _ in range(steps):
        # Update motor speeds
        actual_speeds = motor_step(
            actual_speeds, commands, dt, motors
        )

        # Convert motor speeds into forces and torques
        force, torque = motor_forces_and_torques(
            actual_speeds, motors
        )

        # Calculate drone motion
        state = step(
            state, force, torque,
            np.zeros(3), drone, dt
        )

    return state, actual_speeds


# test 1: hover for 1s
hover_commands = np.full(4, hover_speed)

state, actual_speeds = advance(
    state, actual_speeds, hover_commands, 100
)

print("After hover, position:", np.round(state.position, 4))
print("After hover, angular velocity:", np.round(state.angular_velocity, 4))

# test 2: increase left motor speeds to produce roll
roll_commands = np.array([
    hover_speed + 20,  # FL
    hover_speed - 20,  # FR
    hover_speed - 20,  # RR
    hover_speed + 20   # RL
])

state, actual_speeds = advance(
    state, actual_speeds, roll_commands, 20
)

print("After roll command, angular velocity:",
      np.round(state.angular_velocity, 4))
print("After roll command, orientation:",
      np.round(state.orientation, 4))