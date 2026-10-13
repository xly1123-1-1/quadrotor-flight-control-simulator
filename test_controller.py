import numpy as np
from dynamics import DroneState, DroneParameters, step
from actuators import MotorParams, motor_step, motor_forces_and_torques
from controller import controller_commands, euler_angles

drone = DroneParameters(
    mass=1.0,
    inertia=np.diag([0.008, 0.008, 0.016])
)

motors = MotorParams()

# starting position level, stationary, and 1 meter above ground
state = DroneState(
    position=np.array([0., 0., 1.]),
    velocity=np.zeros(3),
    orientation=np.array([1., 0., 0., 0.]),
    angular_velocity=np.zeros(3)
)

dt = 0.01

# motors at hover speed
hover_speed = np.sqrt(
    drone.mass * drone.gravity / (4 * motors.thrust_coeff)
)

actual_speeds = np.full(4, hover_speed)


# test: 4s
for i in range(400):

    # command 10 degrees of roll for 2s
    # return to level flight in 2s
    desired_roll = np.deg2rad(10.0) if i < 200 else 0.0

    # controller computes the commanded motor speeds
    commands = controller_commands(
        state,
        drone,
        motors,
        target_angles=np.array([desired_roll, 0., 0.]),
        target_altitude=1.0
    )

    actual_speeds = motor_step(
        actual_speeds, commands, dt, motors
    )

    force, torque = motor_forces_and_torques(
        actual_speeds, motors
    )

    state = step(
        state,
        force,
        torque,
        np.zeros(3),
        drone,
        dt
    ) # state advanced

    if i in (199, 399):
        angles = np.rad2deg(
            euler_angles(state.orientation)
        )

        print(f"\nAfter {(i + 1) * dt:.1f} seconds:")
        print("Roll, pitch, yaw:", np.round(angles, 3))
        print("Altitude:", round(state.position[2], 4))
        print("Position:", np.round(state.position, 3))
