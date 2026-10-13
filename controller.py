import numpy as np
from dynamics import normalize_quaternion, rotation_matrix

def euler_angles(q):
    # convert q to roll, pitch, yaw (radians).
    w, x, y, z = normalize_quaternion(q)
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.array([roll, pitch, yaw])


def angle_error(target, current):
    # shortest signed angle difference, between -pi and pi.
    return (target - current + np.pi) % (2 * np.pi) - np.pi


def mix_motor_speeds(total_thrust, torque_body, motors):
    # apply total thrust and body torques across four rotors.
    a = motors.arm_length / np.sqrt(2.0)
    x = np.array([a, a, -a, -a])
    y = np.array([a, -a, -a, a])
    yaw_signs = np.array([-1., 1., -1., 1.])

    # rows correspond to total thrust, roll, pitch, and yaw torque.
    matrix = np.vstack([
        np.ones(4),
        y,
        -x,
        yaw_signs * motors.yaw_torque_coeff / motors.thrust_coeff,
    ])
    desired = np.array([total_thrust, *torque_body])
    rotor_thrusts = np.linalg.solve(matrix, desired)

    max_thrust = motors.thrust_coeff * motors.max_speed**2
    rotor_thrusts = np.clip(rotor_thrusts, 0.0, max_thrust)
    return np.sqrt(rotor_thrusts / motors.thrust_coeff)


def controller_commands(state, drone, motors, target_angles, target_altitude):
    # inner-loop attitude PD control and altitude PD control
    error = angle_error(target_angles, euler_angles(state.orientation))
    kp = np.array([0.32, 0.32, 0.20])
    kd = np.array([0.11, 0.11, 0.12])
    torque_body = kp * error - kd * state.angular_velocity

    vertical_acc = (
        4.0 * (target_altitude - state.position[2])
        - 3.0 * state.velocity[2]
    )
    tilt_factor = np.clip(rotation_matrix(state.orientation)[2, 2], 0.30, 1.0)
    total_thrust = drone.mass * (drone.gravity + vertical_acc) / tilt_factor
    max_total_thrust = 4 * motors.thrust_coeff * motors.max_speed**2
    total_thrust = np.clip(total_thrust, 0.0, max_total_thrust)
    return mix_motor_speeds(total_thrust, torque_body, motors)


def position_controller_commands(state, drone, motors, target_position, target_yaw=0.0):
    # outer-loop horizontal position PD controller computes desired roll/pitch to accelerate toward an XYZ position target
   
    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must have three components")

    xy_error = target_position[:2] - state.position[:2]
    xy_acc = 1.4 * xy_error - 2.2 * state.velocity[:2]

    # limit horizontal acceleration to avoid large tilt 
    max_xy_acc = 2.5 
    magnitude = np.linalg.norm(xy_acc)
    if magnitude > max_xy_acc:
        xy_acc = xy_acc * (max_xy_acc / magnitude)

    # desired world acceleration along the chosen yaw heading
    c, s = np.cos(target_yaw), np.sin(target_yaw)
    forward_acc = c * xy_acc[0] + s * xy_acc[1]
    left_acc = -s * xy_acc[0] + c * xy_acc[1]

    # Tilt the body +z thrust vector toward the requested acceleration
    desired_pitch = np.arctan2(forward_acc, drone.gravity)
    desired_roll = np.arctan2(-left_acc, np.hypot(drone.gravity, forward_acc))
    max_tilt = np.deg2rad(20.0)
    desired_angles = np.array([
        np.clip(desired_roll, -max_tilt, max_tilt),
        np.clip(desired_pitch, -max_tilt, max_tilt),
        target_yaw,
    ])

    return controller_commands(
        state, drone, motors,
        target_angles=desired_angles,
        target_altitude=target_position[2],
    )
