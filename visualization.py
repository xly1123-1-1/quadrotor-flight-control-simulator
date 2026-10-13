import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dynamics import DroneState, DroneParameters, rotation_matrix, step
from actuators import MotorParams, motor_step, motor_forces_and_torques
from controller import controller_commands, position_controller_commands, euler_angles


def simulate_maneuver():
    # roll 10 degrees for 2 s, then return to the original position
    drone = DroneParameters(mass=1.0, inertia=np.diag([0.008, 0.008, 0.016]))
    motors = MotorParams()
    state = DroneState(
        position=np.array([0., 0., 1.]),
        velocity=np.zeros(3),
        orientation=np.array([1., 0., 0., 0.]),
        angular_velocity=np.zeros(3),
    )
    dt = 0.01
    hover_speed = np.sqrt(drone.mass * drone.gravity / (4 * motors.thrust_coeff))
    actual_speeds = np.full(4, hover_speed)

    times = [0.]
    positions = [state.position.copy()]
    orientations = [state.orientation.copy()]
    rolls = [0.]

    for i in range(1200):
        # 2 seconds, tilt 10 degrees, intentionally causing sideways drift
        if i < 200:
            commands = controller_commands(
                state, drone, motors,
                target_angles=np.deg2rad([10., 0., 0.]),
                target_altitude=1.0,
            )
        else:
            # then hold position at (0,0,1) tilt as needed to brake and return
            commands = position_controller_commands(
                state, drone, motors,
                target_position=np.array([0., 0., 1.]),
                target_yaw=0.0,
            )

        actual_speeds = motor_step(actual_speeds, commands, dt, motors)
        force, torque = motor_forces_and_torques(actual_speeds, motors)
        state = step(state, force, torque, np.zeros(3), drone, dt)

        times.append((i + 1) * dt)
        positions.append(state.position.copy())
        orientations.append(state.orientation.copy())
        rolls.append(np.rad2deg(euler_angles(state.orientation)[0]))

    return (
        np.array(times),
        np.array(positions),
        np.array(orientations),
        np.array(rolls),
        motors.arm_length,
    )


def main():
    times, positions, quaternions, rolls, arm_length = simulate_maneuver()

    print("After 2 s: roll", round(rolls[200], 3),
          "deg; position", np.round(positions[200], 3))
    print("After 4 s: position", np.round(positions[400], 3))
    print("After 12 s: roll", round(rolls[1200], 3),
          "deg; position", np.round(positions[1200], 3))

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_y = fig.add_subplot(222)
    ax_roll = fig.add_subplot(224)
    fig.suptitle("10 degree roll, followed by horizontal position recovery")

    a = arm_length / np.sqrt(2.0)
    rotor_body = np.array([
        [ a,  a, 0.],
        [ a, -a, 0.],
        [-a, -a, 0.],
        [-a,  a, 0.],
    ])
    nose_body = np.array([arm_length * 1.35, 0., 0.])

    arm_1, = ax3d.plot([], [], [], linewidth=3)
    arm_2, = ax3d.plot([], [], [], linewidth=3)
    rotors, = ax3d.plot([], [], [], "o", markersize=7)
    nose, = ax3d.plot([], [], [], linewidth=3)
    trace, = ax3d.plot([], [], [], linewidth=1, alpha=0.7)
    ax3d.scatter([0.], [0.], [1.], marker="x", s=60, label="Target position")
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_xlim(-1., 1.)
    ax3d.set_ylim(min(-1., np.min(positions[:, 1]) - 0.5), 1.)
    ax3d.set_zlim(0., 2.)
    ax3d.set_box_aspect((2, 6.5, 2))
    ax3d.view_init(elev=22, azim=-52)

    ax_y.set_title("Horizontal position recovery")
    ax_y.set_xlabel("Time (s)")
    ax_y.set_ylabel("Y position (m)")
    ax_y.set_xlim(0, 12)
    ax_y.set_ylim(min(positions[:, 1]) - 0.5, 1.)
    ax_y.axhline(0, linestyle="--", linewidth=1, label="Target Y = 0")
    ax_y.axvline(2., linestyle=":", linewidth=1)
    y_line, = ax_y.plot([], [], linewidth=2, label="Actual Y")
    ax_y.legend(loc="lower right", fontsize=8)

    ax_roll.set_title("Roll: initial command, then position corrections")
    ax_roll.set_xlabel("Time (s)")
    ax_roll.set_ylabel("Roll (degrees)")
    ax_roll.set_xlim(0, 12)
    ax_roll.set_ylim(min(-20, np.min(rolls)-2), max(14, np.max(rolls)+2))
    ax_roll.plot([0, 2], [10, 10], "--", label="10° initial roll command")
    ax_roll.axvline(2., linestyle=":", linewidth=1, label="Position hold begins")
    roll_line, = ax_roll.plot([], [], linewidth=2, label="Actual roll")
    ax_roll.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

    def set_line(line, points):
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def update(i):
        pos = positions[i]
        R = rotation_matrix(quaternions[i])
        world_rotors = (R @ rotor_body.T).T + pos
        world_nose = pos + R @ nose_body
        set_line(arm_1, world_rotors[[0, 2]])
        set_line(arm_2, world_rotors[[1, 3]])
        set_line(rotors, world_rotors)
        set_line(nose, np.vstack([pos, world_nose]))
        set_line(trace, positions[:i + 1])
        ax3d.set_title(f"Time: {times[i]:.2f} s | Y: {pos[1]:.2f} m")
        y_line.set_data(times[:i + 1], positions[:i + 1, 1])
        roll_line.set_data(times[:i + 1], rolls[:i + 1])
        return arm_1, arm_2, rotors, nose, trace, y_line, roll_line

    animation = FuncAnimation(
        fig, update, frames=range(0, len(times), 4),
        interval=40, blit=False, repeat=False,
    )
    plt.show()
    return animation


if __name__ == "__main__":
    main()
