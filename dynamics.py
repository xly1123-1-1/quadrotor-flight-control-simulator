import numpy as np
from dataclasses import dataclass 


@dataclass
class DroneState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray

@dataclass
class DroneParameters:
    mass: float
    inertia: np.ndarray
    gravity: float = 9.81
    air_density: float = 1.225
    drag_coeff: float = 1.0
    reference_area: float = 0.05
    angular_damping: float = 0

params = DroneParameters(mass=1.0, inertia=np.diag([0.008, 0.008, 0.016])) # assuming a 1kg drone with 0.20m branches at 45 degrees angle from the 0.60kg center, each rotor is 0.10kg. 
state = DroneState(position=np.array([0.,0.,1.]), velocity=np.zeros(3), orientation=np.array([1.,0.,0.,0.,]), angular_velocity=np.zeros(3)) # choice of diag
# initially, drone is stable at 1m above ground, hovering.
# orientation: 

def normalize_quaternion(q):  # q= a+bi+cj+dk
    q = np.asarray(q, dtype=float)

    mag=np.linalg.norm(q) # find |q| from RSS w, x, y, z

    if mag==0:
        raise ValueError("Orientation quaternion cannot be zero")
    
    return q/mag # returns unit quaternerion 

def rotation_matrix(q):
   # convert body-to-world q into a 3x3 rotation matrix. 

    w, x, y, z = normalize_quaternion(q) # cos(x/2), uxsin(x/2), uy(sinx/2), uzsin(x/2)

    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ]) # uses trig identities like cos(x) = 1-2sin2(x/2), giving rotation about the unit axis of our choosing and how that orients the drone with respect to world coordinate
# the physics engine will update the quaternerion using the angular velocity (next aprt)


def quaternion_derivative(q, omega_body):

    w, x, y, z = normalize_quaternion(q) # take the quaternerion
    wx, wy, wz = omega_body # angular velocity in rad/s about x,y,z axes

    return 0.5 * np.array([
        -x*wx - y*wy - z*wz,
         w*wx + y*wz - z*wy,
         w*wy + z*wx - x*wz,
         w*wz + x*wy - y*wx
    ]) # each row computes time derivative of one component in q 


# Translational + rotational dynamics. 

def state_derivative(state, force_body, torque_body, wind, p):

    """Calculates rates of change of the drone's 13 state variables. Actuators provide forces and torque
    """

    # Convert force from body frame to world frame
    thrust = rotation_matrix(state.orientation) @ force_body

    gravity = np.array([0.0, 0.0, -p.mass * p.gravity])

    # Vdrone/air
    air_velocity = state.velocity - wind

    # drag formula
    drag = (
        -0.5 * p.air_density
        * p.drag_coeff
        * p.reference_area
        * np.linalg.norm(air_velocity)
        * air_velocity
    )

    # Translational acceleration: Fnet=ma
    acceleration_world = (
        thrust + gravity + drag
    ) / p.mass # easy! 

    omega = state.angular_velocity # wx, wy, wz
    angular_momentum = p.inertia @ omega # 3D L=Jw
    gyroscopic_coupling = np.cross( # dldt world = dldtbody + wxL 
        omega, angular_momentum
    )

    damping_torque = -p.angular_damping * omega

    angular_acceleration = np.linalg.solve(
        p.inertia,
        torque_body + damping_torque - gyroscopic_coupling
    ) # T_body + T_damping - wxL equals J(alpha) - J is the rotational inertia matrix
    # we are finding the angular acceleration alppha, from J(alpha)=b.

    # Return time derivative of the states
    return DroneState(
        position=state.velocity,
        velocity=acceleration_world,
        orientation=quaternion_derivative(state.orientation, omega),
        angular_velocity=angular_acceleration
    )


def _add_state(state, derivative, multiplier):
    """Intermediate state used by the numerical integrator."""

    return DroneState(
        position=state.position + multiplier * derivative.position,
        velocity=state.velocity + multiplier * derivative.velocity,
        orientation=state.orientation + multiplier * derivative.orientation,
        angular_velocity=state.angular_velocity
                         + multiplier * derivative.angular_velocity
    )


def step(state, force_body, torque_body, wind_world, p, dt):
    """Advance the physical simulation by dt seconds using RK4."""

    if dt <= 0:
        raise ValueError("dt must be positive")

    if p.mass <= 0:
        raise ValueError("mass must be positive")

    # Assume the applied forces and torques remain constant
    # during this short time step.
    derivative = lambda s: state_derivative(
        s, force_body, torque_body, wind_world, p
    )

    k1 = derivative(state)
    k2 = derivative(_add_state(state, k1, dt / 2))
    k3 = derivative(_add_state(state, k2, dt / 2))
    k4 = derivative(_add_state(state, k3, dt))

    def integrate(name):
        return getattr(state, name) + dt / 6 * (
            getattr(k1, name)
            + 2 * getattr(k2, name)
            + 2 * getattr(k3, name)
            + getattr(k4, name)
        )

    new = DroneState(
        position=integrate("position"),
        velocity=integrate("velocity"),
        orientation=normalize_quaternion(integrate("orientation")),
        angular_velocity=integrate("angular_velocity")
    )

    return new


# Basic hover test

if __name__=="__main__":

    params = DroneParameters(
        mass=1.0,
        inertia=np.diag([0.008, 0.008, 0.016])
    )

    state = DroneState(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3)
    )

    # Simulate one second with thrust equal to weight
    for _ in range(100):
        state = step(
            state,
            force_body=np.array([0.0, 0.0, 9.81]),
            torque_body=np.zeros(3),
            wind_world=np.zeros(3),
            p=params,
            dt=0.01
        )

    print("Position after 1 s:", np.round(state.position, 4))
    print("Quaternion:", np.round(state.orientation, 4))
