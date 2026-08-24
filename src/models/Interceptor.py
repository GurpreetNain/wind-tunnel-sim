import numpy as np
from src.common.utils import quat2rot
from src.models.base_vehicle import BaseVehicle
from src.models.Environment import Environment
from src.actuators.params import InterceptorParams
from src.actuators import rotors, lifting_body, elevons
from src.models.InterceptorAllocator import InterceptorAllocator

class Interceptor(BaseVehicle):
    STATE_NAMES = ["xe", "ye", "h", "u", "v", "w",
                   "phi", "theta", "psi", "p", "q", "r"]

    def __init__(self, params: InterceptorParams = None, id=1):
        self.p = params if params is not None else InterceptorParams()
        
        # 1. Properly inherit BaseVehicle to initialize self._mass and self._id
        super().__init__(id_val=id, mass=self.p.mass)
        
        self._state_vector = np.zeros(13)
        self._state_vector[6] = 1.0  # Initialize the Quaternion (Qw = 1)
        
        self._control_vector = np.zeros(10)   # [d1, d2, lam1..4, n1..4]
        self._env = Environment(wingspan_meters=self.p.b, seed=id)
        
        # Pass the parameters directly to the allocator so it can map the geometry
        self._allocator = InterceptorAllocator(params=self.p)

    def get_state_vector(self):
        return self._state_vector

    def set_state_vector(self, x):
        self._state_vector = np.asarray(x, dtype=float).copy()

    def set_control_vector(self, physics_control_vector):
        """
        Receives [Fx, Fy, Fz, Mx, My, Mz] from the agent.
        Translates it into [d1, d2, lam1..4, n1..4] normalized actuator commands.
        """
        # Store the allocated 10-element actuator array directly into the vehicle's state
        self._control_vector = self._allocator.allocate(physics_control_vector)

    def _calculate_dynamics(self, state_vector, wind_speed, wind_rot):
        global_position = state_vector[0:3]
        local_linear_velocity = state_vector[3:6]
        quaternion_attitude = state_vector[6:10]
        local_angular_velocity = state_vector[10:13]
        wx, wy, wz = local_angular_velocity[0], local_angular_velocity[1], local_angular_velocity[2]

        delta1, delta2 = self._control_vector[0], self._control_vector[1]
        lam = self._control_vector[2:6]
        n = self._control_vector[6:10]

        # 1. Rotors
        rotor_forces, rotor_moments, rotor_thrust = rotors.rotor_forces_moments(
            n=n, lam=lam, params=self.p, lam_dot=None, 
            u=local_linear_velocity[0], v=local_linear_velocity[1], w=local_linear_velocity[2],
            p=wx, q=wy, r=wz
        )
        
        # FIX: Extract only the front pair thrust to prevent the 0-dimensional scalar crash
        Tf = rotors.front_pair_thrust(rotor_thrust)
        lam_f = lam[0]

        # 2. Airdata + downwash coupling
        Va, alpha, beta, qbar = lifting_body.airdata(local_linear_velocity[0], local_linear_velocity[1], local_linear_velocity[2], self.p)
        eps, qbar_wing = lifting_body.downwash(Tf, lam_f, Va, qbar, self.p)
        alpha_wing = alpha - eps

        # 3. Elevon input split + stall interlock
        delta_e, delta_a = elevons.split(delta1, delta2)
        delta_e = elevons.apply_stall_constraint(delta_e, alpha_wing, self.p)

        # 4. Clean wing aerodynamics
        CL, CD, Cm, Cl, Cn = lifting_body.aero_coefficients(alpha_wing, wx, wy, wz, beta, Va, self.p)
        Lclean, D, Y, M = lifting_body.clean_forces_moments(CL, CD, Cl, Cn, qbar_wing, wy, Va, beta, self.p, alpha_wing)
        lifting_body_forces, lifting_body_moments = lifting_body.project_to_body(Lclean, D, Y, M, alpha_wing, beta)

        # 5. Elevon forces/moments
        elevon_forces, elevon_moments = elevons.elevon_forces_moments(delta_e, delta_a, alpha_wing, qbar_wing, self.p)

        # 6. Gravity (Safely uses initialized _mass and _env)
        gravity_body = np.transpose(quat2rot(quaternion_attitude)) @ (self._mass * np.array(self._env.g))

        # 7. Sum forces and moments
        SF = elevon_forces + rotor_forces + lifting_body_forces
        SM = lifting_body_moments + rotor_moments + elevon_moments

        # Diagnostics snapshot for external inspection (e.g. scripts/wind_tunnel_gui.py).
        # Pure bookkeeping of values already computed above -- does not affect dynamics.
        self.last_diagnostics = {
            'Va': Va, 'alpha': alpha, 'alpha_wing': alpha_wing, 'beta': beta,
            'qbar': qbar, 'qbar_wing': qbar_wing, 'eps': eps,
            'CL': CL, 'CD': CD, 'Cm': Cm, 'Cl': Cl, 'Cn': Cn,
            # wing_scale is 0 at true hover: the wing is producing no force at all,
            # so the CL/CD above are the coefficients the table gives at this alpha,
            # NOT coefficients of any force actually acting. CL_eff/CD_eff are what
            # the reported Lift/Drag correspond to, which is what makes "CL=-0.0208
            # but Lift=0.00 N" at hover read consistently instead of contradictory.
            'wing_scale': lifting_body.wing_activity(Va, self.p),
            'CL_eff': CL * lifting_body.wing_activity(Va, self.p),
            'CD_eff': CD * lifting_body.wing_activity(Va, self.p),
            'Lclean': Lclean, 'D': D, 'Y': Y,
            'rotor_forces': rotor_forces.copy(), 'rotor_moments': rotor_moments.copy(),
            'rotor_thrust': rotor_thrust.copy(),
            'lifting_body_forces': lifting_body_forces.copy(), 'lifting_body_moments': lifting_body_moments.copy(),
            'elevon_forces': elevon_forces.copy(), 'elevon_moments': elevon_moments.copy(),
            'total_force': SF.copy(), 'total_moment': SM.copy(),
        }

        # 8. State derivatives
        state_derivate = np.zeros(13)
        state_derivate[0:3] = quat2rot(quaternion_attitude) @ local_linear_velocity

        state_derivate[3:6] = (1 / self._mass) * (gravity_body - 
                                                  self._mass * np.cross(local_angular_velocity, local_linear_velocity) + 
                                                  SF)
        
        if (state_vector[2] >= 0.0) and (state_derivate[2] > 0.0):
            state_derivate[2] = 0.0
            local_linear_velocity[2] = 0.0
            state_derivate[5] = 0.0

        Omega = np.array([[  0, -wx, -wy, -wz],
                          [ wx,   0,  wz, -wy],
                          [ wy, -wz,   0,  wx],
                          [ wz,  wy, -wx,   0]])
                
        state_derivate[6:10] = 0.5 * (Omega @ quaternion_attitude)
        
        # Build the Inertia Tensor explicitly from InterceptorParams
        J_matrix = np.array([
            [self.p.Ixx, 0.0, -self.p.Ixz],
            [0.0, self.p.Iyy, 0.0],
            [-self.p.Ixz, 0.0, self.p.Izz]
        ])
        
        state_derivate[10:13] = np.linalg.inv(J_matrix) @ (SM - np.cross(local_angular_velocity, J_matrix @ local_angular_velocity))
        return state_derivate

    def state_update(self, dt):
        """Fixed-step RK4 integration with properly handled environment turbulence."""
        self._env.sample_noise(dt)
        X_env_baseline = self._env._X.copy() 
        
        wind_speed_0 = self._env.get_wind_speed_vector()
        wind_rot_0 = self._env.get_wind_rotation_vector()
        k1 = self._calculate_dynamics(self._state_vector, wind_speed_0, wind_rot_0)

        pos_mid1 = self._state_vector[0:3] + k1[0:3] * (dt / 2.0)
        vel_mid1 = self._state_vector[3:6] + k1[3:6] * (dt / 2.0)
        
        self._env._X = X_env_baseline.copy()
        w_speed_mid1, w_rot_mid1 = self._env.update_turbulence(
            dt / 2.0, h_meters=max(-pos_mid1[2], 0.1), V_mps=np.linalg.norm(vel_mid1 - wind_speed_0)
        )
        k2 = self._calculate_dynamics(self._state_vector + k1 * (dt / 2.0), w_speed_mid1, w_rot_mid1)

        pos_mid2 = self._state_vector[0:3] + k2[0:3] * (dt / 2.0)
        vel_mid2 = self._state_vector[3:6] + k2[3:6] * (dt / 2.0)
        
        self._env._X = X_env_baseline.copy()
        w_speed_mid2, w_rot_mid2 = self._env.update_turbulence(
            dt / 2.0, h_meters=max(-pos_mid2[2], 0.1), V_mps=np.linalg.norm(vel_mid2 - w_speed_mid1)
        )
        k3 = self._calculate_dynamics(self._state_vector + k2 * (dt / 2.0), w_speed_mid2, w_rot_mid2)

        pos_end = self._state_vector[0:3] + k3[0:3] * dt
        vel_end = self._state_vector[3:6] + k3[3:6] * dt
        
        self._env._X = X_env_baseline.copy()
        w_speed_end, w_rot_end = self._env.update_turbulence(
            dt, h_meters=max(-pos_end[2], 0.1), V_mps=np.linalg.norm(vel_end - w_speed_mid2)
        )
        k4 = self._calculate_dynamics(self._state_vector + k3 * dt, w_speed_end, w_rot_end)
        
        # FIX: Correctly bracketed RK4 calculation
        self._state_vector += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        q = self._state_vector[6:10]
        self._state_vector[6:10] = q / np.linalg.norm(q)
        
        self._env._X = X_env_baseline.copy()
        self._env.update_turbulence(dt, h_meters=max(-self._state_vector[2], 0.1), V_mps=np.linalg.norm(self._state_vector[3:6] - w_speed_end))