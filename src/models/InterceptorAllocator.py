import numpy as np

class InterceptorAllocator:
    def __init__(self, params):
        self.p = params
        
        xf = self.p.l_front       # 0.164m
        xa = self.p.l_aft         # 0.132m
        yf = self.p.rotor_y_front # 0.2185m
        ya = self.p.rotor_y_aft   # 0.4515m
        c_tau = 0.015             # Approximate torque-to-thrust ratio
        
        # Asymmetric H-frame Layout matching rotors.py physics:
        # 1: Front-Right (+xf, +yf) CCW (+c_tau)
        # 2: Aft-Left   (-xa, -ya) CCW (+c_tau)
        # 3: Front-Left (+xf, -yf) CW  (-c_tau)
        # 4: Aft-Right  (-xa, +ya) CW  (-c_tau)
        self._control_allocation_matrix = np.array([
            [  1.0,    1.0,    1.0,    1.0  ], # Total Thrust
            [ -yf,     ya,     yf,    -ya   ], # Roll Torque (-y*T)
            [  xf,    -xa,     xf,    -xa   ], # Pitch Torque (+x*T)
            [ c_tau,  c_tau, -c_tau, -c_tau ]  # Yaw Torque (+s_i*Q)
        ])
        
        self._inverse_allocation_matrix = np.linalg.inv(self._control_allocation_matrix)
        
        # Calculate thrust constant at J=0 (Hover) to convert N to rev/s.
        # CT at hover must come from the same CT(J) table rotors.py actually
        # uses (CT_table_val[0], J=0) -- a hardcoded guess here silently
        # mismatches the real rotor thrust model. That mismatch previously
        # under-delivered hover thrust by ~12% (0.125 assumed vs the real
        # 0.11 at J=0), causing "hover trim" to actually sink continuously
        # instead of holding altitude.
        CT_hover = self.p.CT_table_val[0]
        self.k_thrust = CT_hover * self.p.rho * (self.p.rotor_diameter ** 4)
        self.max_thrust = getattr(self.p, 'max_thrust_per_motor', 25.0)

    def allocate(self, control_cmds_6dof):
        """
        Takes [Fx, Fy, Fz, Mx, My, Mz] and returns the 10-actuator command:
        [delta1, delta2, lam1..4, n1..4]
        """
        # Fz is negative for upward thrust in standard NED
        req_thrust = -control_cmds_6dof[2]
        roll_cmd = control_cmds_6dof[3]
        pitch_cmd = control_cmds_6dof[4]
        yaw_cmd = control_cmds_6dof[5]
        
        # Matrix solve for individual physical motor thrusts in Newtons
        quad_cmds = np.array([req_thrust, roll_cmd, pitch_cmd, yaw_cmd])
        motor_thrusts = self._inverse_allocation_matrix @ quad_cmds
        
        # Clip to physical limits to prevent negative square roots or saturation
        motor_thrusts = np.clip(motor_thrusts, 0.0, self.max_thrust)
        
        # Invert the aerodynamic formula to get physical revolutions per second
        n_revs = np.sqrt(motor_thrusts / self.k_thrust)
        
        # Return [delta_e(2), lam(4), n_revs(4)]
        # For pure hover control, elevons are flat (0.0) and tilts point up (0.0)
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_revs[0], n_revs[1], n_revs[2], n_revs[3]])