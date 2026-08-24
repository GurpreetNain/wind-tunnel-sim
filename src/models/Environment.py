import numpy as np
import json
import os

class Environment:
    def __init__(self, seed, wingspan_meters=0.5):
        # 1. Load the JSON configuration parameters once on startup
        current_dir = os.path.dirname(__file__)
        config_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'config', 'Environment.json'))
        
        # 2. Open using the absolute path
        with open(config_path, "r") as env_file:    
            env_data = json.load(env_file)

        self.g = np.array(env_data["gravity"])
        self.air_density = env_data["atmosphere"]["air_density"]
        self.translational_drag_coeff = np.array(env_data["atmosphere"]["drag_coefficients"]["translational"])
        self.rotational_drag_coeff = np.array(env_data["atmosphere"]["drag_coefficients"]["rotational"])
        self.wind_turbulence_model = env_data["atmosphere"]["wind_turbulence"]["model"]
        self.W20_knots = env_data["atmosphere"]["wind_turbulence"]["W20_turbulence"]

        self._wind_speed_vector = np.array([0.0, 0.0, 0.0])
        self._wind_rotation_vector = np.array([0.0, 0.0, 0.0])

        # The Dryden model operates entirely in feet internally
        self._b = wingspan_meters * 3.28084  
        self._master_seed = 42
        self._seed = seed + self._master_seed
        self._rng = np.random.default_rng(self._seed)
        
        # Internal state memory for the filters (State-space vector)
        self._X = np.zeros(6)

    def _get_wind_params(self, h_ft):
        """Calculates standard MIL-F-8785C parameters dynamically in Imperial Units."""
        h = max(h_ft, 10.0)  # Prevent division by zero at ground level
        W20 = self.W20_knots * 1.68781  # knots to ft/s
        
        sigma_w = 0.1 * W20
        denom = (0.177 + 0.000823 * h) ** 0.4
        sigma_u = sigma_w / denom
        sigma_v = sigma_u
        
        L_w = h
        L_u = h / ((0.177 + 0.000823 * h) ** 1.2)
        L_v = L_u
        
        return sigma_u, sigma_v, sigma_w, L_u, L_v, L_w

    def _dryden_dynamics(self, X_state, V, L_u, L_v, L_w, noise):
        """System dynamics function returning system state derivatives (dX/dt)."""
        dX = np.zeros(6)
        
        # 1. Longitudinal time constants
        T_u = L_u / V
        dX[0] = (-1.0 / T_u) * X_state[0] + (1.0 / T_u) * noise[0]
        
        # 2. Lateral state-space matrices
        T_v = L_v / V
        A_v = np.array([[0.0, 1.0], [-1.0/(T_v**2), -2.0/T_v]])
        B_v = np.array([0.0, 1.0])
        dX[1:3] = A_v @ X_state[1:3] + B_v * noise[1]
        
        # 3. Vertical state-space matrices
        T_w = L_w / V
        A_w = np.array([[0.0, 1.0], [-1.0/(T_w**2), -2.0/T_w]])
        B_w = np.array([0.0, 1.0])
        dX[3:5] = A_w @ X_state[3:5] + B_w * noise[2]
        
        # 4. Roll Rate time constants
        T_p = (4.0 * self._b) / (np.pi * V)
        dX[5] = (-1.0 / T_p) * X_state[5] + (1.0 / T_p) * noise[3]
        
        return dX
    
    def sample_noise(self, dt):
        self._noise = self._rng.normal(0.0, 1.0 / np.sqrt(dt), 4)

    def update_turbulence(self, dt, h_meters, V_mps):
        """Updates the filter memory and returns gusts for the current frame in SI units."""
        if self.wind_turbulence_model == "none":
            # self._wind_speed_vector = np.array([0.0, 0.0, 0.0])
            # self._wind_rotation_vector = np.array([0.0, 0.0, 0.0])
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
            
        elif self.wind_turbulence_model == "dryden":
            h_ft = h_meters / 0.3048
            V_kts = V_mps * 1.94384449
            V = max(V_kts * 1.68781, 10.0) 
            
            sig_u, sig_v, sig_w, L_u, L_v, L_w = self._get_wind_params(h_ft)
            
            
            # --- RUNGE-KUTTA 4th ORDER INTEGRATION ---
            k1 = self._dryden_dynamics(self._X,            V, L_u, L_v, L_w, self._noise)
            k2 = self._dryden_dynamics(self._X + 0.5*dt*k1, V, L_u, L_v, L_w, self._noise)
            k3 = self._dryden_dynamics(self._X + 0.5*dt*k2, V, L_u, L_v, L_w, self._noise)
            k4 = self._dryden_dynamics(self._X + dt*k3,     V, L_u, L_v, L_w, self._noise)
            
            self._X += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            # -----------------------------------------

            # Extract outputs from the internal state array
            K_u = sig_u * np.sqrt(2 * L_u / (np.pi * V))
            u_g = K_u * self._X[0]
            
            T_w = L_w / V
            K_w = sig_w * np.sqrt(L_w / (np.pi * V))
            C_w = np.array([1.0, np.sqrt(3.0) * T_w])
            w_g = K_w * (C_w @ self._X[3:5])
            
            T_v = L_v / V
            K_v = sig_v * np.sqrt(L_v / (np.pi * V))
            C_v = np.array([1.0, np.sqrt(3.0) * T_v])
            v_g = K_v * (C_v @ self._X[1:3])

            K_p = sig_w * np.sqrt(0.8 / V) * ((np.pi / (2.0 * self._b))**(1.0/6.0)) / (L_w**(1.0/3.0))
            p_g = K_p * self._X[5]

            # Spatial derivative approximations
            T_q = (4.0 * self._b) / (np.pi * V)
            q_g = (w_g / V) / T_q  
            
            T_r = (3.0 * self._b) / (np.pi * V)
            r_g = -(v_g / V) / T_r

            # self._wind_speed_vector = np.array([u_g * 0.3048, v_g * 0.3048, w_g * 0.3048])
            # self._wind_rotation_vector = np.array([p_g, q_g, r_g])
            return np.array([u_g * 0.3048, v_g * 0.3048, w_g * 0.3048]), np.array([p_g, q_g, r_g])
            
        else:
            const_wind_mps = self.W20_knots * 0.514444 * (1.0 / np.sqrt(3.0))
            # self._wind_speed_vector = np.array([const_wind_mps, const_wind_mps, const_wind_mps])
            # self._wind_rotation_vector = np.array([0.0, 0.0, 0.0])
            return np.array([const_wind_mps, const_wind_mps, const_wind_mps]), np.array([0.0, 0.0, 0.0])
    
    def get_wind_speed_vector(self):
        return self._wind_speed_vector
    
    def get_wind_rotation_vector(self):
        return self._wind_rotation_vector