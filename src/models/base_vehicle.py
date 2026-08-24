import numpy as np
from abc import ABC, abstractmethod

class BaseVehicle(ABC):
    def __init__(self, id_val, mass):
        self._id = id_val
        self._mass = mass
        self._state_vector = None
        self._control_vector = None
        self._env = None 

    def get_state_vector(self):
        """Returns the current state vector of the vehicle."""
        return self._state_vector
        
    def set_state_vector(self, state_vector):
        """Overrides the current state vector (useful for initialization)."""
        self._state_vector = np.asarray(state_vector, dtype=float).copy()
        
    def set_control_vector(self, control_vector):
        """Sets the actuator commands for the next physics step."""
        self._control_vector = np.asarray(control_vector, dtype=float).copy()

    @abstractmethod
    def _calculate_dynamics(self, state_vector, wind_speed, wind_rot):
        """
        Core physics engine. Must return the state derivative (x_dot).
        To be implemented by child classes.
        """
        pass

    @abstractmethod
    def state_update(self, dt):
        """
        Steps the vehicle physics forward by dt (e.g., using RK4).
        To be implemented by child classes.
        """
        pass