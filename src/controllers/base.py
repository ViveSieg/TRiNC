"""Controller abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseController(ABC):
    """Abstract base class for all controllers."""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller state.
        
        This method must be called before each simulation to initialize
        the controller's internal state.
        """
        pass
    
    @abstractmethod
    def compute_control(self, error: float) -> float:
        """Compute control signal.
        
        Parameters
        ----------
        error : float
            Temperature error, defined as T_current - T_ref
        
        Returns
        -------
        float
            Control signal u, typically normalized to [0, 1] range
        """
        pass


__all__ = ["BaseController"]
