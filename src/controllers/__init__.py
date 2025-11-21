"""Controller module: contains all controller implementations."""

from .base import BaseController
from .registry import register_controller, create_controller, list_controllers, CONTROLLER_REGISTRY

# Import all controllers to trigger registration
from .pid_controller import PIDController
from .pid_deadzone import PIDDeadzoneController
from .event_pid import EventPIDController
from .lif_snn_controller import LIFSpikingController
from .trinc_controller import TRINCController

__all__ = [
    "BaseController",
    "register_controller",
    "create_controller",
    "list_controllers",
    "CONTROLLER_REGISTRY",
    "PIDController",
    "PIDDeadzoneController",
    "EventPIDController",
    "LIFSpikingController",
    "TRINCController",
]
