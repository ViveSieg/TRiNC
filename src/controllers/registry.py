"""Controller registry: factory pattern for controller instantiation."""

from __future__ import annotations

from typing import Dict, Type, Any, Callable

from .base import BaseController


# Controller registry
CONTROLLER_REGISTRY: Dict[str, Type[BaseController]] = {}


def register_controller(name: str) -> Callable[[Type[BaseController]], Type[BaseController]]:
    """Decorator: register controller class.
    
    Parameters
    ----------
    name : str
        Controller registration name
    
    Returns
    -------
    Callable
        Decorator function
    """
    def decorator(cls: Type[BaseController]) -> Type[BaseController]:
        CONTROLLER_REGISTRY[name] = cls
        return cls
    return decorator


def create_controller(name: str, params: Dict[str, Any], **kwargs: Any) -> BaseController:
    """Factory function: create controller instance.
    
    Handles controller-specific __init__ arguments (e.g., Ts for PIDs).
    
    Parameters
    ----------
    name : str
        Controller registration name
    params : Dict[str, Any]
        Controller parameter dictionary
    **kwargs
        Additional keyword arguments (e.g., Ts)
    
    Returns
    -------
    BaseController
        Controller instance
    
    Raises
    ------
    KeyError
        If controller name is not registered
    """
    if name not in CONTROLLER_REGISTRY:
        raise KeyError(
            f"Controller '{name}' not registered. "
            f"Available controllers: {list(CONTROLLER_REGISTRY.keys())}"
        )
    
    controller_class = CONTROLLER_REGISTRY[name]
    
    # PID-based controllers require Ts parameter
    if name in {"pid", "pid_deadzone", "event_pid"}:
        if "Ts" not in params and "Ts" not in kwargs:
            raise ValueError(f"Controller '{name}' requires 'Ts' parameter")
        if "Ts" in kwargs:
            params = {**params, "Ts": kwargs["Ts"]}
    
    return controller_class(**params)


def list_controllers() -> list[str]:
    """List all registered controller names.
    
    Returns
    -------
    list[str]
        List of controller names
    """
    return list(CONTROLLER_REGISTRY.keys())


__all__ = ["register_controller", "create_controller", "list_controllers", "CONTROLLER_REGISTRY"]
