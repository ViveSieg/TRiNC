"""Default simulation and controller parameters."""


def default_simulation_params():
    """Return default plant and simulation parameters."""
    return {
        "tau_th": 2.0,  # thermal time constant [s]
        "K_cool": -1.2,  # cooling gain (negative)
        "Ts": 0.1,  # sampling time [s]
        "T_ref": 0.55,  # reference temperature (normalized)
        "T0": 0.5,  # initial temperature
        "duration": 20.0,  # simulation horizon [s]
    }


def default_controller_params():
    """Return tuning dictionaries for each controller."""
    return {
        "pid": {"Kp": 3.0, "Ki": 0.8, "Kd": 0.2},
        "pid_deadzone": {"Kp": 3.0, "Ki": 0.6, "Kd": 0.2, "deadband": 0.015},
        "event_pid": {
            "Kp": 3.0,
            "Ki": 0.6,
            "Kd": 0.2,
            "error_threshold": 0.012,
            "delta_threshold": 0.008,
            "decay": 0.01,
        },
        "lif_snn": {
            "leak": 0.02,
            "threshold": 0.06,
            "reset_value": 0.01,
            "pulse_magnitude": 0.2,
            "decay": 0.05,
        },
        "trinc": {
            "tau_p": 0.033,
            "tau_s": 0.036,
            "theta_h": 0.055,
            "alpha_h": 0.035,
            "w_h": 0.24,
            "rho": 0.88,
            "a_p": 0.32,
            "a_h": 0.125,
            "a_s": 0.27,
            "refractory_steps": 9,
            "base_cooling": 0.316,
        },
    }
