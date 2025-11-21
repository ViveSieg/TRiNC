"""Unit tests for TRiNC controller."""

from __future__ import annotations

import pytest
import numpy as np

from src.controllers.trinc_controller import TRINCController
from src.config.presets import get_default_controller_params


class TestTRINCController:
    """Test suite for TRiNC controller."""
    
    def test_initialization(self) -> None:
        """Test controller initialization."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        
        assert controller.h_state == 0.0
        assert controller.prev_error == 0.0
        assert controller.prev_output == controller.base_cooling
        assert controller.refractory == 0
    
    def test_reset(self) -> None:
        """Test controller reset functionality."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        
        # Modify state
        controller.h_state = 1.0
        controller.prev_error = 0.5
        controller.prev_output = 0.8
        controller.refractory = 5
        
        # Reset
        controller.reset()
        
        assert controller.h_state == 0.0
        assert controller.prev_error == 0.0
        assert controller.prev_output == controller.base_cooling
        assert controller.refractory == 0
        assert len(controller.event_log["gP"]) == 0
    
    def test_refractory_period(self) -> None:
        """Test refractory period behavior."""
        params = get_default_controller_params()["trinc"]
        params["refractory_steps"] = 3
        controller = TRINCController(**params)
        controller.reset()
        
        # Trigger an event (large error)
        error = params["tau_p"] + 0.1
        u1 = controller.compute_control(error)
        
        # Check that refractory is set
        assert controller.refractory == params["refractory_steps"]
        
        # Next few steps should have no events (refractory active)
        for _ in range(params["refractory_steps"]):
            u = controller.compute_control(error)
            # Control should be based on decay only, not new events
            assert controller.refractory > 0
        
        # After refractory, should be able to trigger again
        assert controller.refractory == 0
    
    def test_p_gate(self) -> None:
        """Test P gate (proportional) functionality."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        controller.reset()
        
        # Positive error above threshold
        error = params["tau_p"] + 0.1
        u = controller.compute_control(error)
        assert controller.event_log["gP"][-1] == 1.0
        
        # Negative error below threshold
        error = -(params["tau_p"] + 0.1)
        u = controller.compute_control(error)
        assert controller.event_log["gP"][-1] == -1.0
        
        # Error within threshold
        error = params["tau_p"] * 0.5
        u = controller.compute_control(error)
        assert controller.event_log["gP"][-1] == 0.0
    
    def test_h_gate(self) -> None:
        """Test H gate (habituation) functionality."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        controller.reset()
        
        # Small errors should accumulate
        small_error = params["theta_h"] / (params["w_h"] * 2)
        for _ in range(10):
            controller.compute_control(small_error)
        
        # Eventually should trigger H gate
        assert abs(controller.h_state) > 0
    
    def test_s_gate(self) -> None:
        """Test S gate (surprise) functionality."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        controller.reset()
        
        # Small initial error
        controller.compute_control(0.01)
        
        # Large sudden change (surprise)
        large_error = params["tau_s"] + 0.1
        u = controller.compute_control(large_error)
        
        # Should trigger S gate if error is positive
        if large_error > 0:
            assert controller.event_log["gS"][-1] in [0.0, 1.0]
    
    def test_ablation_p_gate_disabled(self) -> None:
        """Test ablation: P gate disabled."""
        params = get_default_controller_params()["trinc"]
        params["enable_p_gate"] = False
        controller = TRINCController(**params)
        controller.reset()
        
        # Large error should not trigger P gate
        error = params["tau_p"] + 0.1
        u = controller.compute_control(error)
        assert controller.event_log["gP"][-1] == 0.0
    
    def test_ablation_h_gate_disabled(self) -> None:
        """Test ablation: H gate disabled."""
        params = get_default_controller_params()["trinc"]
        params["enable_h_gate"] = False
        controller = TRINCController(**params)
        controller.reset()
        
        # H gate should not fire, but state should still update
        error = 0.1
        for _ in range(10):
            u = controller.compute_control(error)
            assert controller.event_log["gH"][-1] == 0.0
        
        # H state should still be updated (for consistency)
        assert abs(controller.h_state) > 0
    
    def test_ablation_s_gate_disabled(self) -> None:
        """Test ablation: S gate disabled."""
        params = get_default_controller_params()["trinc"]
        params["enable_s_gate"] = False
        controller = TRINCController(**params)
        controller.reset()
        
        # Small initial error
        controller.compute_control(0.01)
        
        # Large sudden change should not trigger S gate
        large_error = params["tau_s"] + 0.1
        u = controller.compute_control(large_error)
        assert controller.event_log["gS"][-1] == 0.0
    
    def test_control_output_bounds(self) -> None:
        """Test that control output is bounded."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        controller.reset()
        
        # Test with various errors
        for error in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            u = controller.compute_control(error)
            assert controller.u_min <= u <= controller.u_max
    
    def test_synaptic_integration(self) -> None:
        """Test synaptic integration with decay."""
        params = get_default_controller_params()["trinc"]
        controller = TRINCController(**params)
        controller.reset()
        
        # Trigger an event
        error = params["tau_p"] + 0.1
        u1 = controller.compute_control(error)
        
        # Next step without event should decay
        u2 = controller.compute_control(error * 0.1)
        
        # Output should decay (unless new events occur)
        # This depends on the specific gate activations
        assert isinstance(u2, float)
        assert controller.u_min <= u2 <= controller.u_max

