"""Pytest for variational decision engine v2 upgrade."""

import numpy as np
import pytest

from cognitive_engine.engine import (
    CognitiveEngine,
    CognitiveState,
    Event,
    LegacyDecisionEngine,
    Observation,
    PersonProfile,
    VariationalDecisionEngine,
)


class TestVariationalDecisionEngine:
    """Test variational solver for continuous action paths."""

    def test_smooth_potential_produces_continuous_path(self):
        """
        Test 1: Given smooth potential, solver produces continuous path.
        
        Assertion: Adjacent points in action_path differ by < bounded amount.
        """
        engine = VariationalDecisionEngine(n_steps=12, kappa=1.0)
        
        state = CognitiveState(
            rho_0=np.array([0.6]),
            H_0=np.array([0.2]),
            mu=0.3,
            sigma_star=np.array([0.5]),
            uncertainty=0.4,
        )
        
        event = Event(
            description="smooth test",
            rho_now=0.6,
            env_pressure=0.2,
            goal_horizon=0.7,
        )
        
        result = engine.decide(state, event)
        
        assert result["action_path"] is not None, "Path should be computed"
        path = np.array(result["action_path"])
        
        # Adjacent points should be close (smooth path)
        diffs = np.diff(path)
        max_diff = np.max(np.abs(diffs))
        assert max_diff < 0.3, f"Path not smooth: max adjacent diff = {max_diff}"
        
        # All points in [0, 1]
        assert np.all(path >= 0.0) and np.all(path <= 1.0), "Path out of bounds"

    def test_is_stationary_true_on_convergence(self):
        """
        Test 2: Stationary check confirms true minimum.
        
        Assertion: is_stationary=True when solution is verified as local minimum.
        """
        engine = VariationalDecisionEngine(n_steps=12, kappa=1.0)
        
        state = CognitiveState(
            rho_0=np.array([0.5]),
            H_0=np.array([0.1]),
            mu=0.2,
            sigma_star=np.array([0.5]),
            uncertainty=0.3,
        )
        
        event = Event(
            description="stationary test",
            rho_now=0.5,
            env_pressure=0.1,
            goal_horizon=0.8,
        )
        
        result = engine.decide(state, event)
        
        # Should converge and verify minimum
        assert result["is_stationary"] is not None, "Stationarity check should run"
        assert result["is_stationary"] == True, "Should detect true minimum for smooth problem"

    def test_action_value_converges_with_n_steps(self):
        """
        Test 3: Action functional value converges (stable) as N increases.
        
        Assertion: Increasing n_steps should not cause Φ to diverge or oscillate wildly.
        """
        state = CognitiveState(
            rho_0=np.array([0.55]),
            H_0=np.array([0.15]),
            mu=0.25,
            sigma_star=np.array([0.5]),
            uncertainty=0.35,
        )
        
        event = Event(
            description="convergence test",
            rho_now=0.55,
            env_pressure=0.15,
            goal_horizon=0.75,
        )
        
        values = []
        for n in [6, 12, 24]:
            engine = VariationalDecisionEngine(n_steps=n, kappa=1.0)
            result = engine.decide(state, event)
            values.append(result["action_value"])
        
        # Check that values are in reasonable range (not diverging)
        assert all(0.0 <= v < 10.0 for v in values), f"Values diverging: {values}"
        
        # Check that differences between steps are bounded (convergence)
        diff_6_12 = abs(values[1] - values[0])
        diff_12_24 = abs(values[2] - values[1])
        
        # Later refinements should produce smaller changes
        assert diff_12_24 < diff_6_12 * 1.5, "Convergence test: should stabilize"

    def test_fallback_on_forced_exception(self):
        """
        Test 4: Fallback mechanism triggers on optimization failure.
        
        Assertion: When optimization fails, engine falls back to legacy and marks fallback=True.
        """
        # Create engine with fallback enabled
        engine = VariationalDecisionEngine(n_steps=12, kappa=1.0, enable_fallback=True)
        
        # Inject NaN to force failure
        state = CognitiveState(
            rho_0=np.array([np.nan]),  # NaN will cause optimization to fail
            H_0=np.array([0.0]),
            mu=0.0,
            sigma_star=np.array([0.5]),
            uncertainty=1.0,
        )
        
        event = Event(
            description="fallback test",
            rho_now=np.nan,
            env_pressure=0.0,
            goal_horizon=0.5,
        )
        
        result = engine.decide(state, event)
        
        # Should fall back to legacy
        assert result["fallback"] == True, "Should trigger fallback on NaN"
        assert result["fallback_reason"] is not None, "Should provide fallback reason"

    def test_fallback_disabled_raises_error(self):
        """
        Test 5: With fallback disabled, optimization failure raises error.
        
        Assertion: disable_fallback=False causes exceptions to propagate.
        """
        engine = VariationalDecisionEngine(n_steps=12, kappa=1.0, enable_fallback=False)
        
        state = CognitiveState(
            rho_0=np.array([np.nan]),
            H_0=np.array([0.0]),
            mu=0.0,
            sigma_star=np.array([0.5]),
            uncertainty=1.0,
        )
        
        event = Event(
            description="error test",
            rho_now=np.nan,
            env_pressure=0.0,
            goal_horizon=0.5,
        )
        
        with pytest.raises(RuntimeError):
            engine.decide(state, event)

    def test_path_respects_bounds(self):
        """
        Test 6: All action path values are in [0, 1].
        
        Assertion: optimizer respects box constraints.
        """
        engine = VariationalDecisionEngine(n_steps=12, kappa=1.0)
        
        state = CognitiveState(
            rho_0=np.array([0.7]),
            H_0=np.array([0.3]),
            mu=0.4,
            sigma_star=np.array([0.6]),
            uncertainty=0.5,
        )
        
        event = Event(
            description="bounds test",
            rho_now=0.7,
            env_pressure=0.3,
            goal_horizon=0.8,
        )
        
        result = engine.decide(state, event)
        path = np.array(result["action_path"])
        
        assert np.all(path >= 0.0), "Path has values < 0"
        assert np.all(path <= 1.0), "Path has values > 1"

    def test_high_pressure_reduces_action(self):
        """
        Test 7: High environmental pressure should reduce action values.
        
        Assertion: action_path mean is lower when env_pressure is high.
        """
        state_low = CognitiveState(
            rho_0=np.array([0.5]),
            H_0=np.array([0.2]),
            mu=0.2,
            sigma_star=np.array([0.5]),
            uncertainty=0.3,
        )
        
        state_high = CognitiveState(
            rho_0=np.array([0.5]),
            H_0=np.array([0.2]),
            mu=0.2,
            sigma_star=np.array([0.5]),
            uncertainty=0.3,
        )
        
        engine = VariationalDecisionEngine(n_steps=12, kappa=1.0)
        
        event_low = Event(description="low pressure", rho_now=0.5, env_pressure=0.1, goal_horizon=0.7)
        event_high = Event(description="high pressure", rho_now=0.5, env_pressure=0.9, goal_horizon=0.7)
        
        result_low = engine.decide(state_low, event_low)
        result_high = engine.decide(state_high, event_high)
        
        path_low = np.array(result_low["action_path"])
        path_high = np.array(result_high["action_path"])
        
        mean_low = np.mean(path_low)
        mean_high = np.mean(path_high)
        
        # High pressure should suppress action
        assert mean_high <= mean_low, f"High pressure path mean {mean_high} should be <= low pressure {mean_low}"


class TestLegacyDecisionEngineBackcompat:
    """Verify legacy engine still works (for fallback compat)."""

    def test_legacy_decide_returns_action(self):
        """Legacy engine should return dict with a_star and best_action."""
        engine = LegacyDecisionEngine()
        
        state = CognitiveState(
            rho_0=np.array([0.5]),
            H_0=np.array([0.0]),
            mu=0.3,
            sigma_star=np.array([1.0]),
            uncertainty=0.8,
        )
        
        event = Event(
            description="legacy test",
            rho_now=0.5,
            env_pressure=0.2,
            goal_horizon=0.7,
        )
        
        result = engine.decide(state, event)
        
        assert "a_star" in result
        assert "best_action" in result
        assert result["best_action"] in ["immediate_relief", "invest_now", "balanced", "defer"]
        assert 0.0 <= result["a_star"] <= 1.0
        assert result["fallback"] == False  # Legacy doesn't use fallback flag


class TestCognitiveEngineVariational:
    """Integration test: CognitiveEngine with variational decision."""

    def test_engine_with_variational(self):
        """Engine should use variational solver by default."""
        engine = CognitiveEngine(use_variational=True)
        
        obs = Observation(
            language_content="This is a test message.",
            language_structure=0.5,
            emotional_tone=0.6,
            behavior_signal=0.4,
            nonverbal_signal=0.5,
            consistency=0.7,
        )
        
        event = Event(
            description="engine test",
            rho_now=0.4,
            env_pressure=0.2,
            goal_horizon=0.8,
        )
        
        result = engine.run(obs, event)
        
        assert result["system_stable"] is not None
        assert result["state"] is not None
        assert result["decision"]["action_path"] is not None, "Should have action_path from variational"

    def test_engine_with_legacy(self):
        """Engine should use legacy when use_variational=False."""
        engine = CognitiveEngine(use_variational=False)
        
        obs = Observation(
            language_content="Test message",
            language_structure=0.5,
            emotional_tone=0.5,
            behavior_signal=0.5,
            nonverbal_signal=0.5,
            consistency=0.5,
        )
        
        event = Event(
            description="legacy engine test",
            rho_now=0.5,
            env_pressure=0.2,
            goal_horizon=0.7,
        )
        
        result = engine.run(obs, event)
        
        assert "best_action" in result["decision"]
        assert result["decision"]["best_action"] in ["immediate_relief", "invest_now", "balanced", "defer"]


class TestMissingChannelHandling:
    """Test handling of missing observation channels (increases uncertainty)."""

    def test_missing_channel_increases_uncertainty(self):
        """
        Assertion: Missing channels should increase uncertainty (R) explicitly.
        """
        from cognitive_engine.engine import Encoder
        
        profile = PersonProfile(person_id="test", importance=0.5)
        encoder = Encoder(profile=profile)
        
        # Complete observation
        obs_complete = Observation(
            language_content="Test",
            language_structure=0.5,
            emotional_tone=0.5,
            behavior_signal=0.5,
            nonverbal_signal=0.5,
            consistency=0.5,
        )
        
        state_complete, _ = encoder.encode(obs_complete)
        
        # Reset encoder for next observation
        encoder.history = []
        encoder.stable_frames = 0
        
        # Observation with missing nonverbal channel
        obs_missing = Observation(
            language_content="Test",
            language_structure=0.5,
            emotional_tone=0.5,
            behavior_signal=0.5,
            nonverbal_signal=None,  # Missing
            consistency=0.5,
        )
        
        state_missing, _ = encoder.encode(obs_missing)
        
        # Missing channel should increase uncertainty
        assert state_missing.uncertainty > state_complete.uncertainty, \
            f"Missing channel should increase R: complete={state_complete.uncertainty}, missing={state_missing.uncertainty}"
        
        # Track missing channels
        assert "nonverbal_signal" in state_missing.missing_channels

    def test_multiple_missing_channels_higher_penalty(self):
        """
        Assertion: More missing channels → higher uncertainty penalty.
        """
        from cognitive_engine.engine import Encoder
        
        profile = PersonProfile(person_id="test2", importance=0.5)
        
        # One missing
        encoder1 = Encoder(profile=profile)
        obs_one_missing = Observation(
            language_content="Test",
            language_structure=None,
            emotional_tone=0.5,
            behavior_signal=0.5,
            nonverbal_signal=0.5,
            consistency=0.5,
        )
        state_one, _ = encoder1.encode(obs_one_missing)
        
        # Reset for two missing
        encoder2 = Encoder(profile=profile)
        obs_two_missing = Observation(
            language_content="Test",
            language_structure=None,
            emotional_tone=None,
            behavior_signal=0.5,
            nonverbal_signal=0.5,
            consistency=0.5,
        )
        state_two, _ = encoder2.encode(obs_two_missing)
        
        # Two missing should have higher uncertainty than one missing
        assert state_two.uncertainty > state_one.uncertainty, \
            f"Two missing should > one missing: one={state_one.uncertainty}, two={state_two.uncertainty}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
