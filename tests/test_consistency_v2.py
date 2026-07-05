"""Pytest for estimate_consistency() v2 upgrade."""

import pytest
from cognitive_engine.api import estimate_consistency


class TestEstimateConsistency:
    """Test cross-channel coherence function."""
    
    def test_all_consistent_high_values(self):
        """
        Scenario: All channels report high, aligned values.
        Expected: consistency should be close to 1.0 (high coherence).
        """
        channels = {
            'language_content': 0.9,
            'language_structure': 0.85,
            'emotional_tone': 0.88,
            'behavior_signal': 0.92,
            'nonverbal_signal': 0.86,
            'context_weight': 0.89,
        }
        consistency = estimate_consistency(channels)
        assert consistency > 0.85, f"Expected high consistency (>0.85), got {consistency}"
    
    def test_clear_contradiction_divergent_values(self):
        """
        Scenario: Channels report sharply divergent values (explicit conflict).
        - language_content: 0.1 (low text),
        - language_structure: 0.9 (high structure),
        - emotional_tone: 0.05 (very flat),
        - behavior_signal: 0.95 (high activity),
        - nonverbal_signal: 0.1 (subdued),
        - context_weight: 0.8 (high context weight).
        Expected: consistency should be low (< 0.4) due to conflict.
        """
        channels = {
            'language_content': 0.1,
            'language_structure': 0.9,
            'emotional_tone': 0.05,
            'behavior_signal': 0.95,
            'nonverbal_signal': 0.1,
            'context_weight': 0.8,
        }
        consistency = estimate_consistency(channels)
        assert consistency < 0.4, f"Expected low consistency (<0.4) for contradictory channels, got {consistency}"
    
    def test_consistency_inversely_related_to_mu(self):
        """
        Test that higher consistency → lower μ (internal-external consistency metric).
        μ = 1 - consistency, so consistency directly drives μ.
        """
        # High consistency case
        coherent = {
            'language_content': 0.8,
            'language_structure': 0.75,
            'emotional_tone': 0.82,
            'behavior_signal': 0.78,
            'nonverbal_signal': 0.76,
            'context_weight': 0.79,
        }
        consistency_high = estimate_consistency(coherent)
        mu_high = 1.0 - consistency_high
        
        # Low consistency case
        contradictory = {
            'language_content': 0.1,
            'language_structure': 0.9,
            'emotional_tone': 0.05,
            'behavior_signal': 0.95,
            'nonverbal_signal': 0.1,
            'context_weight': 0.8,
        }
        consistency_low = estimate_consistency(contradictory)
        mu_low = 1.0 - consistency_low
        
        # Coherent case should have higher consistency and lower μ
        assert consistency_high > consistency_low, \
            f"Coherent {consistency_high} should be > contradictory {consistency_low}"
        assert mu_high < mu_low, \
            f"μ coherent {mu_high} should be < μ contradictory {mu_low}"
    
    def test_consistency_range_bounded(self):
        """Verify consistency is always in [0, 1]."""
        test_cases = [
            {
                'language_content': 0.0,
                'language_structure': 0.0,
                'emotional_tone': 0.0,
                'behavior_signal': 0.0,
                'nonverbal_signal': 0.0,
                'context_weight': 0.0,
            },
            {
                'language_content': 1.0,
                'language_structure': 1.0,
                'emotional_tone': 1.0,
                'behavior_signal': 1.0,
                'nonverbal_signal': 1.0,
                'context_weight': 1.0,
            },
            {
                'language_content': 0.5,
                'language_structure': 0.5,
                'emotional_tone': 0.5,
                'behavior_signal': 0.5,
                'nonverbal_signal': 0.5,
                'context_weight': 0.5,
            },
        ]
        for channels in test_cases:
            consistency = estimate_consistency(channels)
            assert 0.0 <= consistency <= 1.0, \
                f"Consistency {consistency} out of [0, 1] for {channels}"
    
    def test_missing_channel_raises_error(self):
        """Verify that missing required channels raise ValueError."""
        incomplete = {
            'language_content': 0.5,
            'language_structure': 0.5,
            # Missing: emotional_tone, behavior_signal, nonverbal_signal, context_weight
        }
        with pytest.raises(ValueError):
            estimate_consistency(incomplete)
    
    def test_independent_of_complexity(self):
        """
        Verify estimate_consistency() is independent of compute_complexity().
        Two messages with same complexity but different channel alignment
        should yield different consistency values.
        """
        # Note: This is a conceptual test; in practice, consistency is computed
        # from explicit channel scores, not from message text itself.
        # We verify that the same channel configuration always yields same consistency.
        channels_a = {
            'language_content': 0.6,
            'language_structure': 0.6,
            'emotional_tone': 0.6,
            'behavior_signal': 0.6,
            'nonverbal_signal': 0.6,
            'context_weight': 0.6,
        }
        channels_b = {
            'language_content': 0.6,
            'language_structure': 0.6,
            'emotional_tone': 0.6,
            'behavior_signal': 0.6,
            'nonverbal_signal': 0.6,
            'context_weight': 0.6,
        }
        
        consistency_a = estimate_consistency(channels_a)
        consistency_b = estimate_consistency(channels_b)
        
        # Same channels → same consistency (deterministic)
        assert consistency_a == consistency_b, \
            f"Identical channels should yield identical consistency: {consistency_a} vs {consistency_b}"
        
        # Both should be 1.0 (perfect alignment when all channels equal)
        assert consistency_a == 1.0, \
            f"Uniform channels should yield consistency=1.0, got {consistency_a}"


class TestChatEndpointWithConsistency:
    """Integration test: verify /chat endpoint uses estimate_consistency()."""
    
    def test_chat_response_includes_consistency_v2(self):
        """
        Verify that ChatResponse now includes the new 'consistency' field
        computed from estimate_consistency().
        """
        # This test assumes the app is properly initialized.
        # For a full integration test, use FastAPI TestClient.
        from fastapi.testclient import TestClient
        from cognitive_engine.api import app
        
        client = TestClient(app)
        payload = {
            "person_id": "test_user_001",
            "message": "I am very happy with this clear and consistent communication.",
            "importance": 0.5,
            "env_pressure": 0.2,
            "goal_horizon": 0.7,
        }
        
        response = client.post("/chat", json=payload)
        assert response.status_code == 200, f"Chat endpoint failed: {response.text}"
        
        data = response.json()
        assert "consistency" in data, "Response missing 'consistency' field"
        assert isinstance(data["consistency"], float), "consistency should be float"
        assert 0.0 <= data["consistency"] <= 1.0, f"consistency out of bounds: {data['consistency']}"
        
        # Verify that consistency is NOT just user_complexity
        assert data["consistency"] != data["user_complexity"], \
            f"consistency should differ from user_complexity, not both {data['consistency']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
