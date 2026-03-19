import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import rel_entr


@dataclass
class CognitiveState:
    """Structured estimate of current cognitive state."""
    rho_0: np.ndarray
    H_0: np.ndarray
    mu: float
    sigma_star: np.ndarray
    uncertainty: float = 1.0


@dataclass
class Observation:
    """Single observation frame."""
    language_content: str
    language_structure: float
    emotional_tone: float
    behavior_signal: float
    nonverbal_signal: float
    consistency: float
    context_weight: float = 1.0


@dataclass
class PersonProfile:
    """Cross-session persistent user profile."""
    person_id: str
    importance: float
    last_mu: float = 0.5
    last_rho: float = 0.5
    last_uncertainty: float = 1.0
    session_count: int = 0
    converged_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "importance": self.importance,
            "last_mu": self.last_mu,
            "last_rho": self.last_rho,
            "last_uncertainty": self.last_uncertainty,
            "session_count": self.session_count,
            "converged_history": self.converged_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PersonProfile":
        p = cls(person_id=data["person_id"], importance=data["importance"])
        p.last_mu = data["last_mu"]
        p.last_rho = data["last_rho"]
        p.last_uncertainty = data["last_uncertainty"]
        p.session_count = data["session_count"]
        p.converged_history = data["converged_history"]
        return p


class Encoder:
    """Input layer: raw observations -> structured parameter estimate."""

    CHANNEL_WEIGHTS = {
        "language_content": 0.15,
        "language_structure": 0.25,
        "emotional_tone": 0.10,
        "behavior_signal": 0.20,
        "nonverbal_signal": 0.10,
        "consistency": 0.20,
    }
    NOISE_THRESHOLD = 0.3
    CONVERGENCE_K = 3

    def __init__(self, profile: Optional[PersonProfile] = None):
        self.profile = profile
        if profile:
            self.convergence_eps = 0.08 * (1.0 - profile.importance * 0.6)
        else:
            self.convergence_eps = 0.08

        if profile and profile.session_count > 0:
            self.history: List[Dict] = [{
                "rho_raw": profile.last_rho,
                "mu": profile.last_mu,
                "uncertainty": profile.last_uncertainty,
            }]
        else:
            self.history = []

        self.stable_frames = 0

    def _sample_channels(self, obs: Observation) -> Dict[str, float]:
        return {
            "language_content": min(len(obs.language_content) / 300, 1.0),
            "language_structure": obs.language_structure,
            "emotional_tone": obs.emotional_tone,
            "behavior_signal": obs.behavior_signal,
            "nonverbal_signal": obs.nonverbal_signal,
            "consistency": obs.consistency,
        }

    def _denoise(self, signals: Dict[str, float], raw_consistency: float) -> Tuple[Dict[str, float], float, bool]:
        original_vals = np.array(list(signals.values()), dtype=float)
        vals_norm = (original_vals + 1e-9) / (original_vals + 1e-9).sum()
        mean_val = np.full_like(vals_norm, 1.0 / len(vals_norm))
        dkl = float(np.sum(rel_entr(vals_norm, mean_val)))
        has_inconsistency = dkl > self.NOISE_THRESHOLD

        if has_inconsistency:
            adj = np.exp(-0.5 * np.abs(original_vals - original_vals.mean()))
            denoised_vals = np.clip(original_vals * adj, 0.0, 1.0)
            denoised_signals = dict(zip(signals.keys(), denoised_vals))
        else:
            denoised_signals = signals

        mu_est = 1.0 - raw_consistency
        return denoised_signals, mu_est, has_inconsistency

    def _inverse_solve(self, signals: Dict[str, float], mu_est: float) -> Dict[str, float]:
        raw = sum(self.CHANNEL_WEIGHTS[k] * v for k, v in signals.items())
        return {"rho_raw": raw, "mu": mu_est}

    def _bayesian_update(self, new_est: Dict, n: int) -> Tuple[Dict, float]:
        if not self.history:
            return new_est, 1.0
        prev_avg = np.mean([h["rho_raw"] for h in self.history])
        delta = abs(new_est["rho_raw"] - prev_avg)
        session_bonus = 0.1 * (self.profile.session_count if self.profile else 0)
        uncertainty = max(0.0, 1.0 / (1.0 + np.sqrt(n) + session_bonus) + delta)
        return new_est, min(uncertainty, 1.0)

    def _check_convergence(self, uncertainty: float) -> bool:
        if uncertainty < self.convergence_eps:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
        return self.stable_frames >= self.CONVERGENCE_K

    def encode(self, obs: Observation) -> Tuple[CognitiveState, bool]:
        n = len(self.history) + 1
        raw_consistency = obs.consistency
        signals = self._sample_channels(obs)
        signals, mu_est, _ = self._denoise(signals, raw_consistency)
        est = self._inverse_solve(signals, mu_est)
        est, uncertainty = self._bayesian_update(est, n)
        converged = self._check_convergence(uncertainty)
        self.history.append({**est, "uncertainty": uncertainty})

        state = CognitiveState(
            rho_0=np.array([est["rho_raw"]]),
            H_0=np.array([0.0]),
            mu=est["mu"],
            sigma_star=np.array([1.0]),
            uncertainty=uncertainty,
        )
        return state, converged

    def export_profile_update(self, person_id: str, importance: float, converged: bool) -> PersonProfile:
        last = self.history[-1] if self.history else {}
        p = self.profile or PersonProfile(person_id=person_id, importance=importance)
        p.last_rho = last.get("rho_raw", 0.5)
        p.last_mu = last.get("mu", 0.5)
        p.last_uncertainty = last.get("uncertainty", 1.0)
        p.session_count += 1
        if converged:
            p.converged_history.append(p.last_rho)
        return p


@dataclass
class Event:
    description: str
    rho_now: float
    env_pressure: float
    goal_horizon: float = 1.0


class DecisionEngine:
    """Middle layer: structured state + event -> least-dissipation action."""

    ACTION_CANDIDATES = {
        "immediate_relief": {"short_cost": 0.1, "long_cost": 0.8},
        "invest_now": {"short_cost": 0.7, "long_cost": 0.2},
        "balanced": {"short_cost": 0.4, "long_cost": 0.4},
        "defer": {"short_cost": 0.2, "long_cost": 0.5},
    }

    def _path_integral(self, action: str, horizon: float, env_pressure: float, mu: float, rho: float) -> float:
        a = self.ACTION_CANDIDATES[action]
        short_cost_adj = min(
            a["short_cost"] * (1.0 + env_pressure) * (1.0 + mu) / (rho + 0.1),
            2.0,
        )
        return (1.0 - horizon) * short_cost_adj + horizon * a["long_cost"]

    def decide(self, state: CognitiveState, event: Event) -> Dict:
        mu = state.mu
        rho = float(state.rho_0[0])
        best_action = min(
            self.ACTION_CANDIDATES.keys(),
            key=lambda a: self._path_integral(a, event.goal_horizon, event.env_pressure, mu, rho),
        )
        best_cost = self._path_integral(best_action, event.goal_horizon, event.env_pressure, mu, rho)
        a_star = max(0.0, 1.0 - best_cost / 2.0)
        return {
            "a_star": a_star,
            "best_action": best_action,
            "goal_horizon": event.goal_horizon,
            "dim1_behavior": event.rho_now,
            "dim2_education": rho * (1.0 - mu),
            "dim3_environment": event.env_pressure,
            "path_cost": best_cost,
        }


class Decoder:
    """Output layer: adaptive compression based on estimated decode rate."""

    ETA_WEIGHTS = {
        "complexity_ratio": 0.50,
        "follow_up": 0.20,
        "paraphrase": 0.20,
        "latency": 0.10,
    }
    ETA_TARGET = 0.8
    ALPHA = 0.15
    EPS = 0.04

    def __init__(self, init_compression: float = 1.0):
        self.c = init_compression
        self.eta_history: List[float] = []

    def measure_eta(self, complexity_ratio: float, follow_up_rate: float, paraphrase_acc: float, latency_norm: float) -> float:
        r = {
            "complexity_ratio": complexity_ratio,
            "follow_up": 1.0 - follow_up_rate,
            "paraphrase": paraphrase_acc,
            "latency": 1.0 - latency_norm,
        }
        return float(np.clip(sum(self.ETA_WEIGHTS[k] * r[k] for k in r), 0.0, 1.0))

    def _adapt_compression(self, eta: float) -> str:
        self.c = float(np.clip(self.c + self.ALPHA * (eta - self.ETA_TARGET), 0.05, 1.0))
        if self.c > 0.75:
            return "compressed"
        if self.c > 0.40:
            return "metaphor"
        return "slow_expand"

    def is_converged(self) -> bool:
        if len(self.eta_history) < 2:
            return False
        return abs(self.eta_history[-1] - self.eta_history[-2]) < self.EPS

    def decode(self, a_star: float, feedback: Optional[Dict] = None) -> Dict:
        eta = self.ETA_TARGET
        if feedback:
            eta = self.measure_eta(
                feedback.get("complexity_ratio", 0.8),
                feedback.get("follow_up_rate", 0.2),
                feedback.get("paraphrase_acc", 0.8),
                feedback.get("latency_norm", 0.2),
            )
        self.eta_history.append(eta)
        mode = self._adapt_compression(eta)
        return {
            "a_star": a_star,
            "output_mode": mode,
            "compression_rate": self.c,
            "eta": eta,
            "converged": self.is_converged(),
        }


class CognitiveEngine:
    """Complete closed loop: Encoder -> DecisionEngine -> Decoder."""

    def __init__(self, profile: Optional[PersonProfile] = None):
        self.encoder = Encoder(profile=profile)
        self.decision = DecisionEngine()
        self.decoder = Decoder()
        self._last_enc_converged = False

    def run(self, obs: Observation, event: Event, feedback: Optional[Dict] = None) -> Dict:
        state, enc_converged = self.encoder.encode(obs)
        self._last_enc_converged = enc_converged
        decision = self.decision.decide(state, event)
        output = self.decoder.decode(decision["a_star"], feedback)
        return {
            "state": state,
            "decision": decision,
            "output": output,
            "encoder_converged": enc_converged,
            "decoder_converged": output["converged"],
            "system_stable": enc_converged and output["converged"],
        }

    def save_profile(self, person_id: str, importance: float = 0.5) -> PersonProfile:
        return self.encoder.export_profile_update(
            person_id=person_id,
            importance=importance,
            converged=self._last_enc_converged,
        )
