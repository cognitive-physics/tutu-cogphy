import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import rel_entr


@dataclass
class CognitiveState:
    """Structured estimate of current cognitive state."""
    rho_0: np.ndarray
    H_0: np.ndarray
    mu: float
    sigma_star: np.ndarray
    uncertainty: float = 1.0
    missing_channels: List[str] = field(default_factory=list)  # Track missing channels for R penalty


@dataclass
class Observation:
    """Single observation frame."""
    language_content: str
    language_structure: Optional[float] = None  # None = missing channel
    emotional_tone: Optional[float] = None
    behavior_signal: Optional[float] = None
    nonverbal_signal: Optional[float] = None
    consistency: Optional[float] = None
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
    h_history: List[float] = field(default_factory=list)  # Track H_0 history for initialization
    sigma_history: List[float] = field(default_factory=list)  # Track sigma_star history

    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "importance": self.importance,
            "last_mu": self.last_mu,
            "last_rho": self.last_rho,
            "last_uncertainty": self.last_uncertainty,
            "session_count": self.session_count,
            "converged_history": self.converged_history,
            "h_history": self.h_history,
            "sigma_history": self.sigma_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PersonProfile":
        p = cls(person_id=data["person_id"], importance=data["importance"])
        p.last_mu = data.get("last_mu", 0.5)
        p.last_rho = data.get("last_rho", 0.5)
        p.last_uncertainty = data.get("last_uncertainty", 1.0)
        p.session_count = data.get("session_count", 0)
        p.converged_history = data.get("converged_history", [])
        p.h_history = data.get("h_history", [])
        p.sigma_history = data.get("sigma_history", [])
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
    MISSING_CHANNEL_PENALTY = 0.3  # Increase uncertainty per missing channel

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

    def _init_h0_from_history(self) -> float:
        """Initialize H_0 from profile's historical H values (entropy of attention)."""
        if self.profile and self.profile.h_history:
            # Use robust median of historical H values
            h_median = float(np.median(self.profile.h_history))
            return h_median
        return 0.0

    def _init_sigma_star_from_history(self) -> float:
        """Initialize sigma_star from profile's converged_history (robustness estimate)."""
        if self.profile and self.profile.converged_history and len(self.profile.converged_history) >= 2:
            # Use median absolute deviation (robust spread measure)
            values = np.array(self.profile.converged_history)
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            # Normalize to [0, 1], ensuring non-zero
            return max(0.1, min(mad, 1.0))
        return 1.0

    def _sample_channels(self, obs: Observation) -> Tuple[Dict[str, float], List[str]]:
        """
        Sample channel values, mark missing ones explicitly.
        
        Returns:
            (channels_dict, missing_channels_list)
        """
        channels = {}
        missing = []

        channels["language_content"] = min(len(obs.language_content) / 300, 1.0)
        
        if obs.language_structure is not None:
            channels["language_structure"] = obs.language_structure
        else:
            missing.append("language_structure")
        
        if obs.emotional_tone is not None:
            channels["emotional_tone"] = obs.emotional_tone
        else:
            missing.append("emotional_tone")
        
        if obs.behavior_signal is not None:
            channels["behavior_signal"] = obs.behavior_signal
        else:
            missing.append("behavior_signal")
        
        if obs.nonverbal_signal is not None:
            channels["nonverbal_signal"] = obs.nonverbal_signal
        else:
            missing.append("nonverbal_signal")
        
        if obs.consistency is not None:
            channels["consistency"] = obs.consistency
        else:
            missing.append("consistency")

        return channels, missing

    def _denoise(self, signals: Dict[str, float], raw_consistency: float) -> Tuple[Dict[str, float], float, bool]:
        """Denoise signals, compute initial mu estimate."""
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

        mu_est = 1.0 - raw_consistency if raw_consistency is not None else 0.5
        return denoised_signals, mu_est, has_inconsistency

    def _inverse_solve(self, signals: Dict[str, float], mu_est: float) -> Dict[str, float]:
        """Weighted combination of channels → rho_raw."""
        raw = sum(self.CHANNEL_WEIGHTS[k] * v for k, v in signals.items() if k in self.CHANNEL_WEIGHTS)
        return {"rho_raw": raw, "mu": mu_est}

    def _bayesian_update(self, new_est: Dict, n: int, missing_count: int) -> Tuple[Dict, float]:
        """Update uncertainty with missing channel penalty."""
        if not self.history:
            base_uncertainty = 1.0
        else:
            prev_avg = np.mean([h["rho_raw"] for h in self.history])
            delta = abs(new_est["rho_raw"] - prev_avg)
            session_bonus = 0.1 * (self.profile.session_count if self.profile else 0)
            base_uncertainty = max(0.0, 1.0 / (1.0 + np.sqrt(n) + session_bonus) + delta)

        # Penalize missing channels: increase R explicitly
        missing_penalty = missing_count * self.MISSING_CHANNEL_PENALTY
        uncertainty = min(base_uncertainty + missing_penalty, 1.0)
        return new_est, min(uncertainty, 1.0)

    def _check_convergence(self, uncertainty: float) -> bool:
        if uncertainty < self.convergence_eps:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
        return self.stable_frames >= self.CONVERGENCE_K

    def encode(self, obs: Observation) -> Tuple[CognitiveState, bool]:
        n = len(self.history) + 1
        
        # Sample channels and track missing ones
        signals, missing_channels = self._sample_channels(obs)
        raw_consistency = obs.consistency
        
        signals, mu_est, _ = self._denoise(signals, raw_consistency)
        est = self._inverse_solve(signals, mu_est)
        est, uncertainty = self._bayesian_update(est, n, len(missing_channels))
        converged = self._check_convergence(uncertainty)
        self.history.append({**est, "uncertainty": uncertainty})

        # Initialize H_0 and sigma_star from profile history
        h0_value = self._init_h0_from_history()
        sigma_star_value = self._init_sigma_star_from_history()

        state = CognitiveState(
            rho_0=np.array([est["rho_raw"]]),
            H_0=np.array([h0_value]),  # From profile history, not 0.0
            mu=est["mu"],
            sigma_star=np.array([sigma_star_value]),  # From profile history, not 1.0
            uncertainty=uncertainty,
            missing_channels=missing_channels,
        )
        return state, converged

    def export_profile_update(self, person_id: str, importance: float, converged: bool, state: Optional[CognitiveState] = None) -> PersonProfile:
        last = self.history[-1] if self.history else {}
        p = self.profile or PersonProfile(person_id=person_id, importance=importance)
        p.last_rho = last.get("rho_raw", 0.5)
        p.last_mu = last.get("mu", 0.5)
        p.last_uncertainty = last.get("uncertainty", 1.0)
        p.session_count += 1
        
        # Track H_0 and sigma_star for future initialization
        if state:
            p.h_history.append(float(state.H_0[0]))
            p.sigma_history.append(float(state.sigma_star[0]))
        
        if converged:
            p.converged_history.append(p.last_rho)
        
        return p


@dataclass
class Event:
    description: str
    rho_now: float
    env_pressure: float
    goal_horizon: float = 1.0


class DecisionEngine(ABC):
    """Abstract base for decision making (action selection)."""
    
    @abstractmethod
    def decide(self, state: CognitiveState, event: Event) -> Dict:
        pass


class LegacyDecisionEngine(DecisionEngine):
    """Original 4-choice enumeration (v1)."""

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
            "action_path": None,  # Legacy: no path
            "action_value": best_cost,
            "is_stationary": None,
            "fallback": False,
        }


class VariationalDecisionEngine(DecisionEngine):
    """
    Variational decision engine: solves δΦ/δa = 0 for continuous action paths.
    
    Theory:
    -------
    We model action selection as a functional minimization problem:
    
    Φ[a] = ∫ L(a, ȧ, state) ds
    
    Discretized on N points:
    Φ[a] = smoothness_term + potential_term
    
    smoothness_term = (κ/2) Σ (a_{i+1} - a_i)²    [Corresponds to κ/2·|∇ψ|² in continuous]
    potential_term  = Σ V(a_i, state, event)      [Corresponds to μ₀V term]
    
    The Euler-Lagrange equation δΦ/δa_i = 0 gives the stationary path.
    We solve this via numerical optimization (L-BFGS-B).
    """
    
    def __init__(self, n_steps: int = 12, kappa: float = 1.0, enable_fallback: bool = True):
        """
        Args:
            n_steps: Discretization points in action path (dimension of optimization)
            kappa: Smoothness coefficient (κ in κ/2·|∇ψ|²)
            enable_fallback: If True, fall back to LegacyDecisionEngine on optimization failure
        """
        self.n_steps = n_steps
        self.kappa = kappa
        self.enable_fallback = enable_fallback
        self.legacy_engine = LegacyDecisionEngine() if enable_fallback else None

    def _potential_term(self, action_path: np.ndarray, state: CognitiveState, event: Event) -> float:
        """
        Potential energy: penalizes deviation from stable state and high pressure.
        
        V(a_i) = (a_i - σ*)² + (a_i * env_pressure)²
        
        Encourages a_i → σ* when pressure is low, a_i → 0 when pressure is high.
        """
        sigma_star = float(state.sigma_star[0])
        mu = state.mu
        pressure = event.env_pressure
        
        # Deviation from equilibrium: (a_i - σ*)²
        deviation_term = np.sum((action_path - sigma_star) ** 2)
        
        # Pressure penalty: (a_i * pressure)²
        pressure_term = np.sum((action_path * (1.0 + mu * pressure)) ** 2)
        
        return deviation_term + pressure_term

    def _action_functional(self, action_path: np.ndarray, state: CognitiveState, event: Event) -> float:
        """
        Action functional Φ[a]:
        Φ = (κ/2) Σ (a_{i+1} - a_i)² + Σ V(a_i)
        
        First term: smoothness (kinetic-like, resistance to rapid changes)
        Second term: potential (cost of deviating from equilibrium)
        """
        # Smoothness term: κ/2 * sum((a[i+1]-a[i])²)
        diffs = np.diff(action_path)
        smoothness = (self.kappa / 2.0) * np.sum(diffs ** 2)
        
        # Potential term
        potential = self._potential_term(action_path, state, event)
        
        return smoothness + potential

    def _action_functional_grad(self, action_path: np.ndarray, state: CognitiveState, event: Event) -> np.ndarray:
        """
        Numerical gradient of action functional (discrete Euler-Lagrange LHS).
        
        δΦ/δa_i = κ(2a_i - a_{i-1} - a_{i+1}) + dV/da_i
        """
        n = len(action_path)
        grad = np.zeros(n)
        eps = 1e-7
        
        for i in range(n):
            path_plus = action_path.copy()
            path_plus[i] += eps
            path_minus = action_path.copy()
            path_minus[i] -= eps
            
            f_plus = self._action_functional(path_plus, state, event)
            f_minus = self._action_functional(path_minus, state, event)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad

    def _is_true_minimum(self, action_path: np.ndarray, state: CognitiveState, event: Event, 
                         tolerance: float = 1e-4) -> bool:
        """
        Second-order check: verify solution is a true local minimum.
        
        Perturb solution slightly and verify functional doesn't decrease.
        """
        phi_original = self._action_functional(action_path, state, event)
        
        # Random perturbation
        perturbation = np.random.normal(0, 0.01, size=action_path.shape)
        perturbed = np.clip(action_path + perturbation, 0.0, 1.0)
        
        phi_perturbed = self._action_functional(perturbed, state, event)
        
        # Check multiple random perturbations
        for _ in range(3):
            perturbation = np.random.normal(0, 0.01, size=action_path.shape)
            perturbed = np.clip(action_path + perturbation, 0.0, 1.0)
            phi_perturbed = self._action_functional(perturbed, state, event)
            
            # If any perturbation decreases functional, not a true minimum
            if phi_perturbed < phi_original - tolerance:
                return False
        
        return True

    def decide(self, state: CognitiveState, event: Event) -> Dict:
        """
        Solve variational problem to find optimal action path.
        
        Returns:
            {
                "a_star": scalar (mean or first action),
                "best_action": "variational" (indicator),
                "action_path": optimal path array,
                "action_value": functional value Φ[a*],
                "is_stationary": True if second-order check passes,
                "fallback": False if successful, True if fell back to legacy,
                ...other legacy fields...
            }
        """
        try:
            # Initial guess: linear interpolation from 0 to sigma_star
            sigma_star = float(state.sigma_star[0])
            x0 = np.linspace(0.0, sigma_star, self.n_steps)
            
            # Optimize with L-BFGS-B (supports box constraints [0, 1])
            result = minimize(
                fun=self._action_functional,
                x0=x0,
                args=(state, event),
                method="L-BFGS-B",
                jac=self._action_functional_grad,
                bounds=[(0.0, 1.0)] * self.n_steps,
                options={"maxiter": 200, "ftol": 1e-6},
            )
            
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            
            optimal_path = result.x
            phi_optimal = result.fun
            
            # Second-order check: verify true minimum
            is_stationary = self._is_true_minimum(optimal_path, state, event)
            
            # Map path to a_star (use mean or first value)
            a_star = max(0.0, min(np.mean(optimal_path), 1.0))
            
            return {
                "a_star": a_star,
                "best_action": "variational",
                "goal_horizon": event.goal_horizon,
                "dim1_behavior": event.rho_now,
                "dim2_education": float(state.rho_0[0]) * (1.0 - state.mu),
                "dim3_environment": event.env_pressure,
                "path_cost": float(phi_optimal),
                "action_path": optimal_path.tolist(),
                "action_value": float(phi_optimal),
                "is_stationary": is_stationary,
                "fallback": False,
            }
        
        except Exception as e:
            # Graceful fallback to legacy engine
            if self.enable_fallback and self.legacy_engine:
                return {**self.legacy_engine.decide(state, event), "fallback": True, "fallback_reason": str(e)}
            else:
                raise RuntimeError(f"Variational decision failed and fallback disabled: {e}")


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

    def measure_eta(self, complexity_ratio: float, follow_up_rate: float, paraphrase_acc: float, latency_norm: Optional[float]) -> float:
        """Measure decoding rate η, handling missing data gracefully."""
        r = {
            "complexity_ratio": complexity_ratio,
            "follow_up": 1.0 - follow_up_rate,
            "paraphrase": paraphrase_acc,
        }
        
        # Downweight if latency is missing
        if latency_norm is not None:
            r["latency"] = 1.0 - latency_norm
        else:
            # Missing latency: use neutral value and reduce its weight
            r["latency"] = 0.5
            self.ETA_WEIGHTS["latency"] = 0.05  # Reduced from 0.10
        
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
                feedback.get("latency_norm"),  # Can be None
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

    def __init__(self, profile: Optional[PersonProfile] = None, use_variational: bool = True):
        """
        Args:
            profile: User profile for initialization
            use_variational: If True, use VariationalDecisionEngine; else LegacyDecisionEngine
        """
        self.encoder = Encoder(profile=profile)
        if use_variational:
            self.decision = VariationalDecisionEngine(n_steps=12, kappa=1.0, enable_fallback=True)
        else:
            self.decision = LegacyDecisionEngine()
        self.decoder = Decoder()
        self._last_enc_converged = False
        self._last_state: Optional[CognitiveState] = None

    def run(self, obs: Observation, event: Event, feedback: Optional[Dict] = None) -> Dict:
        state, enc_converged = self.encoder.encode(obs)
        self._last_enc_converged = enc_converged
        self._last_state = state
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
            state=self._last_state,
        )
