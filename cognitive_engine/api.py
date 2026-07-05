"""FastAPI service exposing the cognitive engine prototype."""

from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .complexity import compute_complexity, complexity_ratio_between
from .engine import CognitiveEngine, Event, Observation, PersonProfile
from .store import ProfileStore

app = FastAPI(title="Cognitive Engine API", version="1.0.0")
profile_store = ProfileStore()
_engines = {}


def get_engine(person_id: str, importance: float) -> CognitiveEngine:
    if person_id not in _engines:
        profile = profile_store.get(person_id)
        if profile is None:
            profile = PersonProfile(person_id=person_id, importance=importance)
        _engines[person_id] = CognitiveEngine(profile=profile)
    return _engines[person_id]


def estimate_consistency(channels: dict) -> float:
    """
    Compute cross-channel coherence from six signal channels.
    
    Measures consistency = 1 - normalized_divergence across channels.
    Higher values (→1.0) indicate strong alignment across language_content,
    language_structure, emotional_tone, behavior_signal, nonverbal_signal,
    and context_weight. Lower values (→0.0) indicate contradiction/conflict.
    
    Args:
        channels: dict with keys {
            'language_content': float in [0, 1],
            'language_structure': float in [0, 1],
            'emotional_tone': float in [0, 1],
            'behavior_signal': float in [0, 1],
            'nonverbal_signal': float in [0, 1],
            'context_weight': float in [0, 1],
        }
    
    Returns:
        consistency: float in [0, 1]
            0 = maximum divergence (highly contradictory),
            1 = maximum alignment (all channels coherent).
            Unit: dimensionless coherence score.
    """
    required_keys = {
        'language_content', 'language_structure', 'emotional_tone',
        'behavior_signal', 'nonverbal_signal', 'context_weight'
    }
    if not required_keys.issubset(set(channels.keys())):
        raise ValueError(f"Missing channels. Required: {required_keys}")
    
    # Extract channel values and clip to [0, 1]
    values = np.array([
        np.clip(channels['language_content'], 0.0, 1.0),
        np.clip(channels['language_structure'], 0.0, 1.0),
        np.clip(channels['emotional_tone'], 0.0, 1.0),
        np.clip(channels['behavior_signal'], 0.0, 1.0),
        np.clip(channels['nonverbal_signal'], 0.0, 1.0),
        np.clip(channels['context_weight'], 0.0, 1.0),
    ], dtype=np.float64)
    
    # Normalize to probability distribution
    values_sum = values.sum() + 1e-12
    values_norm = values / values_sum
    
    # Reference: uniform distribution (perfect consistency would have all channels equal)
    uniform = np.ones_like(values_norm) / len(values_norm)
    
    # Pairwise divergence: measure spread as variance normalized by mean
    mean_val = values.mean()
    if mean_val < 1e-12:
        return 1.0  # All zeros → treat as consistent (neutral state)
    
    variance = np.sum((values - mean_val) ** 2) / len(values)
    normalized_divergence = variance / (mean_val * mean_val + 1e-12)
    
    # Cap divergence at 1.0 to keep consistency in [0, 1]
    normalized_divergence = min(normalized_divergence, 1.0)
    
    # Consistency = 1 - normalized_divergence
    consistency = 1.0 - normalized_divergence
    return float(np.clip(consistency, 0.0, 1.0))


class ChatRequest(BaseModel):
    person_id: str = Field(..., description="Unique user id")
    message: str = Field(..., description="User input message")
    importance: float = Field(0.5, ge=0.0, le=1.0)
    env_pressure: float = Field(0.2, ge=0.0, le=1.0)
    goal_horizon: float = Field(0.7, ge=0.0, le=1.0)
    last_ai_output: Optional[str] = Field(None, description="Previous AI output")


class ChatResponse(BaseModel):
    output_mode: str
    compression_rate: float
    eta: float
    best_action: str
    a_star: float
    mu: float
    uncertainty: float
    system_stable: bool
    user_complexity: float
    consistency: Optional[float] = Field(None, description="v2: true cross-channel coherence [0,1]")


class FeedbackRequest(BaseModel):
    person_id: str
    user_reply: str
    ai_output: str
    follow_up_rate: float = Field(0.2, ge=0.0, le=1.0)
    paraphrase_acc: float = Field(0.8, ge=0.0, le=1.0)
    latency_norm: float = Field(0.2, ge=0.0, le=1.0)
    save_profile: bool = True


class FeedbackResponse(BaseModel):
    eta: float
    updated_compression: float
    converged: bool


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    engine = get_engine(req.person_id, req.importance)

    user_complexity = compute_complexity(req.message)
    structure_score = min(len(req.message.split(".")) / 5.0, 1.0)
    tone_score = 0.3

    # V2: Compute true cross-channel consistency instead of using complexity
    channels = {
        'language_content': min(len(req.message) / 300.0, 1.0),
        'language_structure': structure_score,
        'emotional_tone': tone_score,
        'behavior_signal': user_complexity,
        'nonverbal_signal': 0.5,
        'context_weight': 1.0,
    }
    true_consistency = estimate_consistency(channels)

    obs = Observation(
        language_content=req.message,
        language_structure=structure_score,
        emotional_tone=tone_score,
        behavior_signal=user_complexity,
        nonverbal_signal=0.5,
        consistency=true_consistency,  # V2: Use true consistency instead of user_complexity
    )

    event = Event(
        description=req.message[:50],
        rho_now=user_complexity,
        env_pressure=req.env_pressure,
        goal_horizon=req.goal_horizon,
    )

    feedback = None
    if req.last_ai_output:
        cr = complexity_ratio_between(req.last_ai_output, req.message)
        feedback = {
            "complexity_ratio": cr,
            "follow_up_rate": 0.2,
            "paraphrase_acc": 0.8,
            "latency_norm": 0.2,
        }

    result = engine.run(obs, event, feedback)

    return ChatResponse(
        output_mode=result["output"]["output_mode"],
        compression_rate=result["output"]["compression_rate"],
        eta=result["output"]["eta"],
        best_action=result["decision"]["best_action"],
        a_star=result["decision"]["a_star"],
        mu=result["state"].mu,
        uncertainty=result["state"].uncertainty,
        system_stable=result["system_stable"],
        user_complexity=user_complexity,
        consistency=true_consistency,  # V2: New field
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    engine = get_engine(req.person_id, 0.5)

    cr = complexity_ratio_between(req.ai_output, req.user_reply)
    eta = engine.decoder.measure_eta(
        cr,
        req.follow_up_rate,
        req.paraphrase_acc,
        req.latency_norm,
    )
    engine.decoder.eta_history.append(eta)
    engine.decoder._adapt_compression(eta)

    if req.save_profile:
        profile = engine.save_profile(req.person_id)
        profile_store.save(profile)

    return FeedbackResponse(
        eta=eta,
        updated_compression=engine.decoder.c,
        converged=engine.decoder.is_converged(),
    )


@app.get("/profile/{person_id}")
def get_profile(person_id: str):
    profile = profile_store.get(person_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile.to_dict()


@app.delete("/profile/{person_id}")
def delete_profile(person_id: str):
    if person_id in _engines:
        del _engines[person_id]
    deleted = profile_store.delete(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"status": "deleted", "person_id": person_id}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
