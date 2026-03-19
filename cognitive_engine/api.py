"""FastAPI service exposing the cognitive engine prototype."""

from typing import Optional

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

    obs = Observation(
        language_content=req.message,
        language_structure=structure_score,
        emotional_tone=tone_score,
        behavior_signal=user_complexity,
        nonverbal_signal=0.5,
        consistency=user_complexity,
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
