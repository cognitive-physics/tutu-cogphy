"""Simple JSON-backed profile storage for the cognitive engine prototype."""

import json
from pathlib import Path
from typing import Optional

from .engine import PersonProfile

PROFILES_PATH = Path("profiles.json")


class ProfileStore:
    def __init__(self):
        self._load()

    def _load(self):
        if PROFILES_PATH.exists():
            raw = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
            self.store = {k: PersonProfile.from_dict(v) for k, v in raw.items()}
        else:
            self.store = {}

    def _save(self):
        PROFILES_PATH.write_text(
            json.dumps(
                {k: v.to_dict() for k, v in self.store.items()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def get(self, person_id: str) -> Optional[PersonProfile]:
        return self.store.get(person_id)

    def save(self, profile: PersonProfile):
        self.store[profile.person_id] = profile
        self._save()

    def delete(self, person_id: str) -> bool:
        if person_id in self.store:
            del self.store[person_id]
            self._save()
            return True
        return False
