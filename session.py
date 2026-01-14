import uuid
from typing import Optional, Dict, Any

# In-memory store for sessions (replace with Redis in production for scaling/multi-instance)
_sessions: Dict[str, Dict[str, Any]] = {}

class SessionManager:
    """
    Manages per-user session state.
    - Generates a unique session_id
    - Stores route, location, conversation history, etc.
    - Easy to extend for Redis later
    """
    def __init__(self, session_id: Optional[str] = None):
        if session_id and session_id in _sessions:
            self.session_id = session_id
            self.data = _sessions[session_id]
        else:
            self.session_id = str(uuid.uuid4())
            self.data = {
                "current_route": "Barlastone → Town",  # Default/fallback
                "origin": None,
                "destination": None,
                "current_road": None,
                "last_road": None,
                "user_location": None,  # (lat, lng) from GPS
                "conversation_history": []  # Optional: list of {"user": text, "assistant": text}
            }
            _sessions[self.session_id] = self.data

    # Helper methods – add more as needed
    def update_route(self, origin: str, destination: str):
        self.data["origin"] = origin
        self.data["destination"] = destination
        self.data["current_route"] = f"{origin} → {destination}"

    def update_location(self, lat: float, lng: float):
        self.data["user_location"] = (lat, lng)

    def set_current_road(self, road: str):
        self.data["last_road"] = self.data["current_road"]
        self.data["current_road"] = road

    # Getters
    @property
    def route(self) -> str:
        return self.data["current_route"]

    @property
    def road(self) -> Optional[str]:
        return self.data["current_road"] or self.data["last_road"]

    def to_dict(self) -> Dict[str, Any]:
        """Useful for logging or debugging"""
        return {"session_id": self.session_id, **self.data}