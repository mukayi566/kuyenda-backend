from supabase_client import get_supabase
from typing import Dict, Any, List

supabase = get_supabase()

# Profiles & Preferences
def create_profile(user_id: str, email: str, full_name: str = None, user_type: str = 'rider'):
    data = {
        "id": user_id,
        "email": email,
        "full_name": full_name,
        "user_type": user_type
    }
    return supabase.table("profiles").insert(data).execute()

def get_profile(user_id: str):
    return supabase.table("profiles").select("*").eq("id", user_id).single().execute()

def update_preferences(user_id: str, language: str = None, voice_speed: float = None, alert_level: str = None):
    update_data = {}
    if language: update_data["language"] = language
    if voice_speed is not None: update_data["voice_speed"] = voice_speed
    if alert_level: update_data["alert_level"] = alert_level
    return supabase.table("profiles").update(update_data).eq("id", user_id).execute()

# Trips
def create_trip(rider_id: str, origin: Dict[str, Any], dest: Dict[str, Any], estimated_fare: float):
    data = {
        "rider_id": rider_id,
        "origin_name": origin["name"],
        "origin_lat": origin["lat"],
        "origin_lng": origin["lng"],
        "dest_name": dest["name"],
        "dest_lat": dest["lat"],
        "dest_lng": dest["lng"],
        "estimated_fare": estimated_fare,
        "status": "pending"
    }
    return supabase.table("trips").insert(data).execute()

def update_trip_status(trip_id: str, status: str, driver_id: str = None):
    update_data = {"status": status}
    if driver_id:
        update_data["driver_id"] = driver_id
    return supabase.table("trips").update(update_data).eq("id", trip_id).execute()

def get_trip_history(user_id: str):
    return supabase.table("trips") \
        .select("*, profiles!trips_rider_id_fkey(full_name)") \
        .or_(f"rider_id.eq.{user_id},driver_id.eq.{user_id}") \
        .order("created_at", desc=True) \
        .execute()

# Saved Routes
def add_saved_route(user_id: str, name: str, origin: Dict[str, Any], dest: Dict[str, Any]):
    data = {
        "user_id": user_id,
        "name": name,
        "origin_name": origin["name"],
        "origin_lat": origin["lat"],
        "origin_lng": origin["lng"],
        "dest_name": dest["name"],
        "dest_lat": dest["lat"],
        "dest_lng": dest["lng"]
    }
    return supabase.table("saved_routes").insert(data).execute()

def get_saved_routes(user_id: str):
    return supabase.table("saved_routes").select("*").eq("user_id", user_id).execute()

# Voice Command Logs
def log_voice_command(user_id: str, text: str, response: str = None, intent: str = None):
    data = {
        "user_id": user_id,
        "command_text": text,
        "ai_response": response,
        "intent": intent
    }
    return supabase.table("voice_logs").insert(data).execute()

# Traffic Statistics
def log_traffic_stat(location: str, lat: float, lng: float, delay: int, level: str):
    data = {
        "location_name": location,
        "lat": lat,
        "lng": lng,
        "avg_delay_minutes": delay,
        "congestion_level": level
    }
    return supabase.table("traffic_stats").insert(data).execute()

def get_aggregated_traffic():
    # Returns recent traffic snapshots
    return supabase.table("traffic_stats").select("*").order("measured_at", desc=True).limit(50).execute()

# Routes (Path Details)
def save_route(trip_id: str, geometry: Dict[str, Any], duration: int, distance: int):
    data = {
        "trip_id": trip_id,
        "geometry": geometry,
        "duration_sec": duration,
        "distance_meters": distance
    }
    return supabase.table("routes").insert(data).execute()

# Payments
def record_payment(trip_id: str, amount: float, method: str, metadata: Dict[str, Any] = None):
    data = {
        "trip_id": trip_id,
        "amount": amount,
        "payment_method": method,
        "metadata": metadata,
        "status": "pending"
    }
    return supabase.table("payments").insert(data).execute()

# Reviews
def submit_review(trip_id: str, reviewer_id: str, reviewee_id: str, rating: int, comment: str):
    data = {
        "trip_id": trip_id,
        "reviewer_id": reviewer_id,
        "reviewee_id": reviewee_id,
        "rating": rating,
        "comment": comment
    }
    return supabase.table("reviews").insert(data).execute()

# Analytics
def log_analytics_event(user_id: str, event_type: str, event_data: Dict[str, Any] = None):
    data = {
        "user_id": user_id,
        "event_type": event_type,
        "event_data": event_data
    }
    return supabase.table("analytics_events").insert(data).execute()

# Incidents
def report_incident(user_id: str, incident_type: str, lat: float, lng: float, description: str = None):
    data = {
        "user_id": user_id,
        "type": incident_type,
        "latitude": lat,
        "longitude": lng,
        "description": description
    }
    return supabase.table("incidents").insert(data).execute()
