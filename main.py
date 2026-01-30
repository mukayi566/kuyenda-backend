# main.py (UPDATED + CLEANED)
# - Fixes duplicated /ws/voice handler
# - Fixes parse order bug (Mapbox uses lng,lat)
# - Keeps your Supabase login/signup but returns your own JWT (SECRET_KEY)
# - Adds /predict-traffic + /traffic-status + /test-ml-model
# - Keeps Lusaka-only geocode search (Local + Mapbox with strict bbox)
# - Removes broken/unreachable code sections

import os
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import httpx
import joblib
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Depends,
    status,
    Response,
    Query,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Body,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# ----------------------------
# Local imports
# ----------------------------
import custom_transformers
from config import SECRET_KEY, ALGORITHM, DEEPGRAM_API_KEY, MAPBOX_TOKEN
from session import SessionManager
from traffic import TrafficPredictor
from database import (
    create_profile,
    get_profile,
    create_trip,
    update_trip_status,
    get_trip_history,
    save_route,
    record_payment,
    submit_review,
    log_analytics_event,
    update_preferences,
    add_saved_route,
    get_saved_routes,
    get_aggregated_traffic,
    log_voice_command,
    report_incident,
)

load_dotenv()

# ----------------------------
# App + Rate Limiter
# ----------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Kuyenda Backend Proxy")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Static files (avatars etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"DEBUG: Incoming {request.method} request to {request.url}")
    resp = await call_next(request)
    print(f"DEBUG: Response status: {resp.status_code}")
    return resp


# ----------------------------
# ML Model Loading
# ----------------------------
# Fix for pickle/joblib loading: ensure transformer class exists
import __main__
__main__.BehavioralFeatureExtractor = custom_transformers.BehavioralFeatureExtractor

MODEL_PATH = "models/traffic_model_v2.pkl"
traffic_predictor = TrafficPredictor(MODEL_PATH)

try:
    traffic_predictor.load()
    print("✅ Traffic Predictor Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading traffic predictor: {e}")


# ----------------------------
# Validation & Utilities
# ----------------------------
ZAMBIA_BOUNDS = {"min_lat": -18.5, "max_lat": -8.0, "min_lng": 21.5, "max_lng": 34.0}

def validate_coords(lat: float, lng: float) -> bool:
    """Validate coordinate ranges (and lightly validate Zambia bounds)."""
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        return False
    if not (ZAMBIA_BOUNDS["min_lat"] <= lat <= ZAMBIA_BOUNDS["max_lat"]):
        print(f"⚠️ Suspicious lat out of Zambia bounds: {lat}, {lng}")
    if not (ZAMBIA_BOUNDS["min_lng"] <= lng <= ZAMBIA_BOUNDS["max_lng"]):
        print(f"⚠️ Suspicious lng out of Zambia bounds: {lat}, {lng}")
    return True

def parse_lng_lat(coord_str: str):
    """
    Parse 'lng,lat' string (Mapbox standard).
    Returns (lng, lat).
    """
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError
        lng = float(parts[0])
        lat = float(parts[1])
        return lng, lat
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid coordinate format. Use 'lng,lat'")


# ----------------------------
# Auth
# ----------------------------
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request, token: Optional[str] = Query(None)):
    """
    Validate JWT for HTTP requests.
    Accepts token from Authorization header OR query param token=...
    """
    auth_header = request.headers.get("Authorization") if hasattr(request, "headers") else None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise JWTError
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_ws_user(ws: WebSocket, token: Optional[str] = None):
    """
    Validate JWT for WebSocket connections.
    Accepts token from query param.
    """
    # Try to get token from headers if available
    auth_header = ws.headers.get("Authorization") if hasattr(ws, "headers") else None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise JWTError
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ----------------------------
# Pydantic Models
# ----------------------------
class UserLogin(BaseModel):
    email: str
    password: str

class UserSignup(BaseModel):
    email: str
    password: str
    full_name: str
    user_type: str = "rider"

class TripCreate(BaseModel):
    origin: dict
    destination: dict
    estimated_fare: float

class PaymentCreate(BaseModel):
    trip_id: str
    amount: float
    method: str
    metadata: Optional[dict] = None

class ReviewCreate(BaseModel):
    trip_id: str
    reviewee_id: str
    rating: int
    comment: str

class AnalyticsEvent(BaseModel):
    event_type: str
    event_data: Optional[dict] = None

class PreferencesUpdate(BaseModel):
    language: Optional[str] = None
    voice_speed: Optional[float] = None
    alert_level: Optional[str] = None

class SavedRouteCreate(BaseModel):
    name: str
    origin: dict
    destination: dict

class IncidentReport(BaseModel):
    type: str  # jam, accident, roadwork, police
    latitude: float
    longitude: float
    description: Optional[str] = None

class RouteRequest(BaseModel):
    origin: str
    destination: str


# ----------------------------
# In-memory incidents (24h)
# ----------------------------
active_incidents = []


# ----------------------------
# Root
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Kuyenda Backend Running"}


# ----------------------------
# WebSocket Voice (CLEANED: single implementation)
# ----------------------------
@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket, token: Optional[str] = Query(None)):
    print(f"DEBUG: WebSocket connection attempt token: {token[:10] if token else 'None'}...")
    try:
        user_id = get_ws_user(ws, token=token)
        print(f"DEBUG: WebSocket Auth OK user: {user_id}")
    except HTTPException as e:
        print(f"DEBUG: WebSocket Auth failed: {e.detail}")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    print("DEBUG: WebSocket accepted")

    try:
        # ✅ Lazy import: server boots even if deps change
        from voice_agent import process_thinking_voice

        while True:
            audio_bytes = await ws.receive_bytes()
            audio_response, transcription, agent_text = await process_thinking_voice(
                audio_bytes,
                DEEPGRAM_API_KEY,
            )

            if audio_response:
                await ws.send_bytes(audio_response)
                log_voice_command(user_id, transcription, agent_text, "thinking_agent")
            else:
                await ws.send_text("EMPTY")
                log_voice_command(user_id, transcription, "IGNORE_ME", "intent_filter")

    except ModuleNotFoundError as e:
        print(f"Voice dependency missing: {e}")
        await ws.send_text("VOICE_SERVICE_UNAVAILABLE")
        await ws.close()
    except WebSocketDisconnect:
        print("Voice WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await ws.close()


# ----------------------------
# Auth Endpoints (Supabase auth, your JWT)
# ----------------------------
@app.post("/auth/signup")
async def signup(request: Request):
    try:
        body = await request.json()
        user = UserSignup(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    from supabase_client import get_supabase
    supabase = get_supabase()

    try:
        auth_response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password,
            "options": {
                "email_redirect_to": "kuyenda://auth",
                "data": {"full_name": user.full_name, "user_type": user.user_type},
            },
        })

        if not auth_response.user:
            raise Exception("Signup failed: No user returned from Supabase Auth")

        user_id = auth_response.user.id

        # Ensure profile exists (non-fatal if fails)
        try:
            profile_check = supabase.table("profiles").select("id").eq("id", user_id).execute()
            if not profile_check.data:
                supabase.table("profiles").insert({
                    "id": user_id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "user_type": user.user_type,
                }).execute()
        except Exception as profile_error:
            print(f"⚠️ Profile creation warning: {profile_error}")

        token = create_access_token({"sub": user_id})
        return {"status": "success", "token": token, "user_id": user_id}

    except Exception as e:
        detail = str(e)
        if "already" in detail.lower():
            detail = "This email address is already registered. Please login."
        raise HTTPException(status_code=400, detail=detail)


@app.post("/auth/login")
async def login(request: Request):
    try:
        body = await request.json()
        user = UserLogin(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    from supabase_client import get_supabase
    supabase = get_supabase()

    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password,
        })
        if not auth_response.user:
            raise Exception("Invalid credentials")

        user_id = auth_response.user.id
        token = create_access_token({"sub": user_id})
        return {"status": "success", "token": token, "user_id": user_id, "message": "Login successful"}

    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@app.post("/auth/verify")
def verify_otp(phone: str, otp: str):
    if otp == "1234":
        token = create_access_token({"sub": "mock-user-id"})
        return {"status": "success", "token": token}
    raise HTTPException(status_code=400, detail="Invalid OTP")


# ----------------------------
# Profiles / Preferences
# ----------------------------
@app.get("/profiles/me")
async def get_profile_me(user_id: str = Depends(get_current_user)):
    try:
        resp = get_profile(user_id)
        return resp.data
    except Exception as e:
        print(f"Get Profile Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/profiles/preferences")
def update_user_preferences(prefs: PreferencesUpdate, user_id: str = Depends(get_current_user)):
    try:
        resp = update_preferences(
            user_id=user_id,
            language=prefs.language,
            voice_speed=prefs.voice_speed,
            alert_level=prefs.alert_level,
        )
        return {"status": "success", "data": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/update")
async def update_user_profile(request: Request, user_id: str = Depends(get_current_user)):
    try:
        data = await request.json()
        profile_data = data.get("profile", data)

        from database import supabase
        update_data = {}
        if "full_name" in profile_data:
            update_data["full_name"] = profile_data["full_name"]
        if "avatar_url" in profile_data:
            update_data["avatar_url"] = profile_data["avatar_url"]

        if update_data:
            supabase.table("profiles").update(update_data).eq("id", user_id).execute()

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/avatar")
async def upload_avatar(
    request: Request,
    file: UploadFile = File(None),
    user_id: str = Depends(get_current_user),
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        ext = file.filename.split(".")[-1] if file.filename else "jpg"
        filename = f"{user_id}_{uuid.uuid4().hex[:8]}.{ext}"

        os.makedirs("static/avatars", exist_ok=True)
        file_path = f"static/avatars/{filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        base_url = str(request.base_url).rstrip("/")
        full_url = f"{base_url}/static/avatars/{filename}"
        return {"url": full_url}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ----------------------------
# Trips / Payments / Reviews / Analytics
# ----------------------------
@app.get("/trips/history")
def get_trips(user_id: str = Depends(get_current_user)):
    try:
        history = get_trip_history(user_id)
        return {"trips": history.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trips")
def start_trip(trip: TripCreate, user_id: str = Depends(get_current_user)):
    try:
        new_trip = create_trip(
            rider_id=user_id,
            origin=trip.origin,
            dest=trip.destination,
            estimated_fare=trip.estimated_fare,
        )
        return new_trip.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payments")
def create_payment(payment: PaymentCreate, user_id: str = Depends(get_current_user)):
    try:
        resp = record_payment(
            trip_id=payment.trip_id,
            amount=payment.amount,
            method=payment.method,
            metadata=payment.metadata,
        )
        return resp.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reviews")
def add_review(review: ReviewCreate, user_id: str = Depends(get_current_user)):
    try:
        resp = submit_review(
            trip_id=review.trip_id,
            reviewer_id=user_id,
            reviewee_id=review.reviewee_id,
            rating=review.rating,
            comment=review.comment,
        )
        return resp.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/log")
def log_event(event: AnalyticsEvent, user_id: str = Depends(get_current_user)):
    try:
        log_analytics_event(user_id, event.event_type, event.event_data)
        return {"status": "logged"}
    except Exception as e:
        print(f"Analytics Error: {e}")
        return {"status": "error"}


# ----------------------------
# Traffic: incidents
# ----------------------------
# ----------------------------
# Traffic: Clustering & Notifications
# ----------------------------
import math

class IncidentCluster(BaseModel):
    id: str
    type: str
    latitude: float
    longitude: float
    description: Optional[str] = None
    status: str = "PENDING"  # PENDING, ACTIVE, CONFIRMED
    reporters: List[str] = []
    created_at: str
    updated_at: str
    expires_at: str

# In-memory stores
active_clusters: Dict[str, dict] = {}  # cluster_id -> cluster_data
user_interactions: Dict[str, dict] = {} # user_id -> { 'seen': [ids], 'dismissed': [ids] }

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.post("/traffic/report")
async def report_traffic_endpoint(report: IncidentReport, user_id: str = Depends(get_current_user)):
    try:
        now = datetime.now()
        
        # 1. Clustering: fast search for nearby existing cluster (within 25m)
        match_id = None
        for cid, cluster in active_clusters.items():
            if cluster["type"] == report.type:
                dist = haversine_distance(report.latitude, report.longitude, cluster["latitude"], cluster["longitude"])
                if dist <= 25:
                    match_id = cid
                    break
        
        if match_id:
            # 2. Update existing cluster
            cluster = active_clusters[match_id]
            if user_id not in cluster["reporters"]:
                cluster["reporters"].append(user_id)
                cluster["updated_at"] = now.isoformat()
                cluster["expires_at"] = (now + timedelta(minutes=30)).isoformat() # Refresh TTL
                
                # Status Logic
                count = len(cluster["reporters"])
                if count >= 3:
                    cluster["status"] = "CONFIRMED"
                elif count >= 2:
                    cluster["status"] = "ACTIVE"
                    
            return {"status": "success", "action": "merged", "cluster": cluster}
            
        else:
            # 3. Create new cluster
            new_id = f"clus_{uuid.uuid4().hex[:8]}"
            cluster = {
                "id": new_id,
                "type": report.type,
                "latitude": report.latitude,
                "longitude": report.longitude,
                "description": report.description,
                "status": "PENDING",
                "reporters": [user_id],
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "expires_at": (now + timedelta(minutes=30)).isoformat()
            }
            active_clusters[new_id] = cluster
            
            # Async persist to DB (fire & forget)
            try:
                report_incident(user_id, report.type, report.latitude, report.longitude, report.description)
            except:
                pass
                
            return {"status": "success", "action": "created", "cluster": cluster}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notifications/user")
async def get_user_notifications(
    lat: float, 
    lng: float, 
    radius: int = 2000, 
    user_id: str = Depends(get_current_user)
):
    now = datetime.now()
    notifications = []
    
    # Init user state if needed
    if user_id not in user_interactions:
        user_interactions[user_id] = {"seen": set(), "dismissed": set()}
    
    user_state = user_interactions[user_id]
    
    # Cleanup expired clusters
    expired_ids = [cid for cid, c in active_clusters.items() if now > datetime.fromisoformat(c["expires_at"])]
    for cid in expired_ids:
        del active_clusters[cid]

    # Filter loop
    for cid, cluster in active_clusters.items():
        # Exclude own reports
        if user_id in cluster["reporters"]:
            continue
            
        # Exclude dismissed
        if cid in user_state["dismissed"]:
            continue
            
        # Check Distance (Geospatial)
        dist = haversine_distance(lat, lng, cluster["latitude"], cluster["longitude"])
        if dist <= radius:
            # Check Status (Only notify for meaningful incidents)
            if cluster["status"] in ["ACTIVE", "CONFIRMED", "PENDING"]: # Allow PENDING for demo speed
                notifications.append({
                    **cluster,
                    "distance_meters": round(dist),
                    "is_new": cid not in user_state["seen"]
                })
                
    return {"notifications": notifications, "count": len(notifications)}

@app.post("/notifications/mark-seen")
async def mark_seen(payload: dict = Body(...), user_id: str = Depends(get_current_user)):
    cluster_id = payload.get("id")
    if user_id not in user_interactions:
        user_interactions[user_id] = {"seen": set(), "dismissed": set()}
    
    if cluster_id:
        user_interactions[user_id]["seen"].add(cluster_id)
    return {"status": "marked"}

@app.delete("/notifications/{cluster_id}/dismiss")
async def dismiss_notification(cluster_id: str, user_id: str = Depends(get_current_user)):
    if user_id not in user_interactions:
        user_interactions[user_id] = {"seen": set(), "dismissed": set()}
        
    user_interactions[user_id]["dismissed"].add(cluster_id)
    return {"status": "dismissed"}

@app.get("/traffic/incidents")
async def get_incidents():
    # Return simple list for map pins (all active)
    now = datetime.now()
    return {"incidents": [
        c for c in active_clusters.values() 
        if now < datetime.fromisoformat(c["expires_at"])
    ]}

@app.get("/traffic/stats")
def get_traffic_metrics():
    try:
        stats = get_aggregated_traffic()
        return {"stats": stats.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Mapbox Tiles Proxy (optional)
# ----------------------------
@app.get("/tiles/{size}/{z}/{x}/{y}")
@limiter.limit("20/second")
async def get_tiles(request: Request, size: int, z: int, x: int, y: int, token: str = Query(...)):
    try:
        get_current_user(request, token=token)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid Token")

    mapbox_url = (
        f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{size}/{z}/{x}/{y}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(mapbox_url)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Mapbox Error")

        return Response(content=resp.content, media_type="image/png")

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Mapbox tile request timed out")
    except Exception as e:
        print(f"Tile proxy error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tile")


# ----------------------------
# Route (Mapbox Directions)
# ----------------------------
@app.get("/route")
@limiter.limit("5/second")
async def get_route(
    request: Request,
    start: str,
    end: str,
    optimize: str = Query("duration", enum=["duration", "distance"]),
    user_id: str = Depends(get_current_user),
):
    """
    start/end from frontend are 'lng,lat'
    optimize: duration (fastest) or distance (shortest)
    """
    lng1, lat1 = parse_lng_lat(start)
    lng2, lat2 = parse_lng_lat(end)

    if not validate_coords(lat1, lng1) or not validate_coords(lat2, lng2):
        raise HTTPException(status_code=400, detail="Coordinates out of bounds")

    coords = f"{lng1},{lat1};{lng2},{lat2}"

    mapbox_url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/{coords}"
        f"?geometries=geojson&steps=true&alternatives=true&overview=full"
        f"&annotations=congestion&access_token={MAPBOX_TOKEN}"
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(mapbox_url)

    if resp.status_code != 200:
        print(f"Mapbox Route Error: {resp.text[:300]}")
        raise HTTPException(status_code=resp.status_code, detail="Routing failed")

    data = resp.json()

    # Sort routes by optimize
    if "routes" in data and len(data["routes"]) > 1:
        if optimize == "distance":
            data["routes"].sort(key=lambda x: x.get("distance", float("inf")))
        else:
            data["routes"].sort(key=lambda x: x.get("duration", float("inf")))

    return data


# ----------------------------
# ML Traffic Prediction Endpoints
# ----------------------------
@app.post("/predict-traffic")
async def predict_traffic_endpoint(payload: dict = Body(...)):
    """
    Expected payload (minimum):
    {
      "route_coordinates": [[lng,lat], [lng,lat], ...],
      "route_segments": [{"duration_s": 12.3, "distance_m": 55, "congestion": "moderate"}, ...] (optional),
      "route_id": "route_0" (optional),
      "origin": "28.2,-15.4" (optional),
      "destination": "28.3,-15.3" (optional),
      "current_time": 8 (optional),
      "day_of_week": 1 (optional)
    }
    """
    coords = payload.get("route_coordinates") or []
    segments = payload.get("route_segments")
    route_id = payload.get("route_id", "route_0")
    origin = payload.get("origin", "")
    destination = payload.get("destination", "")
    hour = payload.get("current_time", None)
    dow = payload.get("day_of_week", None)

    result = traffic_predictor.predict_route(
        coords,
        route_segments=segments,
        route_id=route_id,
        origin=origin,
        destination=destination,
        hour=hour,
        day_of_week=dow,
    )

    return {
        "status": "success",
        "using_ml": result.using_ml,
        "segments": result.segments,
        "breakdown": result.breakdown,
        "segments_count": len(result.segments),
    }

@app.post("/traffic-status")
async def traffic_status_endpoint(payload: dict = Body(...)):
    # Same as predict for now (you can add caching later)
    return await predict_traffic_endpoint(payload)

@app.get("/test-ml-model")
async def test_ml_model():
    try:
        traffic_predictor.load()

        import pandas as pd
        sample = pd.DataFrame([{
            "current_travel_min": 5.0,
            "normal_travel_min": 3.0,
            "reports_count": 0,
            "route_id": "r1",
            "origin": "28.2833,-15.4167",
            "destination": "28.3228,-15.3875",
            "hotspot": "no",
            "day_of_week": "1",
            "congestion_level": "moderate",
        }])

        pred = traffic_predictor.model.predict(sample)  # type: ignore
        proba = None
        if hasattr(traffic_predictor.model, "predict_proba"):  # type: ignore
            proba = traffic_predictor.model.predict_proba(sample)  # type: ignore

        return {
            "status": "success",
            "model_type": str(type(traffic_predictor.model)),
            "prediction": int(pred[0]),
            "proba_max": float(max(proba[0])) if proba is not None else None,
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


# ----------------------------
# Geocode Search (Lusaka-only)
# ----------------------------
KNOWN_STOPS = [
    {"name": "Eden University", "address": "Barlstone Park, Lusaka", "latitude": -15.3636, "longitude": 28.2334, "category": "Education"},
    {"name": "University of Zambia", "address": "Great East Road, Lusaka", "latitude": -15.3875, "longitude": 28.3228, "category": "Education"},
    {"name": "Town Center", "address": "Cairo Road, Lusaka", "latitude": -15.4180, "longitude": 28.2820, "category": "City Center"},
    {"name": "Manda Hill Shopping Mall", "address": "Great East Road, Lusaka", "latitude": -15.3975, "longitude": 28.3070, "category": "Shopping"},
    {"name": "Arcades Shopping Mall", "address": "Great East Road, Lusaka", "latitude": -15.3956, "longitude": 28.3388, "category": "Shopping"},
]

LUSAKA_BBOX = [28.10, -15.60, 29.10, -15.20]  # [west, south, east, north]

def in_lusaka_bbox(lng_val: float, lat_val: float) -> bool:
    w, s, e, n = LUSAKA_BBOX
    return (w <= lng_val <= e) and (s <= lat_val <= n)

def normalize_text(text: str) -> str:
    return (text or "").lower().strip()

@app.get("/geocode/search")
@limiter.limit("30/minute")
async def geocode_search(
    request: Request,
    query: str = Query(..., min_length=2),
    lat: float | None = None,
    lng: float | None = None,
):
    bbox_str = ",".join(str(x) for x in LUSAKA_BBOX)
    default_proximity = (28.3228, -15.3875)  # (lng,lat)
    proximity = (lng, lat) if (lat is not None and lng is not None) else default_proximity
    proximity_str = f"{proximity[0]},{proximity[1]}"

    q_norm = normalize_text(query)
    results = []

    # 1) local stops
    for stop in KNOWN_STOPS:
        if q_norm in normalize_text(stop["name"]) or q_norm in normalize_text(stop["address"]):
            if in_lusaka_bbox(stop["longitude"], stop["latitude"]):
                results.append({
                    "name": stop["name"],
                    "address": stop["address"],
                    "latitude": stop["latitude"],
                    "longitude": stop["longitude"],
                    "category": stop.get("category", "Unknown"),
                    "source": "kuyenda_local",
                })

    # 2) mapbox geocode (v6 preferred, v5 fallback)
    from urllib.parse import quote
    encoded_query = quote(query.strip())
    v6_url = "https://api.mapbox.com/search/geocode/v6/forward"

    params_v6 = {
        "q": query.strip(),
        "access_token": MAPBOX_TOKEN,
        "country": "ZM",
        "bbox": bbox_str,
        "proximity": proximity_str,
        "types": "poi,address,place,street,neighborhood",
        "limit": 15,
        "language": "en",
    }

    resp = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(v6_url, params=params_v6)
            if resp.status_code != 200:
                v5_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_query}.json"
                params_v5 = {
                    "access_token": MAPBOX_TOKEN,
                    "country": "ZM",
                    "bbox": bbox_str,
                    "proximity": proximity_str,
                    "types": "poi,address,place,locality,neighborhood",
                    "limit": 15,
                    "autocomplete": "true",
                    "fuzzyMatch": "true",
                    "language": "en",
                }
                resp = await client.get(v5_url, params=params_v5)
    except Exception as e:
        print(f"⚠️ Mapbox geocode error: {e}")
        resp = None

    if resp and resp.status_code == 200:
        data = resp.json()
        features = data.get("features", [])
        for feat in features:
            geometry = feat.get("geometry", {})
            coordinates = geometry.get("coordinates", []) or feat.get("center", [])
            if len(coordinates) != 2:
                continue

            feat_lng, feat_lat = float(coordinates[0]), float(coordinates[1])
            if not in_lusaka_bbox(feat_lng, feat_lat):
                continue

            name = feat.get("text") or feat.get("place_name", "Unknown")
            address = feat.get("place_name") or feat.get("properties", {}).get("full_address", name)
            results.append({
                "name": name,
                "address": address,
                "latitude": feat_lat,
                "longitude": feat_lng,
                "source": "mapbox",
            })

    # 3) dedupe
    seen = set()
    deduped = []
    for r in results:
        key = (
            normalize_text(r.get("name", "")),
            round(float(r["longitude"]), 4),
            round(float(r["latitude"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # 4) final validation + sort
    valid = [r for r in deduped if in_lusaka_bbox(r["longitude"], r["latitude"])]
    valid.sort(key=lambda x: (0 if x.get("source") == "kuyenda_local" else 1, x.get("name", "").lower()))
    return {"results": valid[:15]}

@app.get("/geocode/reverse")
@limiter.limit("30/minute")
async def reverse_geocode(request: Request, lat: float, lng: float):
    if not validate_coords(lat, lng):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    mapbox_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lng},{lat}.json"
    params = {"access_token": MAPBOX_TOKEN, "limit": 1}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(mapbox_url, params=params)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Reverse geocoding failed")

    data = resp.json()
    if data.get("features"):
        f = data["features"][0]
        return {"name": f.get("text"), "address": f.get("place_name"), "latitude": lat, "longitude": lng}

    return {"name": f"{lat}, {lng}", "address": "Unknown location", "latitude": lat, "longitude": lng}


# ----------------------------
# Legacy AI predict (kept for app stability)
# ----------------------------
@app.post("/ai/predict")
async def predict_departure(route: RouteRequest):
    # Simplified legacy fallback since get_traffic_status is removed
    # In a real scenario, this would use traffic_predictor.predict_route
    return {
        "status": "success",
        "recommended_departure_in": "Now",
        "delay_minutes": 5,
        "message": "Roads are clear in your area. Safe travels!",
    }


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
