
import os
import re
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, status, Response, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from starlette.requests import HTTPConnection
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import jwt, JWTError
from datetime import datetime, timedelta
import uvicorn
import pickle
import sys
import shutil
import uuid
import custom_transformers
from traffic import set_model, get_traffic_status
from stt import transcribe_audio
from intent import detect_intent
from tts import speak
from session import SessionManager
from config import SECRET_KEY, ALGORITHM, DEEPGRAM_API_KEY, CARTESIA_API_KEY, MAPBOX_TOKEN
from database import (
    create_profile, get_profile, create_trip, update_trip_status, 
    get_trip_history, save_route, record_payment, submit_review, log_analytics_event,
    update_preferences, add_saved_route, get_saved_routes, get_aggregated_traffic, log_voice_command,
    report_incident
)

# --- 1. Configuration & Setup ---

# Environment variables are loaded in config.py


# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Kuyenda Backend Proxy")

# Register Rate Limit Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


# CORS (Allow mobile app to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to specific domains/schemes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"DEBUG: Incoming {request.method} request to {request.url}")
    response = await call_next(request)
    print(f"DEBUG: Response status: {response.status_code}")
    return response

# Initialize Deepgram (already handled in stt.py, but keeping here if needed for direct access)
# dg = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

import joblib

# Fix for pickle/joblib loading: ensure the class is available where the model expects it
import __main__
__main__.BehavioralFeatureExtractor = custom_transformers.BehavioralFeatureExtractor

model_path = "models/traffic_model_v2.pkl"
try:
    # joblib is generally better for scikit-Learn models
    model = joblib.load(model_path)
    print("Traffic Model Loaded Successfully via Joblib")
    set_model(model)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- 2. Validation & Utilities ---

# Zambia Bounds (Approximate)
ZAMBIA_BOUNDS = {
    "min_lat": -18.5, "max_lat": -8.0,
    "min_lng": 21.5, "max_lng": 34.0
}

def validate_coords(lat: float, lng: float):
    """Ensure coordinates are valid and within Zambia."""
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        return False
    # Check bounds (optional strictness)
    if not (ZAMBIA_BOUNDS["min_lat"] <= lat <= ZAMBIA_BOUNDS["max_lat"]):
        # Log this suspicious activity?
        print(f"Suspicious: Coords out of Zambia bounds: {lat}, {lng}")
        # Allow for now, or return False to block
        return True 
    return True

def parse_lat_lng(coord_str: str):
    """Parse 'lat,lng' string."""
    try:
        parts = coord_str.split(',')
        if len(parts) != 2:
            raise ValueError
        lat, lng = float(parts[0]), float(parts[1])
        return lat, lng
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid coordinate format. Use 'lat,lng'")

# --- 3. Authentication ---

oauth2_scheme = "Bearer" # Simplification for mobile

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30) # Long expiry for mobile
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(conn: HTTPConnection, token: Optional[str] = Query(None)):
    """
    Validate JWT.
    Accepts token from Authorization Header OR 'token' query param.
    Works for both HTTP Requests and WebSockets because HTTPConnection 
    is a parent for both.
    """
    # 1. Try to get token from headers
    auth_header = conn.headers.get('Authorization')
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        
    
    # 2. Path/Query token is already handled by 'token: str = Query(...)' 
    # but we check if it was found.

    if not token:
         raise HTTPException(status_code=401, detail="Missing authentication")

    try:
        # For Supabase, we could use supabase.auth.get_user(token)
        # But for now, we'll keep the JWT decoding if the SECRET_KEY matches,
        # or transition to Supabase token verification.
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise JWTError
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- 4. Models ---

class UserLogin(BaseModel):
    email: str
    password: str

class UserSignup(BaseModel):
    email: str
    password: str
    full_name: str
    user_type: str = "rider"

class TripCreate(BaseModel):
    origin: dict # {name, lat, lng}
    destination: dict # {name, lat, lng}
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
    origin: dict # {name, lat, lng}
    destination: dict # {name, lat, lng}

class IncidentReport(BaseModel):
    type: str # jam, accident, roadwork, police
    latitude: float
    longitude: float
    description: Optional[str] = None

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None

active_incidents = []

# Authentication, Traffic, and Voice logic moved to modular files.
# Using imports now.

# --- 5. Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Kuyenda Mapbox Proxy Running"}

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket, token: Optional[str] = Query(None)):
    print(f"DEBUG: WebSocket connection attempt with token: {token[:10] if token else 'None'}...")
    try:
        user = get_current_user(conn=ws, token=token)
        print(f"DEBUG: WebSocket Auth successful for user: {user}")
    except HTTPException as e:
        print(f"DEBUG: WebSocket Auth failed: {e.detail}")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    print("DEBUG: WebSocket connection accepted")
    session = SessionManager()

    try:
        # ✅ Lazy import here so server can boot even if Deepgram SDK changes
        from voice_agent import process_thinking_voice

        while True:
            audio_bytes = await ws.receive_bytes()
            print(f"Received {len(audio_bytes)} bytes of audio")

            audio_response, transcription, agent_text = await process_thinking_voice(
                audio_bytes,
                DEEPGRAM_API_KEY
            )

            if audio_response:
                await ws.send_bytes(audio_response)
                log_voice_command(user, transcription, agent_text, "thinking_agent")
            else:
                await ws.send_text("EMPTY")
                log_voice_command(user, transcription, "IGNORE_ME", "intent_filter")

    except ModuleNotFoundError as e:
        # If deepgram agent types missing, don't crash entire service
        print(f"Voice dependency missing: {e}")
        await ws.send_text("VOICE_SERVICE_UNAVAILABLE")
        await ws.close()
    except WebSocketDisconnect:
        print("Voice WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await ws.close()

    print(f"DEBUG: WebSocket connection attempt with token: {token[:10] if token else 'None'}...")
    try:
        # get_current_user will automatically take 'ws' as its 'conn' because WebSocket inherits from HTTPConnection
        user = get_current_user(conn=ws, token=token)
        print(f"DEBUG: WebSocket Auth successful for user: {user}")
    except HTTPException as e:
        print(f"DEBUG: WebSocket Auth failed: {e.detail}")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    print("DEBUG: WebSocket connection accepted")
    session = SessionManager()

    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            print(f"Received {len(audio_bytes)} bytes of audio")

            # Use the new Thinking Agent (Deepgram Agent V1 + Gemini)
            print("Processing with Thinking Agent...")
            audio_response, transcription, agent_text = await process_thinking_voice(audio_bytes, DEEPGRAM_API_KEY)
            
            if audio_response:
                print(f"Sending audio response ({len(audio_response)} bytes)")
                await ws.send_bytes(audio_response)
                
                # Log to Supabase (Success)
                log_voice_command(user, transcription, agent_text, "thinking_agent")
            else:
                print("Thinking Agent returned no audio (Ignored or Error)")
                # Send "EMPTY" so frontend knows to just resume listening quietly
                await ws.send_text("EMPTY")
                # Log even if ignored (optional, but good for analytics)
                log_voice_command(user, transcription, "IGNORE_ME", "intent_filter")
            
    except WebSocketDisconnect:
        print("Voice WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await ws.close()

@app.post("/auth/signup")
async def signup(request: Request):
    """
    Handle user signup with Supabase Auth.
    """
    try:
        body = await request.json()
        print(f"DEBUG: Signup Payload: {body}")
        user = UserSignup(**body)
    except Exception as e:
        print(f"DEBUG: Signup Payload Validation Error: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    from supabase_client import get_supabase
    import asyncio
    supabase = get_supabase()
    
    try:
        # 1. Sign up the user in Supabase Auth
        # Disable email confirmation for faster signup
        auth_response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password,
            "options": {
                "email_redirect_to": "kuyenda://auth",
                "data": {
                    "full_name": user.full_name,
                    "user_type": user.user_type
                }
            }
        })
        
        if not auth_response.user:
             raise Exception("Signup failed: No user returned from Supabase Auth")
             
        user_id = auth_response.user.id
        
        # 2. Create profile manually if trigger doesn't exist or fails
        try:
            # Check if profile exists
            profile_check = supabase.table("profiles").select("id").eq("id", user_id).execute()
            
            if not profile_check.data:
                # Create profile manually
                profile_data = {
                    "id": user_id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "user_type": user.user_type
                }
                supabase.table("profiles").insert(profile_data).execute()
                print(f"Profile created manually for user {user_id}")
        except Exception as profile_error:
            print(f"Profile creation warning: {profile_error}")
            # Don't fail signup if profile creation fails - it might be handled by trigger
        
        access_token = create_access_token(data={"sub": user_id})
        return {"status": "success", "token": access_token, "user_id": user_id}
        
    except Exception as e:
        print(f"SIGNUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Return more descriptive error to frontend
        detail = str(e)
        if "timed out" in detail.lower():
            detail = "Signup is taking longer than expected. Please try again or check your internet connection."
        elif "profiles_id_fkey" in detail:
            detail = "Profile creation failed: User ID not found. Ensure Supabase Auth is correctly configured."
        elif "already registered" in detail or "already exists" in detail or "already been registered" in detail:
            detail = "This email address is already registered. Please try logging in instead."
            
        raise HTTPException(status_code=400, detail=detail)

@app.post("/auth/login")
async def login(request: Request):
    """
    Login with Supabase Auth using phone-email and password.
    """
    try:
        body = await request.json()
        print(f"DEBUG: Login Payload: {body}")
        user = UserLogin(**body)
    except Exception as e:
        print(f"DEBUG: Login Validation Error: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    from supabase_client import get_supabase
    supabase = get_supabase()
    
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        
        if not auth_response.user:
            raise Exception("Login failed: Invalid credentials.")
            
        user_id = auth_response.user.id
        access_token = create_access_token(data={"sub": user_id})
        
        return {
            "status": "success", 
            "token": access_token,
            "user_id": user_id,
            "message": "Login successful"
        }
    except Exception as e:
        print(f"LOGIN ERROR: {e}")
        raise HTTPException(status_code=401, detail="Invalid email or password.")

@app.post("/auth/verify")
def verify_otp(phone: str, otp: str):
    """Verify OTP endpoint."""
    if otp == "1234":
         access_token = create_access_token(data={"sub": "mock-user-id"})
         return {"status": "success", "token": access_token}
    raise HTTPException(status_code=400, detail="Invalid OTP")

    try:
        profile = get_profile(user_id)
        return profile.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Profile not found")

@app.patch("/profiles/preferences")
def update_user_preferences(prefs: PreferencesUpdate, user_id: str = Depends(get_current_user)):
    try:
        resp = update_preferences(
            user_id=user_id,
            language=prefs.language,
            voice_speed=prefs.voice_speed,
            alert_level=prefs.alert_level
        )
        return {"status": "success", "data": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            estimated_fare=trip.estimated_fare
        )
        return new_trip.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/saved-routes")
def get_user_routes(user_id: str = Depends(get_current_user)):
    try:
        routes = get_saved_routes(user_id)
        return {"routes": routes.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/saved-routes")
def save_user_route(route: SavedRouteCreate, user_id: str = Depends(get_current_user)):
    try:
        resp = add_saved_route(
            user_id=user_id,
            name=route.name,
            origin=route.origin,
            dest=route.destination
        )
        return resp.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/traffic/stats")
def get_traffic_metrics():
    try:
        stats = get_aggregated_traffic()
        return {"stats": stats.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payments")
def create_payment(payment: PaymentCreate, user_id: str = Depends(get_current_user)):
    try:
        resp = record_payment(
            trip_id=payment.trip_id,
            amount=payment.amount,
            method=payment.method,
            metadata=payment.metadata
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
            comment=review.comment
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

@app.post("/traffic/report")
async def report_traffic_endpoint(report: IncidentReport, user_id: str = Depends(get_current_user)):
    try:
        # 1. Log to In-Memory for instant delivery to other users
        incident = {
            "id": f"incident_{len(active_incidents) + 1}",
            "user_id": user_id,
            "type": report.type,
            "latitude": report.latitude,
            "longitude": report.longitude,
            "description": report.description,
            "created_at": datetime.now().isoformat()
        }
        active_incidents.append(incident)

        # 2. Persist to Supabase
        try:
            report_incident(
                user_id=user_id,
                incident_type=report.type,
                lat=report.latitude,
                lng=report.longitude,
                description=report.description
            )
        except Exception as db_err:
            print(f"Database insertion failed for incident: {db_err}")
            # We don't raise here so the user still sees "success" from the in-memory log

        return {"status": "success", "incident": incident}
    except Exception as e:
        print(f"Traffic Report Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/traffic/incidents")
async def get_incidents():
    # Return incidents from the last 24 hours
    now = datetime.now()
    filtered = [i for i in active_incidents if (now - datetime.fromisoformat(i["created_at"])).total_seconds() < 86400]
    return {"incidents": filtered}

@app.post("/user/avatar")
async def upload_avatar(request: Request, file: UploadFile = File(None), user_id: str = Depends(get_current_user)):
    try:
        # Check if file was provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # 1. Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # 2. Create unique filename
        ext = file.filename.split(".")[-1] if file.filename else "jpg"
        filename = f"{user_id}_{uuid.uuid4().hex[:8]}.{ext}"
        
        # Ensure avatars directory exists
        import os
        os.makedirs("static/avatars", exist_ok=True)
        
        file_path = f"static/avatars/{filename}"

        # 3. Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 4. Return URL
        base_url = str(request.base_url).rstrip("/")
        full_url = f"{base_url}/static/avatars/{filename}"
        
        print(f"Avatar uploaded successfully: {full_url}")
        return {"url": full_url}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Avatar Upload Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/profiles/me")
async def get_profile_me(user_id: str = Depends(get_current_user)):
    try:
        from database import get_profile
        resp = get_profile(user_id)
        return resp.data
    except Exception as e:
        print(f"Get Profile Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/update")
async def update_user_profile(request: Request, user_id: str = Depends(get_current_user)):
    try:
        data = await request.json()
        print(f"Update Profile Request Body: {data}")
        
        # Handle both flat and embedded 'profile' styles for robustness
        profile_data = data.get("profile", data)
        
        from database import supabase
        update_data = {}
        if "full_name" in profile_data: update_data["full_name"] = profile_data["full_name"]
        if "avatar_url" in profile_data: update_data["avatar_url"] = profile_data["avatar_url"]
        
        if update_data:
            supabase.table("profiles").update(update_data).eq("id", user_id).execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Mapbox Proxy Endpoints ---

@app.get("/tiles/{size}/{z}/{x}/{y}")
@limiter.limit("20/second")
async def get_tiles(request: Request, size: int, z: int, x: int, y: int, token: str = Query(...)):
    """
    Proxy Mapbox Raster Tiles.
    Requires 'token' query param for Auth.
    """
    # Manual auth check to ensure token is valid
    try:
        get_current_user(conn=request, token=token)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid Token")
        
    mapbox_url = f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{size}/{z}/{x}/{y}?access_token={MAPBOX_TOKEN}"
    
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

@app.get("/route")
@limiter.limit("5/second")
async def get_route(
    request: Request, 
    start: str, 
    end: str, 
    optimize: str = Query("duration", enum=["duration", "distance"]),
    user: str = Depends(get_current_user)
):
    """
    Proxy Mapbox Driving Directions.
    Format from Frontend: start=lng,lat & end=lng,lat
    'optimize' can be 'duration' (fastest) or 'distance' (shortest).
    """
    # 1. Parse and Validate
    # Frontend sends lng,lat (Standard GeoJSON/Mapbox format)
    lng1, lat1 = parse_lat_lng(start)
    lng2, lat2 = parse_lat_lng(end)
    
    if not validate_coords(lat1, lng1) or not validate_coords(lat2, lng2):
        raise HTTPException(status_code=400, detail="Coordinates out of bounds")

    # 2. Construct Mapbox URL
    # Mapbox expects: longitude,latitude
    coords = f"{lng1},{lat1};{lng2},{lat2}"
    
    # Use mapbox/driving-traffic for real-time congestion data
    mapbox_url = f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/{coords}?geometries=geojson&steps=true&alternatives=true&overview=full&annotations=congestion&access_token={MAPBOX_TOKEN}"
    
    # 3. Request
    async with httpx.AsyncClient() as client:
        resp = await client.get(mapbox_url)
        
    if resp.status_code != 200:
        # Log error
        print(f"Mapbox Route Error: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail="Routing failed")
        
    data = resp.json()
    
    # 4. Sort routes based on optimization preference
    if "routes" in data and len(data["routes"]) > 1:
        if optimize == "distance":
            # Shortest distance first
            data["routes"].sort(key=lambda x: x.get("distance", float('inf')))
        else:
            # Fastest duration first (default Mapbox behavior but let's be explicit)
            data["routes"].sort(key=lambda x: x.get("duration", float('inf')))
            
    return data

    
# --- Search Configuration ---

# Move known_stops to module level (outside function)
KNOWN_STOPS = [
    # Educational Institutions
    {"name": "Eden University", "address": "Barlstone Park, Lusaka", "latitude": -15.3636, "longitude": 28.2334, "category": "Education"},
    {"name": "University of Zambia", "address": "Great East Road, Lusaka", "latitude": -15.3875, "longitude": 28.3228, "category": "Education"},
    
    # Parks & Recreation
    {"name": "Barlstone Park", "address": "Lusaka West", "latitude": -15.3680, "longitude": 28.2300, "category": "Park"},
    
    # Shopping Malls
    {"name": "Manda Hill Shopping Mall", "address": "Great East Road, Lusaka", "latitude": -15.3975, "longitude": 28.3070, "category": "Shopping"},
    {"name": "Arcades Shopping Mall", "address": "Great East Road, Lusaka", "latitude": -15.3956, "longitude": 28.3388, "category": "Shopping"},
    {"name": "East Park Mall", "address": "Great East Road, Lusaka", "latitude": -15.3950, "longitude": 28.3400, "category": "Shopping"},
    
    # Markets
    {"name": "Kamwala Market", "address": "Kamwala, Lusaka", "latitude": -15.4300, "longitude": 28.2930, "category": "Market"},
    
    # Hospitals & Healthcare
    {"name": "Levy Mwanawasa University Teaching Hospital", "address": "Great East Road, Lusaka", "latitude": -15.3965, "longitude": 28.3490, "category": "Hospital"},
    
    # City Center
    {"name": "Town Center", "address": "Cairo Road, Lusaka", "latitude": -15.4180, "longitude": 28.2820, "category": "City Center"},
    
    # Airport
    {"name": "Kenneth Kaunda International Airport", "address": "Great East Road, Chongwe District", "latitude": -15.3308, "longitude": 28.4535, "category": "Airport"},
    
    # Museums
    {"name": "Zambia National Museum", "address": "Independence Avenue, Lusaka", "latitude": -15.4200, "longitude": 28.2850, "category": "Museum"},
    
    # RESTAURANTS - INDIAN CUISINE
    {"name": "Dil Restaurant", "address": "Ibex Hill, Lusaka", "latitude": -15.3800, "longitude": 28.3100, "category": "Restaurant - Indian"},
    {"name": "Royal Dil", "address": "Acacia Office Park, Lusaka", "latitude": -15.3960, "longitude": 28.3380, "category": "Restaurant - Indian"},
    
    # RESTAURANTS - CHINESE CUISINE
    {"name": "Sichuan Restaurant", "address": "Show Grounds, Lusaka", "latitude": -15.4100, "longitude": 28.2900, "category": "Restaurant - Chinese"},
    
    # RESTAURANTS - ITALIAN CUISINE
    {"name": "Casa Portico", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3000, "category": "Restaurant - Italian"},
    {"name": "Eataly Restaurant & Pizzeria", "address": "Lusaka", "latitude": -15.4050, "longitude": 28.2950, "category": "Restaurant - Italian"},
    
    # RESTAURANTS - STEAKHOUSES
    {"name": "Marlin Restaurant", "address": "Longacres, Lusaka", "latitude": -15.4150, "longitude": 28.2900, "category": "Restaurant - Steakhouse"},
    {"name": "Marlin EastPark Mall", "address": "EastPark Mall, Lusaka", "latitude": -15.3950, "longitude": 28.3400, "category": "Restaurant - Steakhouse"},
    {"name": "Flame Restaurant", "address": "Lusaka", "latitude": -15.3900, "longitude": 28.3200, "category": "Restaurant - Steakhouse"},
    {"name": "Copper Creek Spur", "address": "Manda Hill Shopping Centre, Lusaka", "latitude": -15.3975, "longitude": 28.3070, "category": "Restaurant - Steakhouse"},
    
    # RESTAURANTS - CAFES & CASUAL DINING
    {"name": "Mint Lounge", "address": "Acacia Office Park, near Arcades, Lusaka", "latitude": -15.3960, "longitude": 28.3380, "category": "Restaurant - Cafe"},
    {"name": "The Deli", "address": "Rhodes Park, Lusaka", "latitude": -15.4000, "longitude": 28.3050, "category": "Restaurant - Cafe"},
    {"name": "Sugarbush Cafe", "address": "Sugarbush Farm, outside Lusaka", "latitude": -15.3500, "longitude": 28.2500, "category": "Restaurant - Cafe"},
    {"name": "The Corner Cafe", "address": "Lusaka", "latitude": -15.4100, "longitude": 28.2900, "category": "Restaurant - Cafe"},
    {"name": "Zambean Coffee Co", "address": "Leopards Hill Business Park, Lusaka", "latitude": -15.3700, "longitude": 28.3700, "category": "Restaurant - Cafe"},
    
    # RESTAURANTS - FINE DINING
    {"name": "Botanica", "address": "Lusaka", "latitude": -15.3850, "longitude": 28.3150, "category": "Restaurant - Fine Dining"},
    {"name": "Jessyz Fine Dining", "address": "KK Mall, Alick Nkhata Road, Longacres", "latitude": -15.4150, "longitude": 28.2900, "category": "Restaurant - Fine Dining"},
    {"name": "Latitude 15 Degrees", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3100, "category": "Restaurant - Hotel"},
    
    # RESTAURANTS - ETHIOPIAN CUISINE
    {"name": "Green Ethiopian Restaurant", "address": "Lusaka", "latitude": -15.4050, "longitude": 28.2950, "category": "Restaurant - Ethiopian"},
    
    # RESTAURANTS - GREEK CUISINE
    {"name": "Eviva Greek Restaurant", "address": "Rhodes Park, Lusaka", "latitude": -15.4000, "longitude": 28.3050, "category": "Restaurant - Greek"},
    
    # RESTAURANTS - PORTUGUESE CUISINE
    {"name": "Adega Restaurant", "address": "Levy Junction, Lusaka", "latitude": -15.3965, "longitude": 28.3490, "category": "Restaurant - Portuguese"},
    {"name": "Mozambik Lusaka", "address": "Pinnacle Mall, Lusaka", "latitude": -15.4000, "longitude": 28.3000, "category": "Restaurant - Mozambican"},
    
    # RESTAURANTS - THAI CUISINE
    {"name": "Thai Restaurant", "address": "Lusaka", "latitude": -15.3950, "longitude": 28.3200, "category": "Restaurant - Thai"},
    
    # RESTAURANTS - KOREAN CUISINE
    {"name": "Arirang Restaurant", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3100, "category": "Restaurant - Korean"},
    
    # RESTAURANTS - MOROCCAN CUISINE
    {"name": "Dar Lalla", "address": "Lusaka", "latitude": -15.3900, "longitude": 28.3000, "category": "Restaurant - Moroccan"},
    
    # RESTAURANTS - TURKISH CUISINE
    {"name": "Istanbul Cafe & Restaurant", "address": "Lusaka", "latitude": -15.4050, "longitude": 28.2950, "category": "Restaurant - Turkish"},
    
    # RESTAURANTS - INTERNATIONAL/FUSION
    {"name": "The Fat Chef", "address": "East Park Mall, Lusaka", "latitude": -15.3950, "longitude": 28.3400, "category": "Restaurant - Fusion"},
    {"name": "Four Seasons Bistro", "address": "Suez Road, Ridgeway", "latitude": -15.4200, "longitude": 28.3100, "category": "Restaurant - International"},
    {"name": "The Retreat", "address": "Outskirts of Lusaka", "latitude": -15.3600, "longitude": 28.2800, "category": "Restaurant - International"},
    {"name": "Misty Restaurant and Jazz Cafe", "address": "Lusaka", "latitude": -15.4100, "longitude": 28.2900, "category": "Restaurant - Continental"},
    {"name": "Bojangles Lusaka", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3000, "category": "Restaurant - International"},
    {"name": "Prime Joint", "address": "EastPark Mall, Lusaka", "latitude": -15.3950, "longitude": 28.3400, "category": "Restaurant - International"},
    
    # RESTAURANTS - PIZZA & CASUAL
    {"name": "Scallywags Cafe and Restaurant", "address": "Lusaka", "latitude": -15.3950, "longitude": 28.3100, "category": "Restaurant - Pizza"},
    {"name": "Debonairs Pizza", "address": "Multiple locations, Lusaka", "latitude": -15.4000, "longitude": 28.3000, "category": "Restaurant - Pizza"},
    
    # RESTAURANTS - TRADITIONAL ZAMBIAN
    {"name": "Twapandula", "address": "Lusaka", "latitude": -15.4050, "longitude": 28.2950, "category": "Restaurant - Zambian"},
    {"name": "Down To Earth Restaurant & Bar", "address": "Lusaka", "latitude": -15.3950, "longitude": 28.3050, "category": "Restaurant - Zambian"},
    {"name": "Taste Restaurant", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3000, "category": "Restaurant - Zambian"},
    
    # RESTAURANTS - SEAFOOD
    {"name": "John Dory's Lusaka", "address": "Lusaka", "latitude": -15.3950, "longitude": 28.3100, "category": "Restaurant - Seafood"},
    
    # BARS & LOUNGES
    {"name": "Rhapsody's Cafe & Wine Bar", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3050, "category": "Bar & Lounge"},
    {"name": "Fox & Hound", "address": "Lusaka", "latitude": -15.4050, "longitude": 28.2950, "category": "Bar & Restaurant"},
    {"name": "Fresco Bar & Cafe", "address": "Lusaka", "latitude": -15.3950, "longitude": 28.3000, "category": "Bar & Cafe"},
    {"name": "The Orange Tree Public House", "address": "Lusaka", "latitude": -15.4000, "longitude": 28.3100, "category": "Pub"},
    
    # HOTELS WITH RESTAURANTS
    {"name": "Intercontinental Hotel Lusaka", "address": "Lusaka", "latitude": -15.4150, "longitude": 28.2850, "category": "Hotel"},
    {"name": "Radisson Blu Hotel", "address": "Lusaka", "latitude": -15.3900, "longitude": 28.3100, "category": "Hotel"},
    {"name": "Taj Pamodzi Hotel", "address": "Lusaka", "latitude": -15.4100, "longitude": 28.2900, "category": "Hotel"},
    {"name": "Southern Sun Ridgeway", "address": "Ridgeway, Lusaka", "latitude": -15.4200, "longitude": 28.3100, "category": "Hotel"},
    
    # ADDITIONAL ATTRACTIONS
    {"name": "Chaminuka Nature Reserve", "address": "40 minutes from Lusaka", "latitude": -15.2500, "longitude": 28.4000, "category": "Nature Reserve"},
    {"name": "Kalimba Reptile Park", "address": "Lusaka", "latitude": -15.3800, "longitude": 28.3200, "category": "Attraction"},
]

# Lusaka bounding box - consider expanding slightly if too restrictive
LUSAKA_BBOX = [28.10, -15.60, 29.10, -15.20]  # [west, south, east, north]


def in_lusaka_bbox(lng_val: float, lat_val: float) -> bool:
    """Check if coordinates are within Lusaka bounding box"""
    w, s, e, n = LUSAKA_BBOX
    return (w <= lng_val <= e) and (s <= lat_val <= n)


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    return text.lower().strip()


@app.get("/geocode/search")
@limiter.limit("30/minute")
async def geocode_search(
    request: Request,
    query: str = Query(..., min_length=2),
    lat: float | None = None,
    lng: float | None = None
):
    """
    Lusaka-only search (Local + Mapbox).
    - Uses bbox to restrict results to Lusaka
    - Uses proximity (lat/lng) to bias results near the user
    - Returns only results with valid coordinates
    """
    
    bbox_str = ",".join(str(x) for x in LUSAKA_BBOX)
    
    # Lusaka fallback center (UNZA coordinates)
    default_proximity = (28.3228, -15.3875)  # (lng, lat)
    
    # If frontend provides location, use it (better ranking)
    if lat is not None and lng is not None:
        proximity = (lng, lat)
    else:
        proximity = default_proximity
    
    proximity_str = f"{proximity[0]},{proximity[1]}"
    
    # Normalize query for matching
    q_normalized = normalize_text(query)
    results = []
    
    # 1) Search local known stops (fast + guaranteed Lusaka)
    for stop in KNOWN_STOPS:
        name_normalized = normalize_text(stop["name"])
        address_normalized = normalize_text(stop["address"])
        
        # Check if query matches name or address
        if q_normalized in name_normalized or q_normalized in address_normalized:
            # Validate coordinates are within Lusaka bbox
            if in_lusaka_bbox(stop["longitude"], stop["latitude"]):
                results.append({
                    "name": stop["name"],
                    "address": stop["address"],
                    "latitude": stop["latitude"],
                    "longitude": stop["longitude"],
                    "category": stop.get("category", "Unknown"),
                    "source": "kuyenda_local",
                })
            else:
                # Log stops that are outside bbox for debugging
                print(f"⚠️  Local stop outside bbox: {stop['name']} at ({stop['longitude']}, {stop['latitude']})")
    
    # 2) Mapbox Geocoding API (restricted to Lusaka via bbox)
    from urllib.parse import quote
    encoded_query = quote(query.strip())
    v5_url = f"https://api.mapbox.com/search/geocode/v6/forward"
    
    # Use v6 API with stricter bbox enforcement
    params = {
        "q": query.strip(),
        "access_token": MAPBOX_TOKEN,
        "country": "ZM",
        "bbox": bbox_str,                 # Bounding box filter
        "proximity": proximity_str,       # Bias to user location
        "types": "poi,address,place,street,neighborhood",
        "limit": 15,
        "language": "en",
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try v6 first, fallback to v5 if needed
            resp = await client.get(v5_url, params=params)
            
            # If v6 fails, fallback to v5
            if resp.status_code != 200:
                v5_url_fallback = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_query}.json"
                v5_params = {
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
                resp = await client.get(v5_url_fallback, params=v5_params)
                
    except httpx.TimeoutException:
        print(f"⚠️  Mapbox geocoding timed out for query: {query}")
        # Don't fail the entire request, just skip Mapbox results
        resp = None
    except Exception as e:
        print(f"⚠️  Mapbox geocoding error: {str(e)}")
        resp = None
    
    # Process Mapbox results
    if resp and resp.status_code == 200:
        data = resp.json()
        features = data.get("features", [])
        
        for feat in features:
            # Handle both v5 and v6 response formats
            geometry = feat.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            
            # v5 uses "center", v6 uses "geometry.coordinates"
            if not coordinates:
                center = feat.get("center", [])
                coordinates = center
            
            if len(coordinates) != 2:
                continue
            
            feat_lng, feat_lat = float(coordinates[0]), float(coordinates[1])
            
            # STRICT: Only include results within Lusaka bbox
            if not in_lusaka_bbox(feat_lng, feat_lat):
                print(f"⚠️  Mapbox result outside bbox: {feat.get('text')} at ({feat_lng}, {feat_lat})")
                continue
            
            # Get properties
            properties = feat.get("properties", {})
            name = feat.get("text") or feat.get("place_name", "Unknown")
            address = feat.get("place_name") or feat.get("properties", {}).get("full_address", name)
            
            results.append({
                "name": name,
                "address": address,
                "latitude": feat_lat,
                "longitude": feat_lng,
                "source": "mapbox",
            })
    elif resp:
        # Log error but don't crash
        print(f"❌ Mapbox geocode error: {resp.status_code} - {resp.text[:200]}")
    
    # 3) Deduplicate results
    seen = set()
    deduped = []
    
    for r in results:
        # Create unique key based on name and rounded coordinates
        name_key = normalize_text(r.get("name", ""))
        lng_rounded = round(float(r["longitude"]), 4)  # ~11m precision
        lat_rounded = round(float(r["latitude"]), 4)
        
        key = (name_key, lng_rounded, lat_rounded)
        
        if key in seen:
            continue
        
        seen.add(key)
        deduped.append(r)
    
    # 4) Final validation: ensure all results have valid coordinates
    valid_results = []
    for r in deduped:
        lat_val = r.get("latitude")
        lng_val = r.get("longitude")
        
        if lat_val is not None and lng_val is not None:
            # Double-check bbox (paranoid validation)
            if in_lusaka_bbox(lng_val, lat_val):
                valid_results.append(r)
            else:
                print(f"⚠️  Final filter caught: {r.get('name')} outside bbox")
    
    # 5) Sort: Local results first, then Mapbox
    valid_results.sort(key=lambda x: (
        0 if x.get("source") == "kuyenda_local" else 1,
        x.get("name", "").lower()  # Secondary sort by name
    ))
    
    # Limit to top 15 results
    return {"results": valid_results[:15]}


@app.get("/geocode/reverse")
@limiter.limit("30/minute")
async def reverse_geocode(request: Request, lat: float, lng: float):
    """
    Reverse geocode coordinates to place name/address.
    """
    if not validate_coords(lat, lng):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    
    # Mapbox expects lng,lat
    mapbox_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lng},{lat}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "limit": 1
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(mapbox_url, params=params)
    
    if resp.status_code != 200:
        print(f"Reverse Geocoding Error: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail="Reverse geocoding failed")
    
    data = resp.json()
    
    if data.get("features"):
        feature = data["features"][0]
        return {
            "name": feature.get("text"),
            "address": feature.get("place_name"),
            "latitude": lat,
            "longitude": lng
        }
    
    return {"name": f"{lat}, {lng}", "address": "Unknown location", "latitude": lat, "longitude": lng}

# --- Legacy Mock Endpoints (kept for app stability) ---

# Legacy endpoints removed or deprecated

class RouteRequest(BaseModel):
    origin: str
    destination: str

@app.post("/ai/predict")
async def predict_departure(route: RouteRequest):
    """
    Recommend departure time using the ML model.
    """
    session = SessionManager()
    session.current_route = f"{route.origin} → {route.destination}"
    traffic = await get_traffic_status(session)

    # Log traffic stat to Supabase
    try:
        log_traffic_stat(
            location=traffic['road'],
            lat=0.0, # Placeholder if GPS not in session
            lng=0.0,
            delay=traffic['delay'],
            level=traffic['level']
        )
    except Exception as e:
        print(f"Failed to log traffic stat: {e}")

    if traffic['delay'] > 20:
        recommended = "Wait 30 mins"
        message = f"Heavy traffic on {traffic['road']}. Better to wait."
    else:
        recommended = "Now"
        message = f"Roads are clear on {traffic['road']}. Safe travels!"

    return {
        "status": "success",
        "recommended_departure_in": recommended,
        "delay_minutes": traffic['delay'],
        "message": message
    }

@app.post("/voice/command")
async def process_voice_command(text: str = Query(...)):
    """
    Process voice commands (English + Bemba keywords).
    """
    cmd = text.lower()
    
    # 1. Traffic Query
    if any(w in cmd for w in ["traffic", "jam", "motoka", "blocked"]):
        session = SessionManager()
        traffic = await get_traffic_status(session)
        return {"response": f"Traffic on {traffic['road']}: {traffic['level']}, delay {traffic['delay']} minutes."}
        
    # 2. Departure/Prediction
    elif any(w in cmd for w in ["leave", "time", "nthawi", "departure", "go"]):
        # Reuse logic (mock call with default route)
        pred = await predict_departure(RouteRequest(origin="Voice", destination="Voice"))
        return {"response": f"Recommendation: {pred['recommended_departure_in']}. {pred['message']}"}

    # 3. Navigation
    elif any(w in cmd for w in ["navigate", "start", "yamba", "map", "direction"]):
         return {"response": "Starting navigation. Please select your destination on the screen.", "action": "NAVIGATE"}

    # Default
    response_text = "I didn't quite catch that. Try asking about traffic or when to leave."
    
    # Log to Supabase (Mocking user ID as "legacy_http" if not authenticated)
    # Ideally this endpoint should also have Depends(get_current_user)
    log_voice_command("legacy_http", text, response_text, "manual_command")
    
    return {"response": response_text}

if __name__ == "__main__":
    # Run with 0.0.0.0 to allow mobile connections
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
