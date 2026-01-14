# app/ws_voice.py  (or wherever your router lives)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.responses import Response
from jose import JWTError, jwt
import logging

# Import your existing modules
from stt import transcribe_audio
from intent import detect_intent
from traffic import get_traffic_status
from tts import speak
from session import SessionManager
from config import SECRET_KEY, ALGORITHM

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse your JWT validation logic (copy/adapt from main.py)
def validate_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise JWTError
        return user_id
    except JWTError:
        raise ValueError("Invalid or expired token")

@router.websocket("/ws/voice")
async def voice_ws(
    websocket: WebSocket,
    token: str = Query(..., description="JWT token for authentication")
):

    try:
        validate_token(token)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info("Authenticated WebSocket connection established")

    session = SessionManager()

    # Simple rate limit: max 10 messages per 10 seconds (adjust as needed)
    messages_received = 0
    start_time = None

    try:
        while True:
            # Basic rate limiting per connection
            if start_time is None:
                start_time = logging.time.time()
            elapsed = logging.time.time() - start_time
            if elapsed > 10:
                messages_received = 0
                start_time = logging.time.time()

            if messages_received >= 10:
                await websocket.send_text("Rate limit exceeded. Please slow down.")
                continue

            # 2️⃣ Receive audio chunk (binary)
            audio_bytes = await websocket.receive_bytes()
            messages_received += 1

            # 3️⃣ Speech → Text
            text = await transcribe_audio(audio_bytes)
            if not text.strip():
                logger.info("Empty transcription – skipping")
                continue

            logger.info(f"Transcribed: {text}")

            # 4️⃣ Intent detection
            intent = detect_intent(text)

            if intent["intent"] == "GET_TRAFFIC_STATUS":
                traffic = await get_traffic_status(session)  # Now uses your loaded ML model!

                response_text = (
                    f"There is {'heavy' if traffic['level'] == 'HIGH' else 'light'} traffic on {traffic['road']}. "
                    f"Estimated delay: {traffic['delay']} minutes."
                )

                # 5️⃣ Text → Speech
                audio_response = await speak(response_text)

                if audio_response:
                    await websocket.send_bytes(audio_response)
                    logger.info("Sent audio response")
                else:
                    await websocket.send_text("Sorry, I couldn't generate a response audio.")
            else:
                # Optional: fallback response for unknown intents
                fallback_text = "I didn't understand that. Try asking about traffic on your route."
                fallback_audio = await speak(fallback_text)
                await websocket.send_bytes(fallback_audio)

    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass