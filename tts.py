
import httpx
import logging
from config import CARTESIA_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def speak(text: str) -> bytes:
    """
    Generate TTS audio bytes using Cartesia's /tts/bytes endpoint.
    Returns WAV audio bytes ready to send over WebSocket.
    """
    if not text.strip():
        return b''

    url = "https://api.cartesia.ai/tts/bytes"

    headers = {
        "Cartesia-Version": "2025-04-16",  # Use latest stable version (check docs if needed)
        "X-API-Key": CARTESIA_API_KEY,     # Preferred header name
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": "sonic-3",  # Latest high-quality model (or "sonic-english" / "sonic-turbo")
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": "694f9389-aac1-45b6-b726-9d9369183238"  # Popular calm male voice – replace if you want "calm-driver" equivalent
        },
        "language": "en",
        "output_format": {
            "container": "wav",        # Easy to play in Expo (expo-av supports WAV)
            "encoding": "pcm_s16le",   # Standard 16-bit PCM
            "sample_rate": 44100       # High quality; use 24000 for smaller files
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            logger.info(f"Cartesia TTS success: {len(response.content)} bytes generated")
            return response.content
        else:
            logger.error(f"Cartesia TTS error {response.status_code}: {response.text}")
            return b''  # Fallback silence

    except Exception as e:
        logger.error(f"Cartesia TTS exception: {e}")
        return b''