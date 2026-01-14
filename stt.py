# Robust STT code compatible with Deepgram SDK v5+
from deepgram import DeepgramClient
import logging

# Initialize the client simply
try:
    dg_client = DeepgramClient(api_key=None) # Will use DEEPGRAM_API_KEY from env if available
    # Or explicitly:
    from config import DEEPGRAM_API_KEY
    dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize Deepgram client: {e}")
    dg_client = None

async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe in-memory audio bytes using Deepgram.
    This uses the asynchronous-friendly call structure.
    """
    if dg_client is None:
        return ""

    try:
        if not audio_bytes:
            return ""

        # Using dictionary for options for max compatibility across SDK versions
        payload = {"buffer": audio_bytes}
        options = {
            "model": "nova-2",
            "smart_format": True,
            "utterances": True,
            "detect_language": True,
        }

        # The v1.transcribe_file is async-safe in modern versions
        response = dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # Check if response is a coroutine (some SDK versions require await)
        import inspect
        if inspect.iscoroutine(response):
            response = await response

        if not response or not response.results:
            return ""

        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript.strip()

    except Exception as e:
        logging.error(f"Deepgram transcription error: {e}")
        return ""