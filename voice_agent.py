import os
import io
import time
import json
import threading
import asyncio
from datetime import datetime
from typing import Optional, Callable
import wave

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    AgentV1Agent,
    AgentV1AudioConfig,
    AgentV1AudioInput,
    AgentV1AudioOutput,
    AgentV1DeepgramSpeakProvider,
    AgentV1Listen,
    AgentV1ListenProvider,
    AgentV1OpenAiThinkProvider,
    AgentV1SettingsMessage,
    AgentV1SocketClientResponse,
    AgentV1SpeakProviderConfig,
    AgentV1Think,
)

class KuyendaVoiceAgent:
    def __init__(self, api_key: str, on_audio_callback: Callable[[bytes], None]):
        self.api_key = api_key
        self.on_audio_callback = on_audio_callback
        self.client = DeepgramClient(api_key=api_key)
        self.connection = None
        self.loop = asyncio.get_event_loop()
        self.connected_event = asyncio.Event()
        self.processing_complete = asyncio.Event()
        self.audio_buffer = bytearray()
        self.user_transcription = ""
        self.agent_response = ""

    async def connect(self):
        self.connection = self.client.agent.v1.connect()
        self.connection.on(EventType.Open, lambda e: self.loop.call_soon_threadsafe(self.connected_event.set))
        self.connection.on(EventType.Message, self._on_message)
        self.connection.on(EventType.Error, lambda e: print(f"Deepgram Agent Error: {e}"))
        self.connection.on(EventType.Close, lambda e: print("Deepgram Agent Closed"))
        
        # Start the connection
        self.connection.start()
        
        # Wait for the connection to be open
        await self.connected_event.wait()

        # Configure the Agent
        settings = AgentV1SettingsMessage(
            audio=AgentV1AudioConfig(
                input=AgentV1AudioInput(
                    encoding="linear16",
                    sample_rate=24000,
                ),
                output=AgentV1AudioOutput(
                    encoding="linear16",
                    sample_rate=24000,
                    container="wav", # We want WAV for compatibility with mobile expo-av
                ),
            ),
            agent=AgentV1Agent(
                language="en",
                listen=AgentV1Listen(
                    provider=AgentV1ListenProvider(
                        type="deepgram",
                        model="nova-3",
                    )
                ),
                think=AgentV1Think(
                    provider=AgentV1OpenAiThinkProvider(
                        type="open_ai",
                        model="gpt-4o-mini",
                    ),
                    prompt=(
                        "You are Kuyenda, a friendly AI driving assistant in Lusaka, Zambia. "
                        "You help users with traffic and directions. "
                        "When asked for the 'shortest' route, assure the user you are looking for the minimum distance path. "
                        "If they ask for the 'fastest' or 'quickest' route, focus on the shortest time. "
                        "IMPORTANT: If the user DOES NOT start their sentence with 'Kuyenda' or 'Hey Kuyenda', "
                        "and they are not clearly asking you a direct question, respond with exactly 'IGNORE_ME'. "
                        "Otherwise, help users. Keep responses brief and helpful."
                    ),
                ),
                speak=AgentV1SpeakProviderConfig(
                    provider=AgentV1DeepgramSpeakProvider(
                        type="deepgram",
                        model="aura-2-thalia-en",
                    )
                ),
                greeting="Hello! I'm your Kuyenda co-pilot. Where are we heading?",
            ),
        )
        self.connection.send_settings(settings)

    def _on_message(self, message: AgentV1SocketClientResponse):
        # Handle binary audio data
        if isinstance(message, bytes):
            self.on_audio_callback(message)
            return

        msg_type = getattr(message, "type", "Unknown")
        if msg_type == "AgentAudioDone":
            self.loop.call_soon_threadsafe(self.processing_complete.set)
        elif msg_type == "UserStartedSpeaking":
             pass
        elif msg_type == "ConversationText":
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
            if role == "user":
                self.user_transcription = content
            elif role == "assistant":
                self.agent_response = content
                
            if "IGNORE_ME" in content.upper():
                print("Wake word not detected / Not addressed. Ignoring.")
                self.audio_buffer = bytearray() # Clear buffer
                self.loop.call_soon_threadsafe(self.processing_complete.set) # End early
        elif msg_type == "AgentThought":
            print(f"Agent Thought: {getattr(message, 'thought', '')}")

    def send_audio(self, chunk: bytes):
        if self.connection:
            self.connection.send_audio(chunk)

    def stop(self):
        if self.connection:
            self.connection.finish()

async def process_thinking_voice(audio_bytes: bytes, api_key: str):
    """
    Helper to process a chunk of audio through the Thinking Agent and return the audio response.
    """
    response_audio = bytearray()
    
    def on_audio(chunk):
        response_audio.extend(chunk)

    agent = KuyendaVoiceAgent(api_key, on_audio)
    await agent.connect()
    
    # Send all bytes (assuming it's a small recorded segment from mobile)
    agent.send_audio(audio_bytes)
    
    # Wait for response (timeout 10s)
    try:
        await asyncio.wait_for(agent.processing_complete.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("Agent response timeout")
    
    transcription = agent.user_transcription
    response_text = agent.agent_response
    
    agent.stop()
    return bytes(response_audio), transcription, response_text
