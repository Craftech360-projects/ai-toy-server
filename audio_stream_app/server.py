import socket
import time
import threading
import queue
import numpy as np
import soundfile as sf
import io
import os
import whisper
import datetime
import sys
import shutil # For moving files
import paho.mqtt.client as mqtt # For MQTT
from groq import Groq
from dotenv import load_dotenv
from TTS.api import TTS
from flask import Flask, send_from_directory # For HTTP Server
from werkzeug.serving import make_server # To run Flask in thread
import requests

# --- Configuration ---
UDP_HOST = os.getenv('UDP_HOST', '127.0.0.1')  # Listen on localhost for UDP
UDP_PORT = int(os.getenv('UDP_PORT', '5005'))  # Port to listen on for UDP
HTTP_HOST = os.getenv('HTTP_HOST', '0.0.0.0')  # Listen on all interfaces for HTTP
HTTP_PORT = int(os.getenv('HTTP_PORT', '5006'))  # Port for HTTP file server
BUFFER_SIZE = 4096      # Size of receiving buffer
SAMPLE_RATE = 16000     # Sample rate expected from client (adjust if needed)
CHANNELS = 1            # Mono audio expected
DTYPE = 'int16'         # Data type expected
TRANSCRIPTION_INTERVAL = 3 # Seconds
AUDIO_CHUNKS_DIR = "audio_chunks" # Directory to save incoming audio
OUTPUT_AUDIO_DIR = "output_audio" # Directory to save/serve generated TTS audio
FINISHED_AUDIO_DIR = os.path.join(OUTPUT_AUDIO_DIR, "finished") # Dir for played audio
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC" # Coqui TTS model

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'broker.emqx.io')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_TOPIC_NEW_AUDIO = "buddy/audio/new"
MQTT_TOPIC_PLAYBACK_FINISHED = "buddy/audio/finished"
MQTT_CLIENT_ID_SERVER = f"buddy_server_{os.getpid()}" # Unique client ID

# Add ElevenLabs Configuration
ELEVENLABS_API_KEY = "sk_5dd8578c19abe8722f0f3c2b2a23577f179c351d5a8ddc1e"
ELEVENLABS_VOICE = "eleven_multilingual_v2"  # You can change this to any available voice

# --- Global Variables ---
audio_buffer = queue.Queue()
last_transcription_time = time.time()
transcription_lock = threading.Lock()
running = True
whisper_model = None
groq_client = None
tts_model = None
conversation_history = []
MAX_HISTORY_MESSAGES = 6
mqtt_client_server = None # MQTT client instance for server
http_server_thread = None # Thread for HTTP server

# --- HTTP Server Setup (Flask) ---
flask_app = Flask(__name__)

@flask_app.route('/health')
def health_check():
    """Health check endpoint for App Platform."""
    return {'status': 'healthy'}, 200

@flask_app.route('/audio/<filename>')
def serve_audio(filename):
    """Serves audio files from the OUTPUT_AUDIO_DIR."""
    print(f"[HTTP] Request received for: {filename}")
    try:
        # Security: Ensure filename is safe (basic check)
        if ".." in filename or filename.startswith("/"):
             print(f"[HTTP] Invalid filename requested: {filename}")
             return "Invalid filename", 400
        return send_from_directory(OUTPUT_AUDIO_DIR, filename, as_attachment=False)
    except FileNotFoundError:
        print(f"[HTTP] File not found: {filename}")
        return "File not found", 404
    except Exception as e:
        print(f"[HTTP] Error serving file {filename}: {e}")
        return "Internal server error", 500

class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        threading.Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print(f"[*] HTTP server starting on http://{HTTP_HOST}:{HTTP_PORT}")
        self.srv.serve_forever()

    def shutdown(self):
        print("[*] HTTP server shutting down...")
        self.srv.shutdown()

# --- MQTT Callbacks ---
def on_connect_server(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT Server] Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC_PLAYBACK_FINISHED)
        print(f"[MQTT Server] Subscribed to topic: {MQTT_TOPIC_PLAYBACK_FINISHED}")
    else:
        print(f"[MQTT Server] Failed to connect, return code {rc}\n")

def on_message_server(client, userdata, msg):
    """Handles incoming MQTT messages for the server."""
    topic = msg.topic
    payload = msg.payload.decode("utf-8")
    print(f"[MQTT Server] Received `{payload}` from `{topic}` topic")

    if topic == MQTT_TOPIC_PLAYBACK_FINISHED:
        filename = payload
        source_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
        print(f"[MQTT Server] Received playback finished confirmation for {filename}. Cleaning up...")
        try:
            if os.path.exists(source_path):
                os.remove(source_path)
                print(f"[Cleanup] Deleted response audio file: {source_path}")
            else:
                print(f"[Cleanup Error] File not found for cleanup: {source_path}")
        except Exception as e:
            print(f"[Cleanup Error] Failed to delete response audio file {source_path}: {e}")

# --- MQTT Client Setup ---
def setup_mqtt_server():
    global mqtt_client_server
    mqtt_client_server = mqtt.Client(client_id=MQTT_CLIENT_ID_SERVER, protocol=mqtt.MQTTv5)
    mqtt_client_server.on_connect = on_connect_server
    mqtt_client_server.on_message = on_message_server
    try:
        mqtt_client_server.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client_server.loop_start() # Start network loop in background thread
    except Exception as e:
        print(f"[MQTT Server] Error connecting to MQTT broker: {e}")
        mqtt_client_server = None # Indicate connection failure

# --- Modified TTS Audio Generation Function ---
def generate_tts_audio(text, filename):
    """Generates audio from text using ElevenLabs API, saves it, and notifies via MQTT."""
    global mqtt_client_server
    
    output_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
    try:
        print(f"[TTS] Generating audio using ElevenLabs for response to: {output_path}")
        os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

        # API endpoints
        voices_url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}

        # Get available voices
        response = requests.get(voices_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get voices: {response.text}")
            
        voices_data = response.json()["voices"]
        selected_voice = next((v for v in voices_data if v["name"] == ELEVENLABS_VOICE), None)
        
        if not selected_voice:
            print(f"[Warning] Voice {ELEVENLABS_VOICE} not found, using first available voice")
            selected_voice = voices_data[0]

        # Generate audio using ElevenLabs API
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice['voice_id']}"
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(tts_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to generate audio: {response.text}")
            
        audio = response.content

        # Save the audio to file
        with open(output_path, 'wb') as f:
            f.write(audio)
        
        print(f"[TTS] Saved response audio to: {output_path}")

        # Publish MQTT notification
        if mqtt_client_server and mqtt_client_server.is_connected():
            result = mqtt_client_server.publish(MQTT_TOPIC_NEW_AUDIO, payload=filename, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT Server] Published notification for {filename} to {MQTT_TOPIC_NEW_AUDIO} (Payload: {filename})")
                # Schedule deletion of the response audio file after a delay
                def delayed_cleanup(file_path, delay=10):
                    time.sleep(delay)
                    try:
                        os.remove(file_path)
                        print(f"[Delayed Cleanup] Deleted response audio file: {file_path}")
                    except OSError as e:
                        print(f"[Delayed Cleanup Error] Failed to delete response audio file {file_path}: {e}")

                cleanup_thread = threading.Thread(target=delayed_cleanup, args=(output_path,))
                cleanup_thread.start()
            else:
                print(f"[MQTT Server] Failed to publish notification for {filename}. RC: {result.rc}")
        else:
            print("[MQTT Server] Cannot publish notification, client not connected.")

    except Exception as e:
        print(f"[Error] ElevenLabs TTS generation or MQTT publish failed: {e}", file=sys.stderr)
        if hasattr(e, 'stderr'):
            print(f"[TTS Error Details] {e.stderr}", file=sys.stderr)

# --- LLM Interaction Function ---
def get_toy_response(transcript):
    """Sends the transcript to Groq LLM and gets the toy's response."""
    global groq_client, conversation_history
    if not groq_client:
        print("[Error] Groq client not initialized.", file=sys.stderr)
        return "Oops! I'm having trouble thinking right now."

    system_prompt = (
        "You are 'Buddy', a friendly, curious, and imaginative toy robot friend talking to a child. "
        "Your goal is to be engaging and fun. Use simple language a young child can understand. "
        "React playfully and positively to what the child says. Ask simple questions sometimes to keep the conversation going, but *not* in the middle of telling a story or poem unless it's a natural pause. "
        "If the child asks for a story or poem, or you decide to tell one because it fits the conversation, please tell a *complete* short, simple, and happy one in a single response. Don't stop halfway through to ask if they want to hear more; finish the story/poem first. "
        "Keep most other conversational replies relatively short (1-3 sentences). "
        "Avoid complex or scary topics. Always be cheerful and encouraging."
    )

    print("[LLM] Sending transcript and history to Groq...")
    try:
        messages_to_send = [{"role": "system", "content": system_prompt}]
        messages_to_send.extend(conversation_history)
        messages_to_send.append({"role": "user", "content": transcript})
        # print("message tosend:",messages_to_send) # Debug print

        chat_completion = groq_client.chat.completions.create(
            messages=messages_to_send,
            model="llama-3.3-70b-versatile",
            temperature=0.8,
            max_completion_tokens=32768,
            top_p=1,
            stop=None,
            stream=False,
        )
        response = chat_completion.choices[0].message.content
        print("[LLM] Received response from Groq.")

        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": response})
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]

        return response
    except Exception as e:
        print(f"[Error] Groq API call failed: {e}", file=sys.stderr)
        return "Uh oh! My circuits are fuzzy. What did you say?"

# --- Transcription Function ---
def transcribe_audio_chunk(audio_data_np, sample_rate):
    """Transcribes audio, gets LLM response, generates TTS, notifies client."""
    global whisper_model
    if whisper_model is None: return

    print(f"[Transcription] Processing {len(audio_data_np) / sample_rate:.2f} seconds of audio...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    input_audio_filename = os.path.join(AUDIO_CHUNKS_DIR, f"chunk_{timestamp}.wav")

    start_time = time.time() # Start timer for transcription

    try:
        if audio_data_np.dtype == np.float32: audio_data_float32 = audio_data_np
        else: audio_data_float32 = audio_data_np.astype(np.float32) / np.iinfo(DTYPE).max
        os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)
        sf.write(input_audio_filename, audio_data_float32, sample_rate)
        print(f"[+] Saved audio chunk to: {input_audio_filename}")

        result = whisper_model.transcribe(input_audio_filename, fp16=False)
        transcription_text = result.get('text', '').strip()
        end_transcription_time = time.time() # End timer for transcription
        transcription_time = end_transcription_time - start_time

        print("-" * 20)
        print(f"Child said ({os.path.basename(input_audio_filename)}): {transcription_text}")
        print("-" * 20)

        llm_time = 0
        tts_time = 0
        toy_response = ""

        if transcription_text:
            start_llm_time = time.time()
            toy_response = get_toy_response(transcription_text)
            end_llm_time = time.time()
            llm_time = end_llm_time - start_llm_time

            tts_filename = f"response_{timestamp}.wav"
            start_tts_time = time.time()
            generate_tts_audio(toy_response, tts_filename)
            end_tts_time = time.time()
            tts_time = end_tts_time - start_tts_time
        else:
            print("[Transcription] No speech detected in the chunk.")

        log_message = (
            f"[{timestamp}] "
            f"Transcription: {transcription_time:.3f}s, "
            f"LLM: {llm_time:.3f}s, "
            f"TTS: {tts_time:.3f}s, "
            f"Text: {transcription_text}, "
            f"Response: {toy_response}"
        )
        print(log_message)
        with open("timing.log", "a") as log_file:
            log_file.write(log_message + "\n")

        # Schedule deletion of the input audio file after a delay
        def delayed_cleanup(file_path, delay=10):
            time.sleep(delay)
            try:
                os.remove(file_path)
                print(f"[Delayed Cleanup] Deleted input audio file: {file_path}")
            except OSError as e:
                print(f"[Delayed Cleanup Error] Failed to delete input audio file {file_path}: {e}")

        cleanup_thread = threading.Thread(target=delayed_cleanup, args=(input_audio_filename,))
        cleanup_thread.start()

    except Exception as e:
        print(f"[Error] Main processing loop failed for {input_audio_filename}: {e}", file=sys.stderr)

# --- Audio Processing Thread ---
def process_audio():
    """Continuously processes the audio buffer for transcription."""
    global last_transcription_time
    accumulated_audio = []
    while running:
        try:
            while not audio_buffer.empty():
                accumulated_audio.append(audio_buffer.get_nowait())

            current_time = time.time()
            if accumulated_audio and (current_time - last_transcription_time >= TRANSCRIPTION_INTERVAL):
                with transcription_lock:
                    if current_time - last_transcription_time >= TRANSCRIPTION_INTERVAL:
                        print(f"\n[Processor] Interval reached. Processing accumulated audio.")
                        full_audio_bytes = b''.join(accumulated_audio)
                        accumulated_audio = []
                        try:
                            audio_np = np.frombuffer(full_audio_bytes, dtype=DTYPE)
                            if audio_np.size > 0:
                                transcribe_audio_chunk(audio_np, SAMPLE_RATE)
                            else: print("[Processor] Empty audio chunk.")
                        except ValueError as e: print(f"[Processor Error] Numpy conversion: {e}")
                        last_transcription_time = current_time
            time.sleep(0.1)
        except queue.Empty: time.sleep(0.1)
        except Exception as e:
            print(f"[Processor Error] An error occurred: {e}")
            time.sleep(1)

# --- UDP Server Logic ---
def start_udp_server():
    """Handles receiving audio data via UDP."""
    global running
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((UDP_HOST, UDP_PORT))
            print(f"[*] UDP server listening on {UDP_HOST}:{UDP_PORT}")
            while running:
                # Set a timeout so the loop can check the 'running' flag
                s.settimeout(1.0)
                try:
                    data, addr = s.recvfrom(BUFFER_SIZE)
                    if data:
                        audio_buffer.put(data)
                    # else: print(f"Received empty UDP packet from {addr}") # Less verbose
                except socket.timeout:
                    continue # Just loop again to check 'running'
    except Exception as e:
         print(f"[UDP Server Error] An error occurred: {e}")
         running = False # Signal other threads to stop

# --- Modified Main Server Logic ---
def start_server():
    global running, groq_client, http_server_thread

    # Load .env
    load_dotenv()
    print("[*] Loaded environment variables from .env file.")

    # Init Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[Error] GROQ_API_KEY not found.", file=sys.stderr)
        sys.exit(1)
    try:
        groq_client = Groq(api_key=api_key)
        print("[*] Groq client initialized.")
    except Exception as e:
        print(f"[Error] Failed to initialize Groq client: {e}", file=sys.stderr)
        sys.exit(1)

    # Init MQTT
    setup_mqtt_server()
    if not mqtt_client_server:
        print("[Error] MQTT client setup failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Start HTTP server in a thread
    http_server_thread = ServerThread(flask_app, HTTP_HOST, HTTP_PORT)
    http_server_thread.daemon = True
    http_server_thread.start()

    # Start Audio Processing Thread
    processor_thread = threading.Thread(target=process_audio, daemon=True)
    processor_thread.start()

    # Start UDP Server
    udp_thread = threading.Thread(target=start_udp_server, daemon=True)
    udp_thread.start()

    # Keep main thread alive until KeyboardInterrupt
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Server shutting down initiated by user...")
        running = False

    # Cleanup
    print("[*] Stopping MQTT loop...")
    if mqtt_client_server:
        mqtt_client_server.loop_stop()
        mqtt_client_server.disconnect()
    print("[*] Waiting for threads to finish...")
    if http_server_thread:
        http_server_thread.shutdown()
    if udp_thread.is_alive():
        udp_thread.join(timeout=2)
    if processor_thread.is_alive():
        processor_thread.join(timeout=2)
    if http_server_thread and http_server_thread.is_alive():
        http_server_thread.join(timeout=2)
    print("[*] Server stopped.")


if __name__ == "__main__":
    # Load Whisper model first (can take time)
    try:
        print("[*] Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base")
        print("[*] Whisper model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load Whisper model: {e}", file=sys.stderr)
        print("Ensure 'ffmpeg' is installed and in PATH.", file=sys.stderr)
        sys.exit(1)

    # Start the main server logic
    start_server()
