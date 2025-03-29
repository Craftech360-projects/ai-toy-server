import sounddevice as sd
import soundfile as sf
import numpy as np
import socket
import time
import threading
import sys
import os
import requests # For HTTP fetching
import paho.mqtt.client as mqtt # For MQTT
import pygame # For audio playback
from queue import Queue, Empty # Import Empty exception explicitly

# --- Configuration ---
SERVER_UDP_HOST = '127.0.0.1'
SERVER_UDP_PORT = 5005
SERVER_HTTP_HOST = '127.0.0.1' # Assuming server runs on localhost
SERVER_HTTP_PORT = 5006
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCK_DURATION = 0.1
BUFFER_SIZE = int(SAMPLE_RATE * BLOCK_DURATION * CHANNELS * np.dtype(DTYPE).itemsize)
TEMP_AUDIO_DIR = "temp_client_audio" # Directory to temporarily store downloaded audio

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_NEW_AUDIO = "buddy/audio/new"
MQTT_TOPIC_PLAYBACK_FINISHED = "buddy/audio/finished"
MQTT_CLIENT_ID_CLIENT = f"buddy_client_{os.getpid()}" # Unique client ID

# --- Global Variables ---
stream_active = False
udp_socket = None
stream_thread = None
mqtt_client_client = None
audio_playback_queue = Queue()
playback_active = threading.Event() # To signal if playback is happening
client_running = True # Flag to control main loop and threads

# --- Audio Callback Function (UDP Streaming) ---
def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block to send via UDP."""
    global stream_active, udp_socket
    if status: print(f"Audio callback status: {status}", file=sys.stderr)
    if stream_active and udp_socket:
        try:
            audio_bytes = indata.astype(DTYPE).tobytes()
            udp_socket.sendto(audio_bytes, (SERVER_UDP_HOST, SERVER_UDP_PORT))
        except Exception as e:
            print(f"Error sending audio data: {e}", file=sys.stderr)

# --- Streaming Thread Function ---
def start_streaming_thread():
    """Handles the audio recording stream."""
    global stream_active, udp_socket, client_running
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                            blocksize=int(SAMPLE_RATE * BLOCK_DURATION), callback=audio_callback):
            print("[*] Recording started. Streaming audio...")
            while stream_active and client_running:
                time.sleep(0.1)
            print("[*] Recording stopped.")
    except sd.PortAudioError as e:
         print(f"\n[Error] Sounddevice error: {e}", file=sys.stderr)
         print("Please ensure a working microphone is connected/selected.", file=sys.stderr)
         stream_active = False
    except Exception as e:
        print(f"\n[Error] Streaming thread error: {e}", file=sys.stderr)
        stream_active = False

# --- MQTT Callbacks ---
def on_connect_client(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT Client] Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC_NEW_AUDIO)
        print(f"[MQTT Client] Subscribed to topic: {MQTT_TOPIC_NEW_AUDIO}")
    else:
        print(f"[MQTT Client] Failed to connect, return code {rc}\n")

def on_message_client(client, userdata, msg):
    """Handles incoming MQTT messages for the client (new audio notifications)."""
    topic = msg.topic
    filename = msg.payload.decode("utf-8")
    print(f"\n[MQTT Client] Received notification for new audio: {filename}")
    if topic == MQTT_TOPIC_NEW_AUDIO:
        # Add filename to the playback queue
        print(f"[Queue] Adding {filename} to playback queue.") # Debug print
        audio_playback_queue.put(filename)

# --- MQTT Client Setup ---
def setup_mqtt_client():
    global mqtt_client_client
    mqtt_client_client = mqtt.Client(client_id=MQTT_CLIENT_ID_CLIENT, protocol=mqtt.MQTTv5)
    mqtt_client_client.on_connect = on_connect_client
    mqtt_client_client.on_message = on_message_client
    try:
        mqtt_client_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client_client.loop_start() # Start network loop
    except Exception as e:
        print(f"[MQTT Client] Error connecting to MQTT broker: {e}")
        mqtt_client_client = None

# --- Audio Playback Thread ---
def audio_playback_thread():
    """Fetches and plays audio files from the queue, and sends confirmation."""
    global client_running, playback_active, mqtt_client_client

    print("[Playback Thread] Starting main loop...")
    while client_running:
        try:
            filename = audio_playback_queue.get(timeout=1)
            if not filename: continue

            print(f"[Playback] Processing file: {filename}")
            playback_active.set()

            # 1. Fetch audio via HTTP
            audio_url = f"http://{SERVER_HTTP_HOST}:{SERVER_HTTP_PORT}/audio/{filename}"
            local_filepath = os.path.join(TEMP_AUDIO_DIR, filename)
            downloaded = False
            try:
                print(f"[Playback] Downloading {audio_url}...")
                os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
                response = requests.get(audio_url, stream=True, timeout=10)
                response.raise_for_status()
                with open(local_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[Playback] Downloaded to {local_filepath}")
                downloaded = True
            except requests.exceptions.RequestException as e:
                print(f"[Playback] Error downloading {filename}: {e}")
            except Exception as e:
                print(f"[Playback] Error saving {filename}: {e}")

            # 2. Play audio if downloaded
            if downloaded:
                try:
                    print(f"[Playback] Playing {local_filepath}...")
                    # Load and play the audio using sounddevice
                    data, samplerate = sf.read(local_filepath)
                    sd.play(data, samplerate)
                    sd.wait()  # Wait until audio is done playing
                    
                    print(f"[Playback] Finished playing {filename}.")
                    
                    # 3. Send MQTT confirmation
                    if mqtt_client_client and mqtt_client_client.is_connected():
                        result = mqtt_client_client.publish(
                            MQTT_TOPIC_PLAYBACK_FINISHED, 
                            payload=filename, 
                            qos=1
                        )
                        if result.rc == mqtt.MQTT_ERR_SUCCESS:
                            print(f"[MQTT Client] Published finished confirmation for {filename}")
                        else:
                            print(f"[MQTT Client] Failed to publish confirmation. RC: {result.rc}")
                    else:
                        print("[MQTT Client] Cannot publish confirmation, client not connected.")

                except Exception as e:
                    print(f"[Playback] Error during playback/confirmation: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(local_filepath):
                        try:
                            os.remove(local_filepath)
                            print(f"[Playback] Cleaned up {local_filepath}")
                        except OSError as e:
                            print(f"[Playback] Error deleting temp file: {e}")
            else:
                print(f"[Playback] Skipping playback due to download error.")

            audio_playback_queue.task_done()
            playback_active.clear()

        except Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"[Playback Thread Error] Unexpected error: {e}", file=sys.stderr)
            time.sleep(1)

# --- Main Control Logic ---
def main():
    global stream_active, udp_socket, stream_thread, client_running, mqtt_client_client

    print("Audio Streaming Client")
    print("----------------------")
    print("Initializing MQTT and Playback...")

    # Setup MQTT
    setup_mqtt_client()
    if not mqtt_client_client:
         print("[Error] MQTT Client setup failed. Exiting.", file=sys.stderr)
         return

    # Start Playback Thread
    playback_thread = threading.Thread(target=audio_playback_thread, daemon=True)
    playback_thread.start()

    print("Commands: start, stop, exit")

    try:
        while client_running:
            command = input("> ").strip().lower()

            if command == "start":
                if not stream_active:
                    try:
                        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        print(f"[*] Connecting UDP stream to server at {SERVER_UDP_HOST}:{SERVER_UDP_PORT}...")
                        stream_active = True
                        stream_thread = threading.Thread(target=start_streaming_thread, daemon=True)
                        stream_thread.start()
                    except Exception as e:
                        print(f"[Error] Failed to initialize UDP socket or start stream: {e}", file=sys.stderr)
                        if udp_socket: udp_socket.close()
                        udp_socket = None
                        stream_active = False
                else:
                    print("[!] Already recording.")

            elif command == "stop":
                if stream_active:
                    print("[*] Stopping recording...")
                    stream_active = False
                    if stream_thread and stream_thread.is_alive():
                        stream_thread.join(timeout=1.0)
                    if udp_socket:
                        udp_socket.close()
                        udp_socket = None
                    stream_thread = None
                    print("[*] UDP Stream stopped and socket closed.")
                else:
                    print("[!] Not currently recording.")

            elif command == "exit":
                print("[*] Exiting client...")
                client_running = False # Signal threads to stop
                if stream_active: # Stop recording if active
                     print("[*] Stopping recording before exit...")
                     stream_active = False
                     if stream_thread and stream_thread.is_alive():
                         stream_thread.join(timeout=1.0)
                     if udp_socket: udp_socket.close()
                break # Exit main loop

            else:
                print(f"[?] Unknown command: '{command}'. Use 'start', 'stop', or 'exit'.")

    except KeyboardInterrupt:
         print("\n[*] Exiting client (Ctrl+C)...")
         client_running = False
         if stream_active:
             stream_active = False
             if stream_thread and stream_thread.is_alive():
                 stream_thread.join(timeout=1.0)
             if udp_socket: udp_socket.close()

    finally:
        # Cleanup threads and connections
        print("[*] Cleaning up...")
        if mqtt_client_client:
            mqtt_client_client.loop_stop()
            mqtt_client_client.disconnect()
        if playback_thread.is_alive():
             # Signal playback thread to stop waiting indefinitely if it's stuck on queue.get()
             # A more robust way might involve a sentinel value in the queue
             audio_playback_queue.put(None) # Put a None to unblock .get() if waiting
             playback_thread.join(timeout=2.0) # Wait for playback thread
        print("[*] Client finished.")


if __name__ == "__main__":
    main()
