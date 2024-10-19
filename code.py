import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import os
import json
from vosk import Model, KaldiRecognizer
from datetime import datetime

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    print("Recording audio...")
    audio_data = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype=np.int16)
    sd.wait()
    print("Recording completed")
    return audio_data

def save_audio(audio_data, filename="temp.wav", sample_rate=16000):
    """Save recorded audio to file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def save_text_to_file(text):
    """Save transcribed text to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text saved successfully to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving text to file: {str(e)}")
        return None

def transcribe_audio():
    """Convert speech to text using Vosk"""
    model_path = "vosk-model-small-fa-0.5"
    
    if not os.path.exists(model_path):
        print("Please download the Vosk Persian model and place it in the program directory")
        return

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)

    wf = wave.open("temp.wav", "rb")
    
    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            # Parse JSON result to extract text
            json_result = json.loads(rec.Result())
            if 'text' in json_result:
                result += json_result['text'] + " "

    # Get final result
    json_result = json.loads(rec.FinalResult())
    if 'text' in json_result:
        result += json_result['text']
        
    return result.strip()

def main():
    try:
        print("Speech to Text Converter (Persian)")
        print("---------------------------------")
        
        # Record audio (5 seconds)
        print("\nStep 1: Recording")
        audio_data = record_audio()
        
        # Save to temporary file
        print("\nStep 2: Processing")
        save_audio(audio_data)
        
        # Convert to text
        print("Converting speech to text...")
        text = transcribe_audio()
        
        # Display result
        print("\nStep 3: Results")
        print("Transcribed text:")
        print("-----------------")
        print(text)
        
        # Save to file
        print("\nStep 4: Saving")
        saved_file = save_text_to_file(text)
        if saved_file:
            print(f"Full text has been saved to: {saved_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
            print("\nTemporary audio file cleaned up")

if __name__ == "__main__":
    main()