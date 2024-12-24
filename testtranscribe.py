import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Path to the audio file
audio_file = "001.wav"  # Replace with the actual file path

try:
    with sr.AudioFile(audio_file) as source:
        # Record the audio
        audio = recognizer.record(source)
    # Transcribe audio
    transcription = recognizer.recognize_google(audio, language="vi-VN")
    print(f"Transcription: {transcription}")
except Exception as e:
    print(f"Error: {e}")
