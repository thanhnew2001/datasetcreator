from __future__ import unicode_literals
import os
import speech_recognition as sr
import pandas as pd
import yt_dlp
import ffmpeg
import os
import speech_recognition as sr
import pandas as pd

import os
import subprocess
import json
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from datetime import datetime, timedelta
from pydub import AudioSegment

def download_from_url(url, output_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': output_name,  # Filename pattern for the output
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        print(f"Downloaded and saved as {output_name}.wav")


def transcribe_audio_files(wav_folder, output_csv):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # List to store transcription results
    transcriptions = []

    # Loop through all .wav files in the folder
    for file_name in os.listdir(wav_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(wav_folder, file_name)
            try:
                with sr.AudioFile(file_path) as source:
                    # Record the audio
                    audio = recognizer.record(source)
                # Transcribe audio in Vietnamese
                transcription = recognizer.recognize_google(audio, language="vi-VN")
                print(f"Transcribed {file_name}: {transcription}")
            except Exception as e:
                transcription = f"Error: {str(e)}"
                print(f"Failed to transcribe {file_name}: {e}")
            
            # Add results to the list
            transcriptions.append({"filename.wav": file_name, "transcription": transcription})

    # Create DataFrame and save it to CSV with '|' as delimiter
    df = pd.DataFrame(transcriptions)
    df.to_csv(output_csv, sep='|', index=False)

    print(f"Transcriptions saved to {output_csv}")


# Separate audio using Demucs
def separate_audio(input_audio):
    command = f"demucs --two-stems=vocals {input_audio}"
    
    print("Separating audio into vocals and instruments...")
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode())
    print("Audio separation completed.")

def GetTime(video_seconds):
    if video_seconds < 0:
        return "00:00:00.001"
    else:
        sec = timedelta(seconds=float(video_seconds))
        d = datetime(1, 1, 1) + sec
        return f"{str(d.hour).zfill(2)}:{str(d.minute).zfill(2)}:{str(d.second).zfill(2)}.001"

def windows(signal, window_size, step_size):
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1

def split_audio(input_file, output_dir, min_duration=3, max_duration=5, min_silence_length=0.6, silence_threshold=1e-4, step_duration=0.003):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Splitting audio file {input_file}...")
    sample_rate, samples = wavfile.read(input_file)
    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])
    window_size = int(min_silence_length * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(samples, window_size, step_size)
    window_energy = (energy(w) / max_energy for w in tqdm(signal_windows))
    window_silence = (e > silence_threshold for e in window_energy)
    cut_times = (r * step_duration for r in rising_edges(window_silence))

    cut_samples = [int(t * sample_rate) for t in cut_times]
    cut_samples.append(len(samples))  # Ensure the last segment ends at the file's end

    # Add forced cuts for segments exceeding max_duration
    final_cut_samples = []
    for i in range(len(cut_samples) - 1):
        start, stop = cut_samples[i], cut_samples[i + 1]
        segment_duration = (stop - start) / sample_rate

        if segment_duration > max_duration:
            # Force cuts at max_duration intervals
            forced_cuts = range(start, stop, int(max_duration * sample_rate))
            final_cut_samples.extend(forced_cuts)
        else:
            final_cut_samples.append(start)
    final_cut_samples.append(len(samples))

    cut_ranges = [(final_cut_samples[i], final_cut_samples[i + 1]) for i in range(len(final_cut_samples) - 1)]

    video_sub = {}
    segment_index = 0

    for start, stop in tqdm(cut_ranges):
        duration = (stop - start) / sample_rate  # Duration in seconds
        if min_duration <= duration <= max_duration:
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_{segment_index:03d}.wav")
            wavfile.write(output_file_path, rate=sample_rate, data=samples[start:stop])
            print(f"Written file: {output_file_path}")
            video_sub[str(segment_index)] = [GetTime(start / sample_rate), GetTime(stop / sample_rate)]
            segment_index += 1
        else:
            print(f"Skipped segment from {GetTime(start / sample_rate)} to {GetTime(stop / sample_rate)} (duration: {duration:.2f}s)")

    with open(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.json"), 'w') as output:
        json.dump(video_sub, output)

    print("Audio splitting completed.")


## MAIN
speaker_name = "ngocngan"
audio_input = "ngocngan.wav"

#2.separate and split:
separate_audio(audio_input)
output_dir = f"dataset_raw/{speaker_name}"
split_audio(f"separated/htdemucs/{speaker_name}/vocals.wav", output_dir)

# 3. transcribe
transcribe_audio_files(output_dir, "metadata.csv")

