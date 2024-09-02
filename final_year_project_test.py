

from google.colab import drive
drive.mount('/content/drive')

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import vosk
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from google.colab import drive
from sklearn.metrics import precision_recall_fscore_support
from jiwer import wer, cer
import random

# Mount Google Drive
drive.mount('/content/drive')

# Update model path to your VOSK model location in Google Drive
model_path = "/content/drive/MyDrive/vosk-model-small-en-in-0.4"  # Update this path

if not os.path.exists(model_path):
    raise FileNotFoundError("Please make sure the VOSK model is located at the specified path in your Google Drive.")

vosk_model = vosk.Model(model_path)

def transcribe_audio(audio_data, sample_rate):
    recognizer = vosk.KaldiRecognizer(vosk_model, sample_rate)
    if recognizer.AcceptWaveform(audio_data):
        result = recognizer.Result()
        result_dict = json.loads(result)
        transcription = result_dict.get("text", "")
        return transcription
    result = recognizer.FinalResult()
    if result:
        result_dict = json.loads(result)
        transcription = result_dict.get("text", "")
        return transcription
    return ""

def preprocess_text(text):
    # Preprocess text for comparison
    text = text.lower().strip()
    text = ' '.join(text.split())  # Normalize whitespace
    return text

def map_wer_to_accuracy(wer_score):
    # Define a function to map WER to accuracy
    # Example mapping (you can adjust this mapping based on your needs):
    # Assuming a linear relation where WER 0.36 corresponds to Accuracy 0.74
    max_wer = 1.0  # Maximum possible WER
    min_accuracy = 0.0
    max_accuracy = 1.0
    accuracy = max_accuracy - (wer_score / max_wer) * (max_accuracy - min_accuracy)
    return accuracy

def evaluate_vosk_model(dataset, num_samples):
    start_time = time.time()

    all_transcriptions = []
    all_references = []

    # Randomly select samples
    indices = random.sample(range(len(dataset)), num_samples)

    for i in indices:
        waveform, sample_rate, transcript, *other = dataset[i]

        # Print waveform and sample_rate information
        print(f"Processing sample {i + 1}")
        print(f"Sample Rate: {sample_rate}")
        print(f"Waveform Shape: {waveform.shape}")

        # Ensure waveform is 1D and convert to bytes
        waveform = waveform.squeeze()  # Convert to 1D if needed
        waveform_np = waveform.numpy()
        waveform_np = np.int16(waveform_np * 32767)
        audio_data = waveform_np.tobytes()  # Convert to 16-bit PCM bytes

        # Get transcription from the VOSK model
        transcription = transcribe_audio(audio_data, sample_rate)

        # Store transcriptions and references for accuracy calculation
        all_transcriptions.append(preprocess_text(transcription))
        all_references.append(preprocess_text(transcript))

    elapsed_time = time.time() - start_time

    # Calculate WER and CER
    wer_score = wer(all_references, all_transcriptions)
    cer_score = cer(all_references, all_transcriptions)

    # Map WER to Accuracy
    accuracy = map_wer_to_accuracy(wer_score)

    precision, recall, f1, _ = precision_recall_fscore_support(all_references, all_transcriptions, average='weighted')

    return {
        "sample_size": num_samples,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "wer": wer_score,
        "cer": cer_score,
        "time_taken": elapsed_time
    }

# Load the LibriSpeech dataset
dataset = LIBRISPEECH(root="./", url="test-clean", download=True)

# Define sample sizes to test
sample_sizes = [10, 20, 30, 40,50,60,70,80,90,100]
results = []

for size in sample_sizes:
    print(f"\nEvaluating sample size: {size}")
    result = evaluate_vosk_model(dataset, num_samples=size)
    results.append(result)

# Extract metrics for plotting
sample_sizes = [result["sample_size"] for result in results]
accuracies = [result["accuracy"] for result in results]
wers = [result["wer"] for result in results]
cers = [result["cer"] for result in results]

# Plot the metrics on a single graph
plt.figure(figsize=(12, 6))

# Line chart for accuracy
plt.plot(sample_sizes, accuracies, marker='o', linestyle='-', color='blue', label='Accuracy')

# Line chart for WER
plt.plot(sample_sizes, wers, marker='o', linestyle='--', color='red', label='WER')

# Line chart for CER
plt.plot(sample_sizes, cers, marker='o', linestyle='-.', color='green', label='CER')

plt.xlabel('Sample Size')
plt.ylabel('Score')
plt.title('VOSK Model Evaluation Metrics vs. Sample Size')
plt.legend()

plt.grid(True)
plt.show()

