import whisper
from pathlib import Path

AUDIO_PATH = Path("./data/Laplace_Formula.m4a")

model = whisper.load_model("tiny")
result = model.transcribe(AUDIO_PATH.as_posix())
print(result["text"])

for segment in result["segments"]:
    print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")