import json
import whisper
import torch
from pathlib import Path

VIDEO_ROOT = "/s/babbage/h/nobackup/nblancha/public-datasets/sifat/modeling_internal_states/study2_raw_video_data"
OUT_ROOT = Path("../transcripts/study2/whisper-medium") 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "medium"
LANGUAGE = "en"
WORD_TIMESTAMPS = True
FP16 = torch.cuda.is_available()
OVERWRITE = True

def transcribe_video(model, video_path: Path, language: str, fp16: bool, word_timestamps: bool):
    result = model.transcribe(
        str(video_path),
        language=language,
        fp16=fp16,
        verbose=False,
        word_timestamps=word_timestamps,
    )
    return result

model = whisper.load_model(MODEL, device=DEVICE)
print(f"using device: {DEVICE}")

group_dirs = ["group-2", "group-3", "group-4", "group-5", "group-6", "group-7", "group-8", "group-9", "group-10"]
runs = []

for group_id in group_dirs:
    video_path = Path(VIDEO_ROOT) / group_id / "input.mp4"
    out_dir = OUT_ROOT / group_id
    out_path = out_dir / "raw-undiarized.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not OVERWRITE:
        print(f"Skipping {group_id}, already exists.")
        continue

    try:
        print(f"Transcribing {group_id}...")
        result = transcribe_video(
            model=model,
            video_path=video_path,
            language=LANGUAGE,
            fp16=FP16,
            word_timestamps=WORD_TIMESTAMPS,
        )
        payload = {
            "group": group_id,
            "video_path": str(video_path),
            "whisper_model": MODEL,
            "language": LANGUAGE,
            "word_timestamps": bool(WORD_TIMESTAMPS),
            "result": result,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        runs.append({"group": group_id, "status": "ok", "out": str(out_path)})
    except Exception as e:
        print(f"Error on {group_id}: {e}")
        runs.append({"group": group_id, "status": "error", "error": repr(e), "video": str(video_path)})

summary_path = OUT_ROOT / "run-summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Wrote summary: {summary_path}")
ok = sum(1 for r in runs if r["status"] == "ok")
err = sum(1 for r in runs if r["status"] == "error")
print(f"ok={ok} error={err}")