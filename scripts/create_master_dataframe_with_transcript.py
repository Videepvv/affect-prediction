import json
from pathlib import Path
import pandas as pd

REPORTS_CSV = "../report-data/reports-agg.csv"

TRANSCRIPTS_ROOT = Path("../transcripts/study2/whisper-medium")
GROUP_PREFIX = "group-"
TRANSCRIPT_JSON_NAME = "raw-undiarized.json"

OUT_CSV = "../report-data/reports_agg_5s_windows_transcripts.csv"

HALF_WINDOW = 2.5
TEXT_SEPARATOR = " "


def load_segments(transcript_json_path: Path):
    data = json.loads(transcript_json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    segments = data.get("segments", [])
    out = []
    for s in segments:
        try:
            start = float(s["start"])
            end = float(s["end"])
            text = (s.get("text") or "").strip()
            out.append((start, end, text))
        except Exception:
            continue
    return out


def compute_window(video_time, video_end_time):
    vt = float(video_time)
    ve = float(video_end_time)
    start = max(0.0, vt - HALF_WINDOW)
    end = min(ve, vt + HALF_WINDOW)
    if end < start:
        end = start
    return start, end, end - start

def overlaps(a_start, a_end, b_start, b_end):
    return (a_end > b_start) and (a_start < b_end)

def build_transcript_for_window(segments, start_t, end_t):
    texts = []
    for s_start, s_end, s_text in segments:
        if s_text and overlaps(s_start, s_end, start_t, end_t):
            texts.append(s_text)
    return TEXT_SEPARATOR.join(texts).strip()

def main():
    reports_csv = Path(REPORTS_CSV).resolve()
    transcripts_root = TRANSCRIPTS_ROOT.resolve()
    out_csv = Path(OUT_CSV).resolve()
    df = pd.read_csv(reports_csv)
    required = {"videoTime", "videoEndTime", "groupID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    group_cache = {}
    transcripts = []
    start_times = []
    end_times = []
    window_lengths = []
    for idx, row in df.iterrows():
        group_id = str(row["groupID"]).strip()
        group_dir = transcripts_root / f"{GROUP_PREFIX}{group_id}"
        transcript_path = group_dir / TRANSCRIPT_JSON_NAME
        if group_id not in group_cache:
            if transcript_path.exists():
                group_cache[group_id] = load_segments(transcript_path)
            else:
                print(f"WARNING: transcript not found for group {group_id}: {transcript_path}")
                group_cache[group_id] = []
        segments = group_cache[group_id]
        start_t, end_t, wlen = compute_window(
            row["videoTime"],
            row["videoEndTime"]
        )
        start_times.append(start_t)
        end_times.append(end_t)
        window_lengths.append(wlen)
        transcript_text = build_transcript_for_window(segments, start_t, end_t)
        transcripts.append(transcript_text)

    df["startTime"] = start_times
    df["endTime"] = end_times
    df["windowLength"] = window_lengths
    df["transcript"] = transcripts

    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
