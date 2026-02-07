import json
import math
from pathlib import Path
import pandas as pd


IN_CSV = "../report-data/reports_agg_5s_windows_transcripts.csv"
OUT_CSV = "../report-data/reports_agg_5s_windows_transcripts.csv"
TRANSCRIPTS_ROOT = Path("../transcripts/study2/whisper-medium")
GROUP_PREFIX = "group-"
TRANSCRIPT_JSON_NAME = "raw-undiarized.json"
NONE_LABEL_VALUE = "None"
WINDOW_SIZE = 5.0
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


def overlaps(a_start, a_end, b_start, b_end):
    return (a_end > b_start) and (a_start < b_end)


def window_overlaps_any(window, intervals):
    ws, we = window
    for s, e in intervals:
        if overlaps(ws, we, s, e):
            return True
    return False


def build_transcript_for_window(segments, start_t, end_t):
    texts = []
    for s_start, s_end, s_text in segments:
        if s_text and overlaps(s_start, s_end, start_t, end_t):
            texts.append(s_text)
    return TEXT_SEPARATOR.join(texts).strip()


def main():
    df = pd.read_csv(IN_CSV)
    df["startTime"] = pd.to_numeric(df["startTime"], errors="coerce")
    df["endTime"] = pd.to_numeric(df["endTime"], errors="coerce")
    df["videoEndTime"] = pd.to_numeric(df["videoEndTime"], errors="coerce")
    segments_cache = {}
    new_rows = []
    for group_id, gdf in df.groupby("groupID", dropna=False):
        gid = str(group_id).strip()
        video_end = float(gdf["videoEndTime"].dropna().iloc[0])
        occupied = []
        for _, r in gdf.iterrows():
            s = r["startTime"]
            e = r["endTime"]
            if pd.notna(s) and pd.notna(e):
                occupied.append((float(s), float(e)))
        if gid not in segments_cache:
            transcript_path = (TRANSCRIPTS_ROOT / f"{GROUP_PREFIX}{gid}" / TRANSCRIPT_JSON_NAME).resolve()
            if transcript_path.exists():
                segments_cache[gid] = load_segments(transcript_path)
            else:
                segments_cache[gid] = []
        segments = segments_cache[gid]
        n_steps = int(math.floor(video_end / WINDOW_SIZE))
        for k in range(n_steps + 1):
            start_t = k * WINDOW_SIZE
            end_t = min(video_end, start_t + WINDOW_SIZE)
            if end_t <= start_t:
                continue
            if window_overlaps_any((start_t, end_t), occupied):
                continue
            row = {col: "" for col in df.columns}
            row["groupID"] = group_id
            row["labels"] = NONE_LABEL_VALUE
            row["startTime"] = start_t
            row["endTime"] = end_t
            row["windowLength"] = end_t - start_t
            row["transcript"] = build_transcript_for_window(segments, start_t, end_t)
            row["videoEndTime"] = video_end
            new_rows.append(row)

    if new_rows:
        df_none = pd.DataFrame(new_rows, columns=df.columns)
        out_df = pd.concat([df, df_none], ignore_index=True)
    else:
        out_df = df.copy()

    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {Path(OUT_CSV).resolve()}")
    print(f"Added None windows: {len(new_rows)}")

if __name__ == "__main__":
    main()
