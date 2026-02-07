import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import opensmile


IN_CSV = "../report-data/reports_agg_5s_windows_transcripts.csv"
OUT_CSV = "../report-data/reports_agg_5s_windows_transcripts.csv"
VIDEO_ROOT = Path("../study2_raw_video_data")
GROUP_PREFIX = "group-"
VIDEO_NAME = "input.mp4"
SAMPLE_RATE = 16000
SMILE_FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
SMILE_FEATURE_LEVEL = opensmile.FeatureLevel.Functionals


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg first.")


def extract_wav_window(video_path: Path, start_s: float, end_s: float, out_wav: Path):
    start_s = float(start_s)
    end_s = float(end_s)
    dur = max(0.0, end_s - start_s)
    if dur <= 0:
        out_wav.write_bytes(b"")
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{dur:.3f}",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    run(cmd)
    return out_wav.exists() and out_wav.stat().st_size > 0


def main():
    ensure_ffmpeg()
    df = pd.read_csv(IN_CSV)
    df["startTime"] = pd.to_numeric(df["startTime"], errors="coerce")
    df["endTime"] = pd.to_numeric(df["endTime"], errors="coerce")
    smile = opensmile.Smile(feature_set=SMILE_FEATURE_SET, feature_level=SMILE_FEATURE_LEVEL)
    feature_names = None
    features_rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for _, row in df.iterrows():
            gid = str(row["groupID"]).strip()
            video_path = (VIDEO_ROOT / f"{GROUP_PREFIX}{gid}" / VIDEO_NAME).resolve()
            start_t = row["startTime"]
            end_t = row["endTime"]
            if not video_path.exists() or pd.isna(start_t) or pd.isna(end_t):
                features_rows.append(None)
                continue
            wav_path = tmpdir / f"{gid}_{int(start_t*1000)}_{int(end_t*1000)}.wav"
            ok = extract_wav_window(video_path, float(start_t), float(end_t), wav_path)
            if not ok:
                features_rows.append(None)
                continue
            feats = smile.process_file(str(wav_path))
            feats = feats.reset_index(drop=True)
            if feature_names is None:
                feature_names = list(feats.columns)
            features_rows.append(feats.iloc[0].to_dict())

    if feature_names is None:
        out_df = df.copy()
        out_df.to_csv(OUT_CSV, index=False)
        print(Path(OUT_CSV).resolve())
        return

    feat_df = pd.DataFrame(
        [{k: (v if d is not None else np.nan) for k, v in (d or {}).items()} for d in features_rows],
        columns=feature_names,
    )

    out_df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out_df.to_csv(OUT_CSV, index=False)
    print(Path(OUT_CSV).resolve())


if __name__ == "__main__":
    main()
