"""Extract CLAP audio embeddings for a folder of MP3 files.

For each track, N windows of 10 s are taken evenly across the middle of the song
and embedded with CLAP. One .npy file (n_windows x 512) is written per track, so
the script can be interrupted and resumed. A manifest.csv (file, hash, title,
artist, genre) is written at the end.

Usage:
    python extract_embeddings.py --limit 50          # quick test
    python extract_embeddings.py                     # full library
"""
import argparse
import csv
import hashlib
import os
import time

import librosa
import numpy as np
import torch
from mutagen.easyid3 import EasyID3
from transformers import ClapFeatureExtractor, ClapModel

MODEL_NAME = "laion/clap-htsat-unfused"
SR = 48000
WINDOW_S = 10
N_WINDOWS = 3


def file_key(filename):
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


def load_windows(filepath):
    """Return an array (N_WINDOWS, SR * WINDOW_S) of windows spread over 25%-75% of the track."""
    audio, _ = librosa.load(filepath, sr=SR, mono=True)
    win = SR * WINDOW_S
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = np.linspace(0.25, 0.75, N_WINDOWS) * (len(audio) - win)
    return np.stack([audio[int(s):int(s) + win] for s in starts])


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, folder, filenames):
        self.folder = folder
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        name = self.filenames[i]
        try:
            return name, load_windows(os.path.join(self.folder, name)), None
        except Exception as e:
            return name, None, str(e)


def write_manifest(folder, filenames, out_dir):
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "key", "title", "artist", "genre", "has_embedding"])
        for name in filenames:
            key = file_key(name)
            try:
                tags = EasyID3(os.path.join(folder, name))
            except Exception:
                tags = {}
            get = lambda k: (tags.get(k) or [""])[0] if tags else ""
            done = os.path.exists(os.path.join(out_dir, f"{key}.npy"))
            writer.writerow([name, key, get("title"), get("artist"), get("genre"), int(done)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=r"C:\Users\robin\Music\Musique")
    parser.add_argument("--out", default="embeddings/clap")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    filenames = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(".mp3"))
    if args.limit:
        filenames = filenames[:args.limit]
    todo = [f for f in filenames if not os.path.exists(os.path.join(args.out, f"{file_key(f)}.npy"))]
    print(f"{len(filenames)} files, {len(filenames) - len(todo)} already done, {len(todo)} to process")

    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_NAME} on {device}")
        model = ClapModel.from_pretrained(MODEL_NAME).to(device).eval()
        extractor = ClapFeatureExtractor.from_pretrained(MODEL_NAME)

        loader = torch.utils.data.DataLoader(
            TrackDataset(args.folder, todo), batch_size=None, shuffle=False,
            num_workers=args.workers, prefetch_factor=4 if args.workers else None,
        )
        errors = []
        start = time.time()
        for n, (name, windows, err) in enumerate(loader, 1):
            if err is not None:
                errors.append((name, err))
                print(f"ERROR {name}: {err}")
                continue
            inputs = extractor(list(np.asarray(windows)), sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                emb = model.get_audio_features(input_features=inputs["input_features"].to(device))
            if not torch.is_tensor(emb):
                emb = emb.pooler_output
            np.save(os.path.join(args.out, f"{file_key(name)}.npy"), emb.cpu().numpy().astype(np.float32))
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - start)
                print(f"{n}/{len(todo)}  {rate:.2f} tracks/s  ETA {(len(todo) - n) / rate / 60:.1f} min")

        if errors:
            with open(os.path.join(args.out, "errors.txt"), "w", encoding="utf-8") as f:
                f.writelines(f"{n}\t{e}\n" for n, e in errors)

    write_manifest(args.folder, filenames, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
