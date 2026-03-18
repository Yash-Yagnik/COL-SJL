import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2

# Temporary setup per requirements: a single global dataset path.
DATASET_PATH = r"C:\Users\yashy\Downloads\UCF Crime Dataset"


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class SequenceItem:
    """
    One 'video' represented as a sequence of frame image paths.
    """

    sequence_id: str
    frame_paths: List[str]
    label: int  # 0=Normal, 1=Burglary


def _resolve_split_dir(split: str) -> str:
    """
    Dataset uses Train/Test capitalization; accept either.
    """
    split_norm = split.lower()
    for candidate in (split_norm, split_norm.title(), split_norm.upper()):
        p = os.path.join(DATASET_PATH, candidate)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Expected split directory missing under {DATASET_PATH}: {split} (e.g. Train/ or Test/)."
    )


def _extract_sequence_id(class_name: str, filename: str) -> str:
    """
    Extract a stable sequence id from filenames like:
      - Burglary092_x264_20.jpg  -> Burglary092
      - Normal_Videos_003_x264_0.jpg -> Normal_Videos_003
    """
    name = os.path.splitext(filename)[0]
    if class_name.lower().startswith("burglary"):
        # Take prefix 'Burglary' + digits.
        import re

        m = re.match(r"(Burglary\d+)", name, flags=re.IGNORECASE)
        return m.group(1) if m else f"Burglary_UNKNOWN"

    # Normal (folder is "NormalVideos")
    import re

    m = re.match(r"(Normal[_-]?Videos[_-]?\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Fallback: take up to the first two underscore-separated tokens.
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def _frame_sort_key(path: str) -> int:
    """
    Sort frames by the trailing index after the last underscore, e.g. *_x264_20 -> 20.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(base.split("_")[-1])
    except Exception:
        return 0


def list_split_sequences(split: str) -> List[SequenceItem]:
    """
    Enumerate frame sequences under:

        DATASET_PATH/
            Train/
                Burglary/
                Normal/
            Test/
                Burglary/
                Normal/
    """
    split_dir = _resolve_split_dir(split)

    sequences: List[SequenceItem] = []

    # Your dataset uses "NormalVideos" folder naming.
    for class_name, label in (("NormalVideos", 0), ("Burglary", 1)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected directory missing: {class_dir}. "
                f"Make sure folders exist: {split_dir}\\NormalVideos and {split_dir}\\Burglary."
            )

        groups: dict[str, List[str]] = {}
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            seq_id = _extract_sequence_id(class_name, fname)
            groups.setdefault(seq_id, []).append(os.path.join(class_dir, fname))

        for seq_id, paths in groups.items():
            paths.sort(key=_frame_sort_key)
            sequences.append(SequenceItem(sequence_id=seq_id, frame_paths=paths, label=label))

    return sequences


def iter_sampled_frames_from_images(
    frame_paths: List[str],
    *,
    max_frames: int = 20,
) -> Iterator:
    """
    Stream frames from a list of image paths without loading everything into memory.

    - Uniformly subsamples to at most `max_frames`
    """
    if not frame_paths:
        return

    # Choose a stride so we never exceed max_frames.
    stride = max(len(frame_paths) // max_frames, 1)

    kept = 0
    for i in range(0, len(frame_paths), stride):
        if kept >= max_frames:
            break
        frame = cv2.imread(frame_paths[i])
        if frame is None:
            continue
        yield frame
        kept += 1


def load_sequence_frames(
    frame_paths: List[str],
    *,
    max_frames: int = 20,
) -> List:
    """
    Convenience wrapper returning a bounded list of sampled frames from image paths.
    """
    return list(iter_sampled_frames_from_images(frame_paths, max_frames=max_frames))

