from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameData:
    """
    Container for a single sampled frame.

    Attributes
    ----------
    image:
        BGR image as a NumPy array (OpenCV format).
    frame_index:
        Index of this frame in the original video.
    timestamp:
        Timestamp in seconds from the start of the video.
    """

    image: np.ndarray
    frame_index: int
    timestamp: float


class VideoIngestion:
    """
    Loads a video file with OpenCV and yields frames sampled
    every ``sample_interval`` seconds.
    """

    def __init__(self, video_path: str, sample_interval: float = 0.5) -> None:
        """
        Parameters
        ----------
        video_path:
            Path to the input video file.
        sample_interval:
            Time interval in seconds between sampled frames.
        """
        self.video_path = video_path
        self.sample_interval = sample_interval

    def iter_frames(self) -> Iterator[FrameData]:
        """
        Lazily iterates over sampled frames from the video.

        Yields
        ------
        FrameData
            Each sampled frame along with its index and timestamp.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        # Frames per second; used to convert frame index to time and sampling stride.
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Compute how many frames correspond to the given sampling interval.
        # For example, at 30 FPS and 0.5s interval, we sample every 15 frames.
        sample_every_n_frames = max(int(fps * self.sample_interval), 1)

        frame_index = 0

        while True:
            # Read the next frame from the video.
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame based on the desired time interval.
            if frame_index % sample_every_n_frames == 0:
                timestamp = frame_index / fps
                yield FrameData(image=frame, frame_index=frame_index, timestamp=timestamp)

            frame_index += 1

        cap.release()

