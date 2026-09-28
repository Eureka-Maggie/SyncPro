"""Pull frames from a video at integer seconds with decord.

We deliberately stay at 1 fps (the training distribution). The runner
calls ``read_frame(t_sec)`` for each tick; decord caches the open handle.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class VideoFrameReader:

    def __init__(self, video_path: str):
        import decord
        decord.bridge.set_bridge('native')
        self._vr = decord.VideoReader(video_path)
        self.fps = float(self._vr.get_avg_fps())
        self.n_frames = len(self._vr)
        self.duration_s = self.n_frames / self.fps if self.fps > 0 else 0.0
        self.path = video_path

    def read_frame(self, t_sec: float) -> Image.Image:
        """Return the frame closest to ``t_sec`` as a PIL.Image (RGB)."""
        idx = int(round(t_sec * self.fps))
        if idx < 0:
            idx = 0
        if idx >= self.n_frames:
            idx = self.n_frames - 1
        arr = self._vr[idx].asnumpy()  # H W 3 uint8 RGB
        return Image.fromarray(arr)

    def close(self):
        del self._vr
