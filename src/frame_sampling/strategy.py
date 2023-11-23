"""Implements entire class hierarchy for all frame samplers."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import av
from av.video.frame import VideoFrame
from PIL.Image import Image
from tqdm.auto import tqdm

from .dataset import VideoDataset


@dataclass
class BaseSampler(ABC):
    """Most fundamental abstract base class for frame samplers."""

    sample_rate: int

    @abstractmethod
    def _sample_criteria(self, idx: int, frame: VideoFrame) -> bool:
        """Defines the criteria for frame sampling to occur."""
        pass

    @abstractmethod
    def _save_frame(self, sub_dir: Path, frame: VideoFrame) -> None:
        """Defines how video frames should be stored on disk."""
        pass

    @abstractmethod
    def _create_frame_sample_sub_dir(self, output_dir: Path, idx: int) -> Path:
        """Defines how the sample sub directory should be generated."""
        pass

    @property
    @abstractmethod
    def iter_frames_progress_delay(self) -> float:
        """Defines number of seconds to delay progress bar for frame iteration."""
        pass

    def sample(self, video_dataset: VideoDataset, output_dir: Path) -> None:
        """Loop through frames and store based on certain criteria."""
        # notify of sampling
        print(f"Sampling frames at every {self.sample_rate} frames ...")

        # iterate over video files in video data directory
        for video_idx, video_path in enumerate(tqdm(video_dataset)):
            # create frame samples sub directory
            sample_sub_dir = self._create_frame_sample_sub_dir(output_dir, video_idx)

            # open video file for streaming
            with av.open(str(video_path)) as container:
                # get video stream
                stream = container.streams.video[0]

                # get shortened name
                if len(video_path.name) > 30:
                    short_name = video_path.name[:30] + "(...)" + video_path.suffix
                else:
                    short_name = video_path.name

                # creating tqdm-wrapped video stream
                video_stream = tqdm(
                    container.decode(stream),
                    desc=f"Sampling {short_name}",
                    leave=False,
                    delay=self.iter_frames_progress_delay,
                )

                # iterate frames
                for frame_idx, frame in enumerate(video_stream):
                    # check for frame criteria
                    if self._sample_criteria(frame_idx, frame):
                        # write frame to disk
                        self._save_frame(sample_sub_dir, frame)


class MinimalSampler(BaseSampler):
    """Simplest frame sampling strategy."""

    iter_frames_progress_delay = 2

    def _create_frame_sample_sub_dir(self, output_dir: Path, idx: int) -> Path:
        """Use index of video file from dataset as name of sub directory."""
        # create new sub directory from video dataset index
        sub_dir = output_dir / str(idx)
        sub_dir.mkdir(parents=True)

        # return path to new sub dir
        return sub_dir

    def _sample_criteria(self, idx: int, frame: Image) -> bool:
        """Use sample rate and index to get modulus as a boolean."""
        return not idx % self.sample_rate

    def _save_frame(self, sub_dir: Path, frame: VideoFrame) -> None:
        """Simply write frame as a JPEG to sub directory."""
        # get PIL image
        frame_pil: Image = frame.to_image()

        # use frame timestamp as image file name
        image_file_name = str(frame.time) + ".jpg"

        # save to output dir
        frame_pil.save(sub_dir / image_file_name)
