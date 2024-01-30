"""Implements entire class hierarchy for all frame samplers."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from typing import Tuple

import av
from av.video.frame import VideoFrame
from PIL.Image import Image
from tqdm.auto import tqdm

from .dataset import VideoDataset


@dataclass
class BaseSampler(ABC):
    """Most fundamental abstract base class for frame samplers."""

    sample_rate: int

    def _sample_single_video(self, video_path: Path, sample_subdir: Path) -> None:
        """Sample frames from a single video."""
        # open video
        with av.open(str(video_path)) as container:
            # get first stream (video)
            stream = container.streams.video[0]

            # get shortened name
            video_name = self._get_shortened_name(video_path)

            # create iterable with progressbar and enumeration
            video_stream = self._create_video_stream(container, stream, video_name)

            # begin processing frames
            while True:
                try:
                    # get id and frame
                    frame_idx, frame = next(video_stream)

                    # decide how to process frame
                    self._process_frame(sample_subdir, frame_idx, frame)

                except StopIteration:
                    # end of frames
                    break

                except BaseException as error:
                    # decide how to handle exceptions
                    self._handle_exceptions(error, video_path)

    def _create_video_stream(
        self,
        container: av.container,
        stream: av.video.VideoStream,
        video_name: str,
    ) -> Iterator[Tuple[int, av.VideoFrame]]:
        """Create a tqdm-wrapped iterable video stream."""
        return iter(
            enumerate(
                tqdm(
                    container.decode(stream),
                    desc=f"Sampling {video_name}",
                    leave=False,
                    delay=self.iter_frames_progress_delay,
                )
            )
        )

    def _get_shortened_name(self, video_path: Path) -> str:
        """Get a shortened name for the video file."""
        if len(video_path.stem) > 30:
            chop = 30 - len(video_path.suffix)
            return video_path.stem[:chop] + "(...)" + video_path.suffix
        else:
            return video_path.name

    def _process_frame(
        self, sample_subdir: Path, frame_idx: int, frame: VideoFrame
    ) -> None:
        """Process and save the frame based on certain criteria."""
        if self._sample_criteria(frame_idx, frame):
            self._save_frame(sample_subdir, frame)

    @abstractmethod
    def _sample_criteria(self, idx: int, frame: VideoFrame) -> bool:
        """Defines the criteria for frame sampling to occur."""
        pass

    @abstractmethod
    def _save_frame(self, sub_dir: Path, frame: VideoFrame) -> None:
        """Defines how video frames should be stored on disk."""
        pass

    @abstractmethod
    def _create_subdir_path(self, output_dir: Path, idx: int) -> Path:
        """Defines how the sample sub directory path should be named."""
        pass

    @abstractmethod
    def _handle_exceptions(self, error: BaseException, video_path: Path) -> None:
        """Defines how to handle any exceptions while sampling frames."""
        pass

    @property
    @abstractmethod
    def iter_frames_progress_delay(self) -> float:
        """Defines number of seconds to delay progress bar for frame iteration."""
        pass

    def sample(
        self, video_dataset: VideoDataset, output_dir: Path, exist_ok: bool = False
    ) -> None:
        """Loop through frames and store based on certain criteria."""
        # notify of sampling
        print(f"Sampling frames at every {self.sample_rate} frames ...")

        # loop over videos and their id
        for video_idx, video_path in enumerate(tqdm(video_dataset)):
            # get name of sample subdir
            sample_subdir = self._create_subdir_path(output_dir, video_idx)

            # check if dir exists
            if exist_ok or not sample_subdir.exists():
                # create dir
                sample_subdir.mkdir(parents=True, exist_ok=exist_ok)

                # get frame samples from single video
                self._sample_single_video(video_path, sample_subdir)


class MinimalSampler(BaseSampler):
    """Simplest frame sampling strategy."""

    iter_frames_progress_delay = 2

    def _handle_exceptions(self, error: BaseException, video_path: Path) -> None:
        """Pass on all exceptions and continue sampling frames."""
        print(f"Skipping error from {video_path.name}: {error}")

    def _create_subdir_path(self, output_dir: Path, idx: int) -> Path:
        """Use index of video file from dataset as name of sub directory."""
        # create new sub directory from video dataset index
        return output_dir / str(idx)

    def _sample_criteria(self, idx: int, frame: VideoFrame) -> bool:
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
