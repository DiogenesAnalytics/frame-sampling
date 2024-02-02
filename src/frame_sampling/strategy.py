"""Implements entire class hierarchy for all frame samplers."""
import subprocess
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterator
from typing import Tuple
from typing import Union

import av
from av.audio.frame import AudioFrame
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
            video_stream = self._create_stream(container, stream, video_name)

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

    def _create_stream(
        self,
        container: av.container,
        stream: Union[av.video.VideoStream, av.audio.AudioStream],
        video_name: str,
    ) -> Iterator[Tuple[int, Union[VideoFrame, AudioFrame]]]:
        """Create a tqdm-wrapped iterable video/audio stream."""
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

    def default_criteria(self, idx: int) -> bool:
        """Use sample rate and index to get modulus as a boolean."""
        return not idx % self.sample_rate

    def _handle_exceptions(self, error: BaseException, video_path: Path) -> None:
        """Pass on all exceptions and continue sampling frames."""
        print(f"Skipping error from {video_path.name}: {error}")

    def _create_subdir_path(self, output_dir: Path, idx: int) -> Path:
        """Use index of video file from dataset as name of sub directory."""
        # create new sub directory from video dataset index
        return output_dir / str(idx)

    def _sample_criteria(self, idx: int, frame: VideoFrame) -> bool:
        """Apply default criteria for checking frame interval."""
        return self.default_criteria(idx)

    def _save_frame(self, sub_dir: Path, frame: VideoFrame) -> None:
        """Simply write frame as a JPEG to sub directory."""
        # get PIL image
        frame_pil: Image = frame.to_image()

        # use frame timestamp as image file name
        image_file_name = str(frame.time) + ".jpg"

        # save to output dir
        frame_pil.save(sub_dir / image_file_name)


class CustomCriteriaSampler(MinimalSampler):
    """Applies custom criteria function to determine if frame should be sampled."""

    def __init__(
        self, sample_rate: int, criteria_func: Callable[[Image], bool]
    ) -> None:
        """Overrides ABC constructor to add a custom criteria function."""
        # first call abc constructor
        super().__init__(sample_rate=sample_rate)

        # now store criteria_func
        self.criteria_func = criteria_func

    def _sample_criteria(self, idx: int, frame: VideoFrame) -> bool:
        """Apply the custom criteria function."""
        return self.default_criteria(idx) and self.criteria_func(frame.to_image())


class VideoClipSampler(CustomCriteriaSampler):
    """Detects content in video frames and creates video clips."""

    min_clip_duration: int = 1

    def _sample_criteria(self, idx: int, frame: VideoFrame) -> bool:
        """Apply the custom criteria function."""
        # only apply custom criteria func
        return self.criteria_func(frame.to_image())

    def _save_video_clip(
        self, input_file: str, output_file: str, start_time: float, end_time: float
    ) -> None:
        """Save a subclip from the input video file using ffmpeg."""
        # build ffmpeg command
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-ss",
            f"{start_time:.2f}",
            "-to",
            f"{end_time:.2f}",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            output_file,
        ]

        # run command
        subprocess.run(command, check=True)

        # Add this print statement in your _save_video_clip method
        print(f"FFmpeg command: {' '.join(command)}")

    def _process_clip(
        self,
        container: av.container,
        start_time: float,
        end_time: float,
        sample_subdir: Path,
    ) -> None:
        """Get everything set up to create a new clip."""
        # calculate clip duration in seconds
        clip_duration = end_time - start_time

        # check if clip duration meets the minimum requirement
        if clip_duration >= self.min_clip_duration:
            # create subclip output path
            output_path = sample_subdir / f"clip_{start_time:.2f}_{end_time:.2f}.mp4"

            # use create_subclip_ffmpeg function to create subclip
            self._save_video_clip(
                container.name,
                str(output_path),
                start_time,
                end_time,
            )

    def _sample_single_video(self, video_path: Path, sample_subdir: Path) -> None:
        """Sample frames from a single video and create video clips."""
        with av.open(str(video_path)) as container:
            # cut vid name to max size
            video_name = self._get_shortened_name(video_path)

            # get raw streams
            video_stream = container.streams.video[0]

            # create tqdm-wrapped video/audio streams
            video_frames = self._create_stream(container, video_stream, video_name)

            # set the start point
            start_time = None

            # begin processing frames
            for video_frame_idx, video_frame in video_frames:
                # check if custom criteria is met
                if self._sample_criteria(video_frame_idx, video_frame):
                    # start recording the time
                    if start_time is None:
                        start_time = video_frame.time

                # criteria not met ...
                else:
                    # ... and start time was recorded ...
                    if start_time is not None:
                        # get end time
                        end_time = video_frame.time

                        # process the clip
                        self._process_clip(
                            container,
                            start_time,
                            end_time,
                            sample_subdir,
                        )

                        # reset start time for next clip
                        start_time = None

            # handle remaining frames
            if start_time is not None:
                # last frame's timestamp
                end_time = video_frame.time

                # now begin creating final clip
                self._process_clip(
                    container,
                    start_time,
                    end_time,
                    sample_subdir,
                )
