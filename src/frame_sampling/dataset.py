"""Implements classes for dealing with video data."""
from abc import ABC
from abc import abstractmethod
from dataclasses import InitVar
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
from typing import Iterator
from typing import List
from typing import Union


@dataclass
class Dataset(ABC):
    """Defined the abstract base class for all datasets."""

    data_dir: InitVar[Union[str, Path]]

    def __post_init__(self, data_dir: Union[str, Path]) -> None:
        """Apply post constructor processing to args."""
        # get data path object
        self._data_path: Path = Path(data_dir)

        # make sure path exists
        self._dataset_exists()

        # create index
        self.index: List[Path] = [path for path in self._get_filepaths()]

    def __iter__(self) -> Iterator[Path]:
        """Defining the iteration behavior."""
        return iter(self.index)

    def __len__(self) -> int:
        """Defining how to calculate length of dataset."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Path:
        """Defining how data path objects will be accessed."""
        return self.index[idx]

    def _dataset_exists(self) -> None:
        """Make sure path to data dir exists."""
        assert self._data_path.exists()

    def _get_filepaths(self) -> Generator[Path, None, None]:
        """Scan file system for video files and grab their file paths."""
        # iterate over video file extensions
        for ext in self.file_extensions:
            # loop through video paths matching ext
            yield from self._data_path.glob(f"**/{ext}")

    @property
    @abstractmethod
    def type(self) -> str:
        """Defines the type of the data found in dataset."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Defines the file extensions accepted for the given data type."""
        pass

    @property
    def path(self) -> str:
        """Retuns the data path as a string."""
        return str(self._data_path)


class VideoDataset(Dataset):
    """Dataset of video files."""

    type = "video"
    file_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]
