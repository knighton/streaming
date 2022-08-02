from typing import Any, Iterator, Optional


class FileInfo(object):
    def __init__(self, basename: str, bytes: int, hashes: dict[str, str]) -> None:
        self.basename = basename
        self.bytes = bytes
        self.hashes = hashes


class Reader(object):
    """Provides random access to the samples of a shard.

    Args:
        dirname (str): Local dataset directory.
        compression (Optional[str], default: None): Optional compression or compression:level.
        hashes (Optional[list[str]], default: None): Optional list of hash algorithms to apply to
            shard files.
        samples (int): Number of samples in this shard.
        size_limit (Optional[int], default: 1 << 26): Optional shard size limit, after which point
            to start a new shard. If None, puts everything in one shard.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: list[str],
        samples: int,
        size_limit: Optional[int]
    ) -> None:
        self.dirname = dirname
        self.split = split or ''
        self.compression = compression
        self.hashes = hashes
        self.samples = samples
        self.size_limit = size_limit

    def __len__(self) -> int:
        """Get the number of samples in this shard.

        Returns:
            int: Sample count.
        """
        return self.samples

    def _decode_sample(self, data: bytes) -> dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            dict[str, Any]: Sample dict.
        """
        raise NotImplementedError

    def _get_sample_data(self, idx: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            idx (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            dict[str, Any]: Sample dict.
        """
        data = self._get_sample_data(idx)
        return self._decode_sample(data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the samples of this shard.

        Returns:
            Iterator[dict[str, Any]]: Iterator over samples.
        """
        for i in range(len(self)):
            yield self[i]


class JointReader(Reader):
    """Provides random access to the samples of a joint shard.

    Args:
        dirname (str): Local dataset directory.
        compression (Optional[str], default: None): Optional compression or compression:level.
        hashes (Optional[list[str]], default: None): Optional list of hash algorithms to apply to
            shard files.
        raw_data (FileInfo): Uncompressed data file info.
        samples (int): Number of samples in this shard.
        size_limit (Optional[int], default: 1 << 26): Optional shard size limit, after which point
            to start a new shard. If None, puts everything in one shard.
        zip_data (FileInfo): Compressed data file info.
    """
    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: list[str],
        raw_data: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: FileInfo
    ) -> None:
        super().__init__(dirname, split, compression, hashes, samples, size_limit)
        self.raw_data = raw_data
        self.zip_data = zip_data


class SplitReader(Reader):
    """Provides random access to the samples of a split shard.

    Args:
        dirname (str): Local dataset directory.
        compression (Optional[str], default: None): Optional compression or compression:level.
        hashes (Optional[list[str]], default: None): Optional list of hash algorithms to apply to
            shard files.
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (Optional[int], default: 1 << 26): Optional shard size limit, after which point
            to start a new shard. If None, puts everything in one shard.
        zip_data (FileInfo): Compressed data file info.
        zip_meta (FileInfo): Compressed meta file info.
    """
    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: list[str],
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: FileInfo,
        zip_meta: FileInfo
    ) -> None:
        super().__init__(dirname, split, compression, hashes, samples, size_limit)
        self.raw_data = raw_data
        self.raw_meta = raw_meta
        self.zip_data = zip_data
        self.zip_meta = zip_meta
