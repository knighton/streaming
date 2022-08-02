import numpy as np


class Index(object):
    """An index of shard ranges."""

    def __init__(self, shard_sizes: list[int]) -> None:
        self.size = sum(shard_sizes)
        self.shard_sizes = shard_sizes
        self.shard_offsets = np.array([0] + shard_sizes).cumsum().tolist()

    def find(self, idx: int) -> tuple[int, int]:
        """Get the shard and offset where a sample will be found.

        Args:
            idx (int): Global sample index.

        Returns:
            tuple[int, int]: Shard and sample index within that shard.
        """
        low = 0
        high = len(self.shard_offsets) - 1
        while True:
            if low + 1 == high:
                if idx == self.shard_offsets[high]:
                    shard = high
                else:
                    shard = low
                break
            mid = (low + high) // 2
            div = self.shard_offsets[mid]
            if idx < div:
                high = mid
            elif div < idx:
                low = mid
            else:
                shard = mid
                break
        offset = idx - self.shard_offsets[shard]
        return shard, offset

