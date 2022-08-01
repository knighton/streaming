import json
import numpy as np
from typing import Any, Optional

from ..base.writer import SplitWriter
from .encodings import is_xsv_encoding, xsv_encode


class XSVWriter(SplitWriter):
    format = 'xsv'

    def __init__(
        self,
        dirname: str,
        columns: list[tuple[str, str]],
        separator: str,
        compression: Optional[str] = None,
        hashes: Optional[list[str]] = None,
        size_limit: Optional[int] = 1 << 26,
        newline: str = '\n'
    ) -> None:
        super().__init__(dirname, compression, hashes, size_limit)

        column_names, column_encodings = zip(*columns)
        assert len(set(column_names)) == len(column_names)
        for name in column_names:
            assert newline not in name
            assert separator not in name
        for encoding in column_encodings:
            assert is_xsv_encoding(encoding)

        self.columns = columns
        self.separator = separator
        self.newline = newline

    def _encode_sample(self, sample: dict[str, Any]) -> bytes:
        values = []
        for name, encoding in self.columns:
            value = xsv_encode(encoding, sample[name])
            assert self.newline not in value
            assert self.separator not in value
            values.append(value)
        text = self.separator.join(values) + self.newline
        return text.encode('utf-8')

    def _get_config(self) -> dict[str, Any]:
        obj = super()._get_config()
        obj.update({
            'columns': self.columns,
            'separator': self.separator,
            'newline': self.newline
        })
        return obj

    def _encode_split_shard(self) -> tuple[bytes, bytes]:
        column_names, _ = zip(*self.columns)
        header = self.separator.join(column_names) + self.newline
        header = header.encode('utf-8')
        data = b''.join([header] + self.new_samples)
        header_offset = len(header)

        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = header_offset + np.array([0] + sizes).cumsum().astype(np.uint32)
        obj = self._get_config()
        text = json.dumps(obj, sort_keys=True)
        meta = num_samples.tobytes() + offsets.tobytes() + text.encode('utf-8')

        return data, meta


class CSVWriter(XSVWriter):
    format = 'csv'
    separator = ','

    def __init__(
        self,
        dirname: str,
        columns: list[tuple[str, str]],
        compression: Optional[str] = None,
        hashes: Optional[list[str]] = None,
        size_limit: Optional[int] = 1 << 26,
        newline: str = '\n'
    ) -> None:
        super().__init__(dirname, columns, self.separator, compression, hashes, size_limit, newline)


class TSVWriter(XSVWriter):
    format = 'tsv'
    separator = '\t'

    def __init__(
        self,
        dirname: str,
        columns: list[tuple[str, str]],
        compression: Optional[str] = None,
        hashes: Optional[list[str]] = None,
        size_limit: Optional[int] = 1 << 26,
        newline: str = '\n'
    ) -> None:
        super().__init__(dirname, columns, self.separator, compression, hashes, size_limit, newline)
