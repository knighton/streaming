from copy import deepcopy
import json
import numpy as np
import os
from typing import Any, Optional
from typing_extensions import Self

from ..base.reader import FileInfo, SplitReader
from .encodings import xsv_decode

'''
    {
      "column_encodings": [
        "int",
        "str"
      ],
      "column_names": [
        "number",
        "words"
      ],
      "compression": "zstd:7",
      "format": "csv",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "newline": "\n",
      "raw_data": {
        "basename": "shard.00000.csv",
        "bytes": 1048523,
        "hashes": {
          "sha1": "39f6ea99d882d3652e34fe5bd4682454664efeda",
          "xxh3_64": "ea1572efa0207ff6"
        }
      },
      "raw_meta": {
        "basename": "shard.00000.csv.meta",
        "bytes": 77486,
        "hashes": {
          "sha1": "8874e88494214b45f807098dab9e55d59b6c4aec",
          "xxh3_64": "3b1837601382af2c"
        }
      },
      "samples": 19315,
      "separator": ",",
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.csv.zstd",
        "bytes": 197040,
        "hashes": {
          "sha1": "021d288a317ae0ecacba8a1b985ee107f966710d",
          "xxh3_64": "5daa4fd69d3578e4"
        }
      },
      "zip_meta": {
        "basename": "shard.00000.csv.meta.zstd",
        "bytes": 60981,
        "hashes": {
          "sha1": "f2a35f65279fbc45e8996fa599b25290608990b2",
          "xxh3_64": "7c38dee2b3980deb"
        }
      }
    }
'''


class XSVReader(SplitReader):
    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: list[str],
        column_names: list[str],
        compression: Optional[str],
        hashes: list[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        separator: str,
        size_limit: Optional[int],
        zip_data: FileInfo,
        zip_meta: FileInfo
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, raw_meta, samples,
                         size_limit, zip_data, zip_meta)
        self.column_encodings = column_encodings
        self.column_names = column_names
        self.newline = newline
        self.separator = separator

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['version'] == 2
        del args['version']
        assert args['format'] == 'xsv'
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            args[key] = FileInfo(**args[key])
        return cls(**args)

    def _decode_sample(self, data: bytes) -> dict[str, Any]:
        text = data.decode('utf-8')
        text = text[:-len(self.newline)]
        parts = text.split(self.separator)
        sample = {}
        for name, encoding, part in zip(self.column_names, self.column_encodings, parts):
            sample[name] = xsv_decode(encoding, part)
        return sample

    def _get_sample_data(self, idx: int) -> bytes:
        meta_filename = os.path.join(self.dirname, self.split, self.raw_meta.basename)
        offset = (1 + idx) * 4
        with open(meta_filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
        data_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        with open(data_filename, 'rb', 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data


class CSVReader(XSVReader):
    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['format'] == 'csv'
        args['format'] = 'xsv'
        args['separator'] = ','
        return super().from_json(dirname, split, args)


class TSVReader(XSVReader):
    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        args = deepcopy(obj)
        assert args['format'] == 'tsv'
        args['format'] = 'xsv'
        args['separator'] = '\t'
        return super().from_json(dirname, split, args)
