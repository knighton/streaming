from typing import Any, Optional

from .base.reader import Reader
from .json import JSONReader, JSONWriter
from .mds import MDSReader, MDSWriter
from .xsv import CSVReader, CSVWriter, TSVReader, TSVWriter, XSVReader, XSVWriter


_readers = {
    'csv': CSVReader,
    'json': JSONReader,
    'mds': MDSReader,
    'tsv': TSVReader,
    'xsv': XSVReader
}


def reader_from_json(dirname: str, split: Optional[str], obj: dict[str, Any]) -> Reader:
    assert obj['version'] == 2
    cls = _readers[obj['format']]
    return cls.from_json(dirname, split, obj)


__all__ = ['CSVWriter', 'JSONReader', 'JSONWriter', 'MDSReader', 'MDSWriter', 'TSVWriter',
           'XSVWriter', 'reader_from_json']
