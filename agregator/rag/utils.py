from __future__ import annotations

from array import array
from typing import List, Sequence
import sys


def vector_to_bytes(values: Sequence[float]) -> bytes:
    """Сериализует последовательность float32 в bytes (little-endian)."""
    arr_obj = array("f", values)
    if sys.byteorder != "little":
        arr_obj.byteswap()
    return arr_obj.tobytes()


def bytes_to_vector(data: bytes) -> List[float]:
    """Десериализует bytes (float32 little-endian) в список чисел."""
    arr_obj = array("f")
    arr_obj.frombytes(data)
    if sys.byteorder != "little":
        arr_obj.byteswap()
    return list(arr_obj)
