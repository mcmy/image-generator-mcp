from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BinaryInput:
    data: bytes
    filename: str
    mime_type: str
