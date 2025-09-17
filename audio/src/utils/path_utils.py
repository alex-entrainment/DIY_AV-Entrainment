"""Utility helpers for validating file paths used by the audio toolchain."""

from __future__ import annotations

from typing import Iterable

_REMOTE_PREFIXES: Iterable[str] = (
    "http://",
    "https://",
    "ftp://",
    "ftps://",
)


def is_remote_path(path: str) -> bool:
    """Return ``True`` when ``path`` appears to point at a remote resource.

    The check is intentionally conservative: anything starting with a known
    network protocol (HTTP, HTTPS or FTP variants) or a protocol-relative
    prefix (``//``) is considered remote.  Windows UNC shares begin with ``\\``
    and are likewise treated as non-local so that audio playback only accesses
    facilitator-provided files.
    """

    if not path:
        return False

    value = path.strip()
    lowered = value.lower()
    if lowered.startswith(_REMOTE_PREFIXES):
        return True
    if lowered.startswith("//") or value.startswith("\\\\"):
        return True
    return False


__all__ = ["is_remote_path"]

