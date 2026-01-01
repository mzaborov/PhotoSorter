from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from yadisk_client import get_disk


def _get(item: Any, key: str) -> Optional[Any]:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def norm(p: str) -> str:
    p = (p or "").strip()
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


@dataclass
class DebugResult:
    root: str
    files: int
    dirs: int
    seconds: float
    error_path: Optional[str]
    error: Optional[str]


def listdir_with_retry(disk, path: str, retries: int = 2, delay: float = 0.5):
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return list(disk.listdir(path))
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < retries:
                time.sleep(delay)
                continue
            raise last_exc


def debug_count(root: str) -> DebugResult:
    disk = get_disk()
    root_norm = norm(root)

    files = 0
    dirs = 0
    stack = [root_norm]
    t0 = time.perf_counter()
    error_path: Optional[str] = None
    error: Optional[str] = None

    try:
        while stack:
            current = stack.pop()
            try:
                items = listdir_with_retry(disk, current, retries=2, delay=0.5)
            except Exception as e:  # noqa: BLE001
                error_path = current
                error = f"{type(e).__name__}: {e}"
                break

            for it in items:
                t = _get(it, "type")
                if t == "dir":
                    dirs += 1
                    p = _get(it, "path")
                    if p:
                        stack.append(norm(str(p)))
                elif t == "file":
                    files += 1
    finally:
        dt = time.perf_counter() - t0

    return DebugResult(
        root=root,
        files=files,
        dirs=dirs,
        seconds=dt,
        error_path=error_path,
        error=error,
    )


if __name__ == "__main__":
    root = "disk:/Фото/Путешествия"
    r = debug_count(root)
    print(f"root={r.root}")
    print(f"files={r.files} dirs={r.dirs} seconds={r.seconds:.2f}")
    if r.error:
        print(f"ERROR at {r.error_path}: {r.error}")


