from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import urllib.error
import zlib
from pathlib import Path


# PlantUML server uses a custom base64 alphabet.
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

# Список серверов PlantUML по умолчанию (пробуем по очереди при ошибках)
_DEFAULT_SERVERS = [
    "https://www.plantuml.com/plantuml",
    "http://www.plantuml.com/plantuml",
]


def _encode_6bit(b: int) -> str:
    if b < 0:
        b = 0
    if b > 63:
        b = 63
    return _ALPHABET[b]


def _append_3bytes(b1: int, b2: int, b3: int) -> str:
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return _encode_6bit(c1) + _encode_6bit(c2) + _encode_6bit(c3) + _encode_6bit(c4)


def plantuml_encode(text: str) -> str:
    data = text.encode("utf-8")
    # zlib -> raw deflate stream expected by PlantUML server
    z = zlib.compress(data, 9)
    raw = z[2:-4]
    res = []
    i = 0
    while i < len(raw):
        b1 = raw[i]
        b2 = raw[i + 1] if i + 1 < len(raw) else 0
        b3 = raw[i + 2] if i + 2 < len(raw) else 0
        res.append(_append_3bytes(b1, b2, b3))
        i += 3
    return "".join(res)


def render_one(puml_path: Path, out_png: Path, servers: list[str]) -> None:
    """
    Рендерит PlantUML файл в PNG, пробуя серверы из списка по очереди.
    Для каждого сервера делается до 5 попыток с backoff.
    """
    text = puml_path.read_text(encoding="utf-8")
    encoded = plantuml_encode(text)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_suffix(out_png.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    all_errors: list[tuple[str, Exception]] = []
    
    for server_idx, server_base in enumerate(servers, 1):
        url = server_base.rstrip("/") + "/png/" + encoded
        server_name = server_base.split("//")[-1].split("/")[0] if "//" in server_base else server_base
        
        for attempt in range(1, 6):
            try:
                # Важно: короткий таймаут, чтобы не "висеть" на сети/сервере.
                with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
                    data = resp.read()
                tmp.write_bytes(data)
                tmp.replace(out_png)
                if server_idx > 1 or attempt > 1:
                    print(f"  success on server {server_idx}/{len(servers)} ({server_name})", file=sys.stderr)
                return
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                all_errors.append((server_name, e))
                # простой backoff
                time_sleep = min(2.0 * attempt, 6.0)
                print(f"warn: server {server_idx}/{len(servers)} ({server_name}) failed (attempt {attempt}/5): {type(e).__name__}: {e}. sleep {time_sleep:.1f}s", file=sys.stderr)
                time.sleep(time_sleep)
                continue
        
        # Если все попытки для этого сервера не удались, пробуем следующий
        if server_idx < len(servers):
            print(f"  switching to next server...", file=sys.stderr)

    # Все серверы не сработали
    error_summary = "; ".join(f"{name}: {type(e).__name__}" for name, e in all_errors[-3:])  # последние 3 ошибки
    raise RuntimeError(f"Failed to render {puml_path.name} after trying {len(servers)} server(s): {error_summary}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render PlantUML .puml files to PNG via PlantUML server.")
    ap.add_argument("--in-dir", default="docs/diagrams", help="Directory with .puml files")
    ap.add_argument("--server", help="PlantUML server base URL (если указан, используется только он; иначе пробуются серверы по умолчанию)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"ERROR: dir not found: {in_dir}", file=sys.stderr)
        return 2

    pumls = sorted(in_dir.glob("*.puml"))
    if not pumls:
        print(f"ERROR: no .puml files in {in_dir}", file=sys.stderr)
        return 2

    # Определяем список серверов
    if args.server:
        servers = [args.server]
    else:
        servers = _DEFAULT_SERVERS
        print(f"Using {len(servers)} default server(s): {', '.join(s.split('//')[-1].split('/')[0] if '//' in s else s for s in servers)}", file=sys.stderr)

    for p in pumls:
        out = p.with_suffix(".png")
        print(f"render: {p} -> {out}")
        try:
            render_one(p, out, servers)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {p.name}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


