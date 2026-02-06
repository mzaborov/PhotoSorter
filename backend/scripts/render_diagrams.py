from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

# Путь к plantuml.jar: PLANTUML_JAR в secrets.env/.env, иначе рядом со скриптом
def _default_jar_path() -> Path:
    try:
        from dotenv import load_dotenv
        repo_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=repo_root / "secrets.env", override=False)
        load_dotenv(dotenv_path=repo_root / ".env", override=False)
    except ImportError:
        pass
    env_path = os.environ.get("PLANTUML_JAR", "").strip().strip('"\'')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return Path(__file__).resolve().parent / "plantuml.jar"


# PlantUML воспринимает круглые скобки в описаниях атрибутов как методы и переносит в секцию методов.
# Заменяем ( и ) на [ и ] перед рендером, чтобы описания корректно оставались в атрибутах.
_REPLACE_PARENS = str.maketrans("()", "[]")


def render_one_jar(jar_path: Path, puml_path: Path) -> None:
    """
    Рендерит PlantUML файл в PNG через локальный java -jar plantuml.jar.
    PNG создаётся рядом с .puml (то же имя, расширение .png).
    Перед рендером заменяются круглые скобки на квадратные (PlantUML иначе воспринимает как методы).
    """
    puml_abs = puml_path.resolve()
    if not puml_abs.exists():
        raise FileNotFoundError(puml_abs)
    out_png = puml_abs.with_suffix(".png")
    content = puml_abs.read_text(encoding="utf-8").translate(_REPLACE_PARENS)
    result = subprocess.run(
        ["java", "-jar", str(jar_path), "-tpng", "-pipe"],
        input=content.encode("utf-8"),
        capture_output=True,
        timeout=60,
        cwd=str(puml_abs.parent),
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"PlantUML failed (exit {result.returncode}): {err}")
    if not result.stdout:
        raise RuntimeError(f"PlantUML did not produce output for {puml_abs.name}")
    out_png.write_bytes(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render PlantUML .puml files to PNG via local plantuml.jar (java -jar)."
    )
    ap.add_argument("--in-dir", default="docs/diagrams", help="Directory with .puml files")
    ap.add_argument("--jar", help="Path to plantuml.jar (default: PLANTUML_JAR from secrets.env or script dir)")
    args = ap.parse_args()

    jar_path = Path(args.jar).resolve() if args.jar else _default_jar_path()
    if not jar_path.exists():
        print(f"ERROR: plantuml.jar not found: {jar_path}", file=sys.stderr)
        return 2

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"ERROR: dir not found: {in_dir}", file=sys.stderr)
        return 2

    pumls = sorted(in_dir.glob("*.puml"))
    if not pumls:
        print(f"ERROR: no .puml files in {in_dir}", file=sys.stderr)
        return 2

    print(f"Using local plantuml.jar: {jar_path}", file=sys.stderr)
    for p in pumls:
        out = p.with_suffix(".png")
        print(f"render: {p} -> {out}")
        try:
            render_one_jar(jar_path, p)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {p.name}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    # Встраивание README в screens.html для работы без сервера (file://)
    readme_path = in_dir.parent / "README.md"
    screens_html = in_dir / "screens.html"
    if readme_path.exists() and screens_html.exists():
        readme_b64 = base64.b64encode(readme_path.read_bytes()).decode("ascii")
        html = screens_html.read_text(encoding="utf-8")
        if "__REPLACE_README__" in html:
            html = html.replace('__REPLACE_README__', readme_b64, 1)  # только первое вхождение (в var), не трогаем проверку в JS
            screens_html.write_text(html, encoding="utf-8")
            print(f"embed: README -> screens.html", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
