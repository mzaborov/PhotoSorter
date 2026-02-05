"""
Удаляет дублирующий блок «(план): сортировка...» из backend/web_api/templates/index.html.
Запуск: python backend/scripts/tools/remove_duplicate_div_index_html.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = REPO_ROOT / "backend" / "web_api" / "templates" / "index.html"

# Дублирующий блок (кавычки в файле — Unicode \u201c и \u201d, стрелка \u2192)
OLD_BLOCK = (
    '          <div class="muted" style="margin-top: 12px;">\n'
    "            (план): сортировка \u201cнет людей \u2192 время/места\u201d \u2192 определение людей \u2192 перенос по правилам.\n"
    "          </div>\n\n"
)


def main() -> None:
    if not INDEX_HTML.exists():
        print(f"Файл не найден: {INDEX_HTML}")
        return
    content = INDEX_HTML.read_text(encoding="utf-8")
    if OLD_BLOCK in content:
        content = content.replace(OLD_BLOCK, "")
        INDEX_HTML.write_text(content, encoding="utf-8")
        print("Removed duplicate div")
    else:
        # вариант без переноса в конце
        old2 = OLD_BLOCK.rstrip()
        if old2 in content:
            content = content.replace(old2, "")
            INDEX_HTML.write_text(content, encoding="utf-8")
            print("Removed (variant 2)")
        else:
            print("Pattern not found")


if __name__ == "__main__":
    main()
