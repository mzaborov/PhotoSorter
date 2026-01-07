from __future__ import annotations

from DB.db import get_connection, init_db


def main() -> int:
    init_db()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT code, name, path, location, role, sort_order
            FROM folders
            ORDER BY
              (sort_order IS NULL) ASC,
              sort_order ASC,
              name ASC
            """
        )
        rows = cur.fetchall()
        print("count:", len(rows))
        for r in rows:
            print(
                f"{r['sort_order'] if r['sort_order'] is not None else ''}\t"
                f"{r['code']}\t{r['name']}\t{r['path']}\t{r['role']}"
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


















