from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from common.db import DedupStore, FaceStore, PipelineStore
from web_api.routers import gold as gold_api

_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

router = APIRouter()

_WEB_API_DIR = Path(__file__).resolve().parents[1]
_templates = Jinja2Templates(directory=str(_WEB_API_DIR / "templates"))


@router.get("/sorting-folders", response_class=HTMLResponse)
def sorting_folders_page(request: Request) -> HTMLResponse:
    """Шаг 4 — Сортировка по финальным папкам (результаты: sorting-folders)."""
    return _templates.TemplateResponse("sorting_folders.html", {"request": request})

# Локальный "конвейер" (ML в отдельном процессе через .venv-face).
_LOCAL_PIPELINE_EXEC = ThreadPoolExecutor(max_workers=1)
_LOCAL_PIPELINE_LOCK = threading.Lock()
_LOCAL_PIPELINE_FUTURE: Any = None
_LOCAL_PIPELINE_RUN_ID: Optional[int] = None
_LOCAL_PIPELINE_STATE: dict[str, Any] = {
    "running": False,
    "root_path": None,
    "apply": False,
    "skip_dedup": False,
    "no_dedup_move": False,
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "error": None,
    "log": "",
}


def _repo_root() -> Path:
    # backend/web_api/routers/local_pipeline.py -> repo root
    return Path(__file__).resolve().parents[3]


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _local_pipeline_log_append(line: str) -> None:
    with _LOCAL_PIPELINE_LOCK:
        s = _LOCAL_PIPELINE_STATE.get("log") or ""
        s = (s + (line or "")).replace("\r\n", "\n")
        if len(s) > 120_000:
            s = s[-120_000:]
        _LOCAL_PIPELINE_STATE["log"] = s
        run_id = _LOCAL_PIPELINE_RUN_ID
    if run_id is not None:
        ps = PipelineStore()
        try:
            try:
                ps.append_log(run_id=int(run_id), line=line or "")
            except sqlite3.OperationalError as e:
                # ВАЖНО (Windows/SQLite): при активной записи из worker-процесса SQLite может кратко держать write-lock.
                # Логи НЕ должны ронять весь прогон: если БД занята — держим лог в памяти и идём дальше.
                if "locked" in str(e).lower():
                    return
                raise
        finally:
            ps.close()


def _run_local_pipeline(*, root_path: str, apply: bool, skip_dedup: bool, no_dedup_move: bool, pipeline_run_id: int) -> None:
    rr = _repo_root()
    py = rr / ".venv-face" / "Scripts" / "python.exe"
    script = rr / "backend" / "scripts" / "tools" / "local_sort_by_faces.py"

    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE.update(
            {
                "running": True,
                "root_path": root_path,
                "apply": bool(apply),
                "skip_dedup": bool(skip_dedup),
                "no_dedup_move": bool(no_dedup_move),
                "started_at": _now_utc_iso(),
                "finished_at": None,
                "exit_code": None,
                "error": None,
                "log": "",
            }
        )
        global _LOCAL_PIPELINE_RUN_ID  # noqa: PLW0603
        _LOCAL_PIPELINE_RUN_ID = int(pipeline_run_id)

    if not py.exists():
        _local_pipeline_log_append(f"ERROR: not found: {py}\n")
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": 2, "error": ".venv-face python not found"})
        return
    if not script.exists():
        _local_pipeline_log_append(f"ERROR: not found: {script}\n")
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": 2, "error": "local_sort_by_faces.py not found"})
        return

    cmd: list[str] = [
        str(py),
        str(script.relative_to(rr)),
        "--root",
        root_path,
        "--pipeline-run-id",
        str(int(pipeline_run_id)),
    ]
    if apply:
        cmd.append("--apply")
    if skip_dedup:
        cmd.append("--skip-dedup")
    if no_dedup_move:
        cmd.append("--no-dedup-move")

    try:
        vs = int(os.getenv("LOCAL_PIPELINE_VIDEO_SAMPLES") or "0")
    except Exception:
        vs = 0
    vs = max(0, min(3, int(vs)))
    if vs > 0:
        cmd += ["--video-samples", str(vs)]
        try:
            vmd = int(os.getenv("LOCAL_PIPELINE_VIDEO_MAX_DIM") or "0")
        except Exception:
            vmd = 0
        if vmd and vmd > 0:
            cmd += ["--video-max-dim", str(int(vmd))]

    _local_pipeline_log_append("RUN: " + " ".join(cmd) + "\n")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(rr)

        p = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(rr),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert p.stdout is not None
        ps = PipelineStore()
        try:
            try:
                ps.update_run(run_id=int(pipeline_run_id), pid=int(p.pid), status="running")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # best-effort: не валим прогон из-за transient lock
                    pass
                else:
                    raise
        finally:
            ps.close()
        for line in p.stdout:
            _local_pipeline_log_append(line)
        rc = int(p.wait())
        ps2 = PipelineStore()
        try:
            try:
                ps2.update_run(
                    run_id=int(pipeline_run_id),
                    status="completed" if rc == 0 else "failed",
                    last_error="" if rc == 0 else f"exit_code={rc}",
                    finished_at=_now_utc_iso(),
                )
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    pass
                else:
                    raise
        finally:
            ps2.close()
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": rc, "error": None if rc == 0 else f"exit_code={rc}"})
    except Exception as e:  # noqa: BLE001
        _local_pipeline_log_append(f"ERROR: {type(e).__name__}: {e}\n")
        ps3 = PipelineStore()
        try:
            try:
                ps3.update_run(run_id=int(pipeline_run_id), status="failed", last_error=f"{type(e).__name__}: {e}", finished_at=_now_utc_iso())
            except sqlite3.OperationalError as e2:
                if "locked" in str(e2).lower():
                    _local_pipeline_log_append(f"ERROR: не удалось записать статус failed в БД (database is locked): {e2}\n")
                    _local_pipeline_log_append(f"ERROR: исходная ошибка: {type(e).__name__}: {e}\n")
                else:
                    raise
        finally:
            ps3.close()
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": 1, "error": f"{type(e).__name__}: {e}"})


def _get_step4_total(pipeline_run_id: int) -> int | None:
    """Возвращает общее число файлов в прогоне (total + unsorted_count) из step4-report."""
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/api/faces/step4-report?pipeline_run_id={pipeline_run_id}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8")) if r.length else {}
        total = data.get("total") or 0
        unsorted = data.get("unsorted_count") or 0
        return int(total) + int(unsorted)
    except Exception:
        return None


def _run_step4_only(pipeline_run_id: int, apply: bool, clear_first: bool = False) -> None:
    """Выполняет только шаг 4: опционально обнуление target_folder → fill-target-folders → sort-into-folders (через HTTP к API)."""
    global _LOCAL_PIPELINE_RUN_ID  # noqa: PLW0603
    step4_total = _get_step4_total(pipeline_run_id) or 0
    # #region agent log
    try:
        _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
        _dl.write(json.dumps({"hypothesisId": "H1,H2", "location": "local_pipeline.py:_run_step4_only", "message": "step4_total from _get_step4_total", "data": {"pipeline_run_id": pipeline_run_id, "step4_total": step4_total}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
        _dl.close()
    except Exception:
        pass
    # #endregion
    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE.update(
            {
                "running": True,
                "root_path": None,
                "apply": bool(apply),
                "started_at": _now_utc_iso(),
                "finished_at": None,
                "exit_code": None,
                "error": None,
                "log": "",
                "step4_pct": 0,
                "step4_phase": "clear" if clear_first else "fill",
                "step4_processed": 0,
                "step4_total": step4_total,
            }
        )
        _LOCAL_PIPELINE_RUN_ID = int(pipeline_run_id)
    # Прогресс 0–50% = fill, 50–100% = move; в БД step4_total = 2 * кол-во файлов
    step4_total_doubled = max(0, int(step4_total) * 2)
    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE["step4_total"] = step4_total_doubled
    ps_start = PipelineStore()
    try:
        ps_start.update_run(run_id=pipeline_run_id, step4_processed=0, step4_total=step4_total_doubled)
        # #region agent log
        try:
            _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
            _dl.write(json.dumps({"hypothesisId": "H1,H2", "location": "local_pipeline.py:after update_run start", "message": "wrote step4_processed=0 step4_total to DB", "data": {"pipeline_run_id": pipeline_run_id, "step4_total": step4_total}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
            _dl.close()
        except Exception:
            pass
        # #endregion
    finally:
        ps_start.close()

    # Не меняем status прогона в БД: при «только шаг 4» прогон остаётся completed (после шага 3),
    # чтобы прогресс шагов 1–3 не сбрасывался в UI. Факт «шаг 4 идёт» только в _LOCAL_PIPELINE_STATE.
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=pipeline_run_id)
        root_path = str(pr.get("root_path") or "") if pr else ""
    finally:
        ps.close()

    _local_pipeline_log_append(f"STEP4: pipeline_run_id={pipeline_run_id} apply={apply} clear_first={clear_first} root={root_path}\n")

    if clear_first:
        _local_pipeline_log_append("STEP4: clear-target-folders-for-run...\n")
        try:
            from web_api.routers.faces import clear_target_folders_for_run_impl as _clear_impl
            data_clear = _clear_impl(pipeline_run_id)
        except Exception as e_clear:
            err_clear = str(e_clear)
            _local_pipeline_log_append(f"STEP4: clear-target-folders error: {err_clear}\n")
            with _LOCAL_PIPELINE_LOCK:
                _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": 1, "error": err_clear})
            ps_clear = PipelineStore()
            try:
                ps_clear.update_run(run_id=pipeline_run_id, status="failed", last_error=err_clear or "clear-target-folders failed", finished_at=_now_utc_iso())
            finally:
                ps_clear.close()
            return
        cleared = data_clear.get("cleared_count") or 0
        _local_pipeline_log_append(f"STEP4: cleared target_folder for {cleared} files\n")
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE["step4_pct"] = 15
            _LOCAL_PIPELINE_STATE["step4_phase"] = "fill"
        ps_clear2 = PipelineStore()
        try:
            ps_clear2.update_run(run_id=pipeline_run_id, step4_processed=0)
        finally:
            ps_clear2.close()

    def _post(path: str, body: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str]:
        req = urllib.request.Request(
            f"{_API_BASE}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read().decode("utf-8")) if r.length else {}
                return True, data, ""
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
                try:
                    err_data = json.loads(err_body)
                    msg = err_data.get("detail", err_body)
                except Exception:
                    msg = err_body
            except Exception:
                msg = str(e)
            return False, None, msg
        except Exception as e:
            return False, None, str(e)

    # 1) Заполнить целевые папки (прямой вызов, без HTTP — иначе дедлок при одном воркере)
    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE["step4_pct"] = 25
        _LOCAL_PIPELINE_STATE["step4_phase"] = "fill"
    _local_pipeline_log_append("STEP4: fill-target-folders...\n")
    # Ленивый импорт, чтобы избежать циклических зависимостей при загрузке модуля
    try:
        from web_api.routers.faces import fill_target_folders_impl as _fill_impl
    except ImportError as _e_imp:
        _local_pipeline_log_append(f"STEP4: ImportError fill_target_folders_impl: {_e_imp}\n")
        try:
            _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
            _dl.write(json.dumps({"location": "local_pipeline.py:fill import", "message": "ImportError", "data": {"error": str(_e_imp)}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
            _dl.close()
        except Exception:
            pass
        raise RuntimeError(f"Не удалось загрузить fill_target_folders_impl: {_e_imp}") from _e_imp
    try:
        try:
            _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
            _dl.write(json.dumps({"location": "local_pipeline.py:before fill call", "message": "calling fill_target_folders_impl", "data": {"pipeline_run_id": pipeline_run_id}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
            _dl.close()
        except Exception:
            pass
        data_fill = _fill_impl(pipeline_run_id)
    except Exception as e:
        err_fill = str(e)
        # #region agent log
        try:
            _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
            _dl.write(json.dumps({"hypothesisId": "fill-fail", "location": "local_pipeline.py:fill except", "message": "fill_target_folders_impl raised", "data": {"error": err_fill, "type": type(e).__name__}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
            _dl.close()
        except Exception:
            pass
        # #endregion
        _local_pipeline_log_append(f"STEP4: fill-target-folders error: {err_fill}\n")
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": 1, "error": err_fill})
        ps2 = PipelineStore()
        try:
            try:
                ps2.update_run(run_id=pipeline_run_id, status="failed", last_error=err_fill or "fill-target-folders failed", finished_at=_now_utc_iso())
            except sqlite3.OperationalError as e2:
                if "locked" in str(e2).lower():
                    # Не глушить: пишем в лог прогона, чтобы пользователь видел и блокировку, и исходную ошибку
                    _local_pipeline_log_append(f"STEP4: не удалось записать статус failed в БД (database is locked): {e2}\n")
                    _local_pipeline_log_append(f"STEP4: исходная ошибка: {err_fill}\n")
                else:
                    raise
        finally:
            ps2.close()
        return
    filled = data_fill.get("filled_count") or 0
    _local_pipeline_log_append(f"STEP4: fill-target-folders ok, filled={filled}\n")
    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE["step4_pct"] = 50
        _LOCAL_PIPELINE_STATE["step4_phase"] = "move"
        _LOCAL_PIPELINE_STATE["step4_processed"] = filled
        step4_total_state = _LOCAL_PIPELINE_STATE.get("step4_total")
        if step4_total_state is None:
            _LOCAL_PIPELINE_STATE["step4_total"] = max(filled * 2, filled)
            step4_total_state = _LOCAL_PIPELINE_STATE["step4_total"]
    ps_fill = PipelineStore()
    try:
        ps_fill.update_run(run_id=pipeline_run_id, step4_processed=int(filled), step4_total=int(step4_total_state))
    finally:
        ps_fill.close()

    # 2) Переместить локально (прямой вызов, без HTTP — иначе дедлок при одном воркере)
    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE["step4_pct"] = 55
    _local_pipeline_log_append("STEP4: sort-into-folders (destination=local)...\n")
    try:
        from web_api.routers.faces import sort_into_folders_impl as _sort_impl
        data_sort = _sort_impl(pipeline_run_id=pipeline_run_id, dry_run=not apply, destination="local")
    except Exception as e_sort:
        err_sort = str(e_sort)
        _local_pipeline_log_append(f"STEP4: sort-into-folders error: {err_sort}\n")
        with _LOCAL_PIPELINE_LOCK:
            _LOCAL_PIPELINE_STATE.update({"running": False, "finished_at": _now_utc_iso(), "exit_code": 1, "error": err_sort})
        ps3 = PipelineStore()
        try:
            ps3.update_run(run_id=pipeline_run_id, status="failed", last_error=err_sort or "sort-into-folders failed", finished_at=_now_utc_iso())
        finally:
            ps3.close()
        return
    moved = data_sort.get("moved_count") or 0
    _local_pipeline_log_append(f"STEP4: sort-into-folders ok, moved={moved}\n")

    step4_total_final = _LOCAL_PIPELINE_STATE.get("step4_total") or 0
    with _LOCAL_PIPELINE_LOCK:
        _LOCAL_PIPELINE_STATE.update({
            "running": False,
            "finished_at": _now_utc_iso(),
            "exit_code": 0,
            "error": None,
            "step4_pct": 100,
            "step4_processed": int(step4_total_final),
        })
    ps4 = PipelineStore()
    try:
        ps4.update_run(
            run_id=pipeline_run_id,
            status="completed",
            last_error="",
            finished_at=_now_utc_iso(),
            step4_processed=int(step4_total_final),
            step4_total=int(step4_total_final),
        )
    finally:
        ps4.close()


@router.post("/api/local-pipeline/start")
def api_local_pipeline_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    root_path = str(payload.get("root_path") or "").strip()
    apply = bool(payload.get("apply") or False)
    skip_dedup = bool(payload.get("skip_dedup") or False)
    start_mode = str(payload.get("start_mode") or "").strip().lower()
    resume_run_id = payload.get("resume_run_id")
    no_dedup_move = False

    if not root_path:
        raise HTTPException(status_code=400, detail="root_path is required")
    if root_path.startswith("disk:"):
        raise HTTPException(status_code=400, detail="Only local folders are supported here (use C:\\... path)")
    if not os.path.isdir(root_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {root_path}")

    rr = _repo_root()
    py = rr / ".venv-face" / "Scripts" / "python.exe"
    script = rr / "backend" / "scripts" / "tools" / "local_sort_by_faces.py"
    if not py.exists():
        raise HTTPException(status_code=500, detail="Missing .venv-face (Python 3.12) — create it before running ML pipeline")
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"Missing: {script}")

    ps = PipelineStore()
    try:
        latest = ps.get_latest_run(kind="local_sort", root_path=root_path)
        resumed = False
        prev_run_id: int | None = None
        prev_face_run_id: int | None = None
        prev_root_like: str | None = None
        prev_status: str | None = None
        if latest:
            try:
                prev_run_id = int(latest.get("id") or 0) or None
            except Exception:
                prev_run_id = None
            try:
                prev_face_run_id = int(latest.get("face_run_id") or 0) or None
            except Exception:
                prev_face_run_id = None
            prev_status = str(latest.get("status") or "")
            try:
                prev_root_like = gold_api._root_like_for_pipeline_run_id(int(prev_run_id)) if prev_run_id else None
            except Exception:
                prev_root_like = None

        if resume_run_id is not None:
            try:
                rid = int(resume_run_id)
            except Exception:
                raise HTTPException(status_code=400, detail="resume_run_id must be int") from None
            pr0 = ps.get_run_by_id(run_id=rid)
            if not pr0:
                raise HTTPException(status_code=404, detail="resume_run_id not found")
            if str(pr0.get("kind") or "") != "local_sort":
                raise HTTPException(status_code=400, detail="resume_run_id kind mismatch")
            if str(pr0.get("root_path") or "") != root_path:
                raise HTTPException(status_code=400, detail="resume_run_id root_path mismatch")
            st0 = str(pr0.get("status") or "")
            if st0 not in ("failed", "running"):
                raise HTTPException(status_code=400, detail=f"resume_run_id not resumable (status={st0})")
            same_apply = bool(int(pr0.get("apply") or 0)) == bool(apply)
            same_skip = bool(int(pr0.get("skip_dedup") or 0)) == bool(skip_dedup)
            if not (same_apply and same_skip):
                raise HTTPException(status_code=400, detail="resume_run_id options mismatch (apply/skip_dedup)")
            pipeline_run_id = rid
            resumed = True
            ps.update_run(run_id=pipeline_run_id, status="running", last_error="", finished_at="")
        elif start_mode == "new":
            pipeline_run_id = ps.create_run(kind="local_sort", root_path=root_path, apply=apply, skip_dedup=skip_dedup)
        elif latest:
            same_opts = bool(int(latest.get("apply") or 0)) == bool(apply) and bool(int(latest.get("skip_dedup") or 0)) == bool(skip_dedup)
            st = str(latest.get("status") or "")
            if st in ("failed",) and same_opts:
                pipeline_run_id = int(latest["id"])
                resumed = True
                ps.update_run(run_id=pipeline_run_id, status="running", last_error="", finished_at="")
            elif st == "running" and same_opts:
                updated_at = str(latest.get("updated_at") or "")
                is_stale = True
                try:
                    from datetime import datetime, timezone

                    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    is_stale = (datetime.now(timezone.utc) - dt).total_seconds() > 180
                except Exception:
                    is_stale = True
                if not is_stale:
                    return {"ok": False, "message": "local pipeline already running", "run_id": int(latest["id"])}
                pipeline_run_id = int(latest["id"])
                resumed = True
                ps.update_run(run_id=pipeline_run_id, status="running", last_error="stale: restarting worker", finished_at="")
            else:
                pipeline_run_id = ps.create_run(kind="local_sort", root_path=root_path, apply=apply, skip_dedup=skip_dedup)
        else:
            pipeline_run_id = ps.create_run(kind="local_sort", root_path=root_path, apply=apply, skip_dedup=skip_dedup)
    finally:
        ps.close()

    try:
        if not bool(resumed):
            if prev_run_id and int(prev_run_id) != int(pipeline_run_id) and (prev_status in ("completed", "failed")) and prev_face_run_id:
                psm = PipelineStore()
                try:
                    already = psm.get_metrics_for_run(pipeline_run_id=int(prev_run_id))
                    if not already:
                        ds = DedupStore()
                        fs = FaceStore()
                        try:
                            cats = gold_api._count_misses_for_gold(
                                ds=ds,
                                pipeline_run_id=int(prev_run_id),
                                face_run_id=int(prev_face_run_id),
                                root_like=prev_root_like,
                                gold_name="cats_gold",
                                expected_tab="animals",
                            )
                            faces = gold_api._count_misses_for_gold(
                                ds=ds,
                                pipeline_run_id=int(prev_run_id),
                                face_run_id=int(prev_face_run_id),
                                root_like=prev_root_like,
                                gold_name="faces_gold",
                                expected_tab="faces",
                            )
                            no_faces = gold_api._count_misses_for_gold(
                                ds=ds,
                                pipeline_run_id=int(prev_run_id),
                                face_run_id=int(prev_face_run_id),
                                root_like=prev_root_like,
                                gold_name="no_faces_gold",
                                expected_tab="no_faces",
                            )
                        finally:
                            ds.close()
                        step2_total = step2_processed = None
                        try:
                            fr = fs.get_run_by_id(run_id=int(prev_face_run_id)) or {}
                            step2_total = fr.get("total_files")
                            step2_processed = fr.get("processed_files")
                        finally:
                            fs.close()
                        psm.upsert_metrics(
                            pipeline_run_id=int(prev_run_id),
                            metrics={
                                "computed_at": _now_utc_iso(),
                                "face_run_id": int(prev_face_run_id),
                                "step2_total": step2_total,
                                "step2_processed": step2_processed,
                                "cats_total": cats.get("total"),
                                "cats_mism": cats.get("mism"),
                                "faces_total": faces.get("total"),
                                "faces_mism": faces.get("mism"),
                                "no_faces_total": no_faces.get("total"),
                                "no_faces_mism": no_faces.get("mism"),
                            },
                        )
                finally:
                    psm.close()
    except Exception:
        pass

    try:
        ps_fix = PipelineStore()
        try:
            pr = ps_fix.get_run_by_id(run_id=int(pipeline_run_id))
        finally:
            ps_fix.close()
        dedup_run_id = pr.get("dedup_run_id") if pr else None
        if dedup_run_id:
            ds_fix = DedupStore()
            try:
                dr = ds_fix.get_run_by_id(run_id=int(dedup_run_id))
                if dr and str(dr.get("status") or "") == "completed":
                    total = dr.get("total_files")
                    proc = dr.get("processed_files")
                    if total is not None:
                        try:
                            ti = int(total)
                            pi = int(proc or 0)
                            if ti > 0 and pi < ti:
                                ds_fix.update_run_progress(run_id=int(dedup_run_id), processed_files=ti)
                        except Exception:
                            pass
            finally:
                ds_fix.close()
    except Exception:
        pass

    # При resume: если шаг 3 уже выполнен (face_run_id задан), продолжаем только шаг 4 (не перезапускаем 1–3).
    run_only_step4 = bool(resumed) and pr is not None and pr.get("face_run_id")
    global _LOCAL_PIPELINE_FUTURE  # noqa: PLW0603
    with _LOCAL_PIPELINE_LOCK:
        fut = _LOCAL_PIPELINE_FUTURE
        if fut is not None and not fut.done():
            return {"ok": False, "message": "local pipeline already running"}
        global _LOCAL_PIPELINE_RUN_ID  # noqa: PLW0603
        _LOCAL_PIPELINE_RUN_ID = int(pipeline_run_id)
        if run_only_step4:
            _LOCAL_PIPELINE_FUTURE = _LOCAL_PIPELINE_EXEC.submit(
                _run_step4_only,
                pipeline_run_id=int(pipeline_run_id),
                apply=apply,
            )
        else:
            _LOCAL_PIPELINE_FUTURE = _LOCAL_PIPELINE_EXEC.submit(
                _run_local_pipeline,
                root_path=root_path,
                apply=apply,
                skip_dedup=skip_dedup,
                no_dedup_move=no_dedup_move,
                pipeline_run_id=int(pipeline_run_id),
            )

    return {
        "ok": True,
        "message": "step4_only" if run_only_step4 else "started",
        "run_id": int(pipeline_run_id),
        "resumed": bool(resumed),
        "start_mode": start_mode or ("continue" if resumed else "new"),
        "root_path": root_path,
        "apply": apply,
        "skip_dedup": skip_dedup,
        "no_dedup_move": no_dedup_move,
    }


@router.post("/api/local-pipeline/start-step4")
def api_local_pipeline_start_step4(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Запускает только шаг 4 (заполнить целевые папки → переместить локально).
    Требуется pipeline_run_id с уже выполненным шагом 3 (face_run_id задан).
    """
    pipeline_run_id = payload.get("pipeline_run_id")
    if not isinstance(pipeline_run_id, int):
        raise HTTPException(status_code=400, detail="pipeline_run_id is required")
    apply = bool(payload.get("apply") or False)
    clear_first = bool(payload.get("clear_first") or False)

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        raise HTTPException(status_code=404, detail="pipeline_run_id not found")
    if str(pr.get("kind") or "") != "local_sort":
        raise HTTPException(status_code=400, detail="run is not local_sort")
    if not pr.get("face_run_id"):
        raise HTTPException(status_code=400, detail="face_run_id not set (step 3 not done); run full pipeline first")

    global _LOCAL_PIPELINE_FUTURE  # noqa: PLW0603
    with _LOCAL_PIPELINE_LOCK:
        fut = _LOCAL_PIPELINE_FUTURE
        if fut is not None and not fut.done():
            return {"ok": False, "message": "local pipeline or step4 already running", "run_id": int(pipeline_run_id)}
        global _LOCAL_PIPELINE_RUN_ID  # noqa: PLW0603
        _LOCAL_PIPELINE_RUN_ID = int(pipeline_run_id)
        _LOCAL_PIPELINE_FUTURE = _LOCAL_PIPELINE_EXEC.submit(
            _run_step4_only,
            pipeline_run_id=int(pipeline_run_id),
            apply=apply,
            clear_first=clear_first,
        )

    return {
        "ok": True,
        "message": "step4 started",
        "run_id": int(pipeline_run_id),
        "apply": apply,
        "clear_first": clear_first,
    }


@router.get("/api/local-pipeline/latest")
def api_local_pipeline_latest(root_path: str) -> dict[str, Any]:
    rp = str(root_path or "").strip()
    if not rp:
        raise HTTPException(status_code=400, detail="root_path is required")
    ps = PipelineStore()
    try:
        pr = ps.get_latest_run(kind="local_sort", root_path=rp)
    finally:
        ps.close()
    return {"ok": True, "latest": pr}


@router.get("/api/local-pipeline/status")
def api_local_pipeline_status() -> dict[str, Any]:
    ps = PipelineStore()
    try:
        pr = ps.get_latest_any(kind="local_sort")
    finally:
        ps.close()

    if not pr:
        return {
            "running": False,
            "exit_code": None,
            "error": None,
            "run_id": None,
            "root_path": None,
            "apply": False,
            "skip_dedup": False,
            "step": {"num": 0, "total": 0, "title": ""},
            "step1": {"status": "idle", "pct": 0, "processed": None, "total": None},
            "step2": {"status": "idle", "pct": 0, "images": None, "total": None, "faces": None},
            "step3": {"status": "idle", "pct": 0},
            "log_tail": "",
        }

    status = str(pr.get("status") or "")
    running = status == "running"
    exit_code: int | None
    if status == "completed":
        exit_code = 0
    elif status == "failed":
        le = str(pr.get("last_error") or "")
        if le.startswith("exit_code="):
            try:
                exit_code = int(le.split("=", 1)[1].strip())
            except Exception:
                exit_code = 1
        else:
            exit_code = 1
    else:
        exit_code = None

    step_num = int(pr.get("step_num") or 0)
    step_total = int(pr.get("step_total") or 0)
    step_title = str(pr.get("step_title") or "")

    log = str(pr.get("log_tail") or "").replace("\r\n", "\n")
    log_tail = "\n".join(log.splitlines()[-160:])

    video_samples_env: int | None
    try:
        video_samples_env = int(os.getenv("LOCAL_PIPELINE_VIDEO_SAMPLES") or "0")
    except Exception:
        video_samples_env = None

    def _last_run_cmd_from_log(log_text: str) -> str | None:
        for line in reversed((log_text or "").splitlines()):
            if line.startswith("RUN: "):
                return line[len("RUN: ") :].strip()
        return None

    run_cmd = _last_run_cmd_from_log(log_tail)
    cmd_has_video_samples = bool(run_cmd and "--video-samples" in run_cmd)
    cmd_video_samples: int | None = None
    if run_cmd and "--video-samples" in run_cmd:
        try:
            parts = run_cmd.split()
            i = parts.index("--video-samples")
            if i + 1 < len(parts):
                cmd_video_samples = int(parts[i + 1])
        except Exception:
            cmd_video_samples = None

    step0_checked = step0_non_media = step0_broken_media = None
    try:
        import re

        # Берём ПОСЛЕДНЕЕ совпадение: при resume в log_tail могут оставаться старые строки (checked=200),
        # но нас интересует актуальный прогресс.
        matches = list(re.finditer(r"preclean:\s*checked=(\d+)\s+moved_non_media=(\d+)\s+moved_broken_media=(\d+)", log_tail))
        if matches:
            m0 = matches[-1]
            step0_checked = int(m0.group(1))
            step0_non_media = int(m0.group(2))
            step0_broken_media = int(m0.group(3))
    except Exception:
        pass

    title_l = (step_title or "").lower()
    if status == "completed":
        step0_status = "done"
        step0_pct = 100
    elif status == "failed":
        step0_status = "error" if ("предочист" in title_l or step_num <= 1) else "done"
        step0_pct = 100 if step0_status == "done" else (50 if step0_checked is not None else 0)
    elif running:
        if "предочист" in title_l:
            step0_status = "running"
            step0_pct = 50 if step0_checked is not None else 10
        elif step_num >= 2:
            step0_status = "done"
            step0_pct = 100
        else:
            step0_status = "pending"
            step0_pct = 0
    else:
        step0_status = "idle"
        step0_pct = 0

    dedup_proc = dedup_total = None
    dedup_run_id = pr.get("dedup_run_id")
    if dedup_run_id:
        ds = DedupStore()
        try:
            dr = ds.get_run_by_id(run_id=int(dedup_run_id))
        finally:
            ds.close()
        if dr:
            dedup_proc = dr.get("processed_files")
            dedup_total = dr.get("total_files")

    if dedup_total and int(dedup_total) > 0 and dedup_proc is not None:
        step1_pct = int(round((int(dedup_proc) / int(dedup_total)) * 100))
        step1_pct = max(0, min(100, step1_pct))
    else:
        step1_pct = 100 if step_num >= 2 or status == "completed" else 0
    if step_num >= 2 or status == "completed":
        step1_status = "done"
    elif running and step_num == 1:
        step1_status = "running"
    elif status == "failed" and step_num <= 1:
        step1_status = "error"
    else:
        step1_status = "idle"

    faces_img = faces_total = None
    faces_found = None
    face_run_id = pr.get("face_run_id")
    if face_run_id:
        fs = FaceStore()
        try:
            fr = fs.get_run_by_id(run_id=int(face_run_id))
        finally:
            fs.close()
        if fr:
            faces_img = fr.get("processed_files")
            faces_total = fr.get("total_files")
            faces_found = fr.get("faces_found")

    def _scan_pct() -> int:
        if faces_total and int(faces_total) > 0 and faces_img is not None:
            pct = int(round((int(faces_img) / int(faces_total)) * 90))
            return max(0, min(90, pct))
        return 0

    scan_pct = _scan_pct()
    is_split_phase = "разлож" in step_title.lower()
    # Шаг 4 идёт отдельно (без смены status в БД): определяем по in-memory state.
    step4_only_running = (
        not running
        and _LOCAL_PIPELINE_RUN_ID is not None
        and pr.get("id") == _LOCAL_PIPELINE_RUN_ID
        and _LOCAL_PIPELINE_STATE.get("running") is True
    )
    # Шаги 5/6: «Переместить локально» / «Перенести в архив» — прогресс из БД, статус done после успешного завершения
    step4_phase = (str(pr.get("step4_phase") or "")).strip()
    step5_running = step4_phase == "local"
    step6_running = step4_phase == "archive"
    step5_done = bool(int(pr.get("step5_done") or 0))
    step6_done = bool(int(pr.get("step6_done") or 0))

    # Шаг 3 (лица): при failed прогоне считаем выполненным, если дошли до step_num >= 2; ошибку относим к шагу 4.
    if status == "completed":
        step2_pct = 100
        step2_status = "done"
    elif running:
        if step_num < 2:
            step2_pct = 0
            step2_status = "pending"
        else:
            step2_status = "running"
            step2_pct = scan_pct
            if is_split_phase:
                step2_pct = max(step2_pct, 95)
    elif status == "failed":
        if step_num >= 2:
            step2_status = "done"
            step2_pct = 100
        else:
            step2_status = "error" if step_num >= 2 else "idle"
            step2_pct = scan_pct
        if is_split_phase:
            step2_pct = max(step2_pct, 95)
    else:
        step2_pct = 0
        step2_status = "idle"

    # Шаг 4 (разложить по правилам): при step4_only_running показываем running и прогресс из БД (видят все воркеры).
    step3_processed = None
    step3_total = None
    if step4_only_running:
        step3_status = "running"
        step3_processed = pr.get("step4_processed")
        step3_total = pr.get("step4_total")
        if step3_total is not None and int(step3_total) > 0 and step3_processed is not None:
            step3_pct = int(round((int(step3_processed) / int(step3_total)) * 100))
            step3_pct = max(0, min(100, step3_pct))
        else:
            step3_pct = int(_LOCAL_PIPELINE_STATE.get("step4_pct") or 0)
            step3_pct = max(0, min(100, step3_pct))
        # #region agent log
        try:
            _dl = open(r"c:\Projects\PhotoSorter\.cursor\debug.log", "a", encoding="utf-8")
            _dl.write(json.dumps({"hypothesisId": "H1,H3", "location": "local_pipeline.py:status step4", "message": "step4 progress from pr", "data": {"step4_processed": step3_processed, "step4_total": step3_total, "step3_pct": step3_pct, "run_id": pr.get("id")}, "timestamp": time.time()}, ensure_ascii=False) + "\n")
            _dl.close()
        except Exception:
            pass
        # #endregion
    elif status == "completed":
        step3_status = "done"
        step3_pct = 100
    elif status == "failed" and step_num >= 2:
        step3_status = "error"
        step3_pct = 0
    elif running and step_num >= 2:
        step3_status = "pending" if not is_split_phase else "running"
        step3_pct = 0
    else:
        step3_status = "idle"
        step3_pct = 0

    # Шаг 5 (переместить локально) и шаг 6 (перенести в архив): прогресс из step4_phase + step4_processed/step4_total
    step5_processed = step5_total = step5_pct = None
    step6_processed = step6_total = step6_pct = None
    sp = pr.get("step4_processed")
    st = pr.get("step4_total")
    if step5_running or step6_running:
        if st is not None and int(st) > 0 and sp is not None:
            pct = int(round((int(sp) / int(st)) * 100))
            pct = max(0, min(100, pct))
            if step5_running:
                step5_processed = int(sp)
                step5_total = int(st)
                step5_pct = pct
            else:
                step6_processed = int(sp)
                step6_total = int(st)
                step6_pct = pct
    elif step5_done and st is not None and int(st) > 0:
        # При «Выполнено» показываем total/total, чтобы не было «5/2476 · 100%» после прерванного прогона
        step5_processed = int(st)
        step5_total = int(st)
        step5_pct = 100
    elif step6_done and st is not None and int(st) > 0:
        step6_processed = int(st)
        step6_total = int(st)
        step6_pct = 100
    step5_status = "running" if step5_running else ("done" if step5_done else "idle")
    step6_status = "running" if step6_running else ("done" if step6_done else "idle")
    # Был прогресс, но не done и не running — значит прервано (например перезапуск uvicorn)
    if step5_status == "idle" and sp is not None and int(sp) > 0 and not step5_done:
        step5_status = "interrupted"
    if step6_status == "idle" and sp is not None and int(sp) > 0 and not step6_done:
        step6_status = "interrupted"
    if step5_pct is None:
        step5_pct = 100 if step5_done else 0
    if step6_pct is None:
        step6_pct = 100 if step6_done else 0

    # Чтобы главная страница показывала «прогон идёт», возвращаем running=True и при только шаге 4, и при шагах 5/6.
    if step4_only_running or step5_running or step6_running:
        running = True

    plan_total = 6
    if "предочист" in title_l:
        plan_num = 1
        plan_title = "предочистка"
    elif "дедуп" in title_l:
        plan_num = 2
        plan_title = "дедупликация"
    elif "лица" in title_l:
        plan_num = 3
        plan_title = "лица/животные/нет людей"
    else:
        plan_num = max(0, int(step_num or 0))
        plan_title = step_title

    return {
        "running": running,
        "exit_code": exit_code,
        "error": pr.get("last_error"),
        "run_id": pr.get("id"),
        "root_path": pr.get("root_path"),
        "apply": bool(int(pr.get("apply") or 0)),
        "skip_dedup": bool(int(pr.get("skip_dedup") or 0)),
        "started_at": pr.get("started_at"),
        "updated_at": pr.get("updated_at"),
        "finished_at": pr.get("finished_at"),
        "step": {"num": step_num, "total": step_total, "title": step_title},
        "plan_step": {"num": int(plan_num), "total": int(plan_total), "title": str(plan_title or "")},
        "step0": {"status": step0_status, "pct": step0_pct, "checked": step0_checked, "non_media": step0_non_media, "broken_media": step0_broken_media},
        "step1": {"status": step1_status, "pct": step1_pct, "processed": dedup_proc, "total": dedup_total},
        "step2": {"status": step2_status, "pct": step2_pct, "images": faces_img, "total": faces_total, "faces": faces_found},
        "step3": {"status": step3_status, "pct": step3_pct, "processed": step3_processed, "total": step3_total},
        "step4_phase": step4_phase,
        "step5": {"status": step5_status, "pct": step5_pct, "processed": step5_processed, "total": step5_total},
        "step6": {"status": step6_status, "pct": step6_pct, "processed": step6_processed, "total": step6_total},
        "face_run_id": int(face_run_id) if face_run_id else None,
        "debug": {"video_samples_env": video_samples_env, "cmd_has_video_samples": cmd_has_video_samples, "cmd_video_samples": cmd_video_samples},
        "log_tail": log_tail,
    }

