"""Utilities for fetching data from the PanDA server.

PanDA server configuration
--------------------------
``HttpClient`` reads the server URL from the environment:

* ``PANDA_API_URL_SSL`` – HTTPS API base URL (default: ``https://pandaserver.cern.ch:25443/api/v1``)
* ``PANDA_AUTH``        – set to ``oidc`` to use OIDC token auth instead of X.509
* ``PANDA_AUTH_VO``     – VO name when using OIDC
* ``PANDA_AUTH_ID_TOKEN`` – OIDC token (or ``file:<path>``)
* ``X509_USER_PROXY``   – path to X.509 proxy certificate

Set these variables (or place them in your ``.env`` file) to point at a
different PanDA instance (e.g. a development server).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import gzip
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from bamboo.utils.narrator import say, thinking

import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings('ignore', category=InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-user PanDA identity (OIDC)
# ---------------------------------------------------------------------------
#
# By default ``HttpClient`` authenticates from process-global environment
# variables (PANDA_AUTH/PANDA_AUTH_ID_TOKEN/...). To let a multi-user frontend
# (e.g. the Mattermost bot) act as the *invoking* user, a per-call OIDC token can
# be bound for the duration of a request via :func:`panda_credentials`. It is
# stored in a ContextVar, which propagates across ``asyncio.to_thread`` (the
# context is copied), so the blocking ``HttpClient`` call running in a worker
# thread still sees the right token. When unset, behavior is unchanged (the
# client reads the environment as before).


@dataclass(frozen=True)
class PandaCredentials:
    """An OIDC token (and VO) to act as a specific PanDA user."""

    id_token: str
    auth_vo: str = ""


_panda_credentials: contextvars.ContextVar[Optional[PandaCredentials]] = contextvars.ContextVar(
    "panda_credentials", default=None
)


@contextlib.contextmanager
def panda_credentials(creds: Optional[PandaCredentials]):
    """Bind per-user PanDA OIDC credentials for the duration of the block.

    ``with panda_credentials(creds): await fetch_task_data(...)`` makes every
    PanDA API call inside act as that user. Passing ``None`` is a no-op (keeps
    the process-default identity).
    """
    token = _panda_credentials.set(creds)
    try:
        yield
    finally:
        _panda_credentials.reset(token)


def _apply_credentials(client: Any) -> None:
    """Apply the bound per-user OIDC credentials to *client*, if any.

    Uses ``HttpClient.override_oidc(oidc, id_token, auth_vo)`` to switch identity
    after construction. No-op when no credentials are bound.
    """
    creds = _panda_credentials.get()
    if creds is None:
        return
    override = getattr(client, "override_oidc", None)
    if override is None:  # pragma: no cover - older client without override
        logger.warning("HttpClient has no override_oidc; per-user identity ignored.")
        return
    override(True, creds.id_token, creds.auth_vo)

# Matches <a href="URL">...</a> in HTML fragments (e.g. errorDialog values).
_HREF_RE = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)


def extract_log_urls(text: str) -> list[str]:
    """Return all href URLs found in HTML anchor tags within *text*."""
    return _HREF_RE.findall(text)


def fetch_log_url(url: str, timeout: float = 30.0) -> str | None:
    """Fetch a plain-text log file from *url* synchronously.

    Returns the content as a string, or ``None`` on any error.
    Used by ``bamboo fetch-task --verbose`` to display linked log files.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("fetch_log_url: httpx not installed — cannot fetch %s", url)
        return None
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        if response.status_code != 200:
            logger.warning(
                "fetch_log_url: %s returned HTTP %s", url, response.status_code
            )
            return None
        return response.text
    except Exception as exc:
        logger.warning("fetch_log_url: failed to fetch %s: %s", url, exc)
        return None


async def async_fetch_log_content(url: str, timeout: float = 30.0) -> str | None:
    """Async version of :func:`fetch_log_url` — used inside async extractors.

    Returns the response body as a string, or ``None`` on any error
    (network failure, non-200 status, non-text content-type).
    Failures are logged at WARNING level and never interrupt extraction.
    """
    try:
        import httpx
    except ImportError:
        logger.warning(
            "async_fetch_log_content: httpx not installed — cannot fetch %s", url
        )
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
        if response.status_code != 200:
            logger.warning(
                "async_fetch_log_content: %s returned HTTP %s — skipping",
                url,
                response.status_code,
            )
            return None
        content_type = response.headers.get("content-type", "")
        content_encoding = response.headers.get("content-encoding", "")
        # httpx transparently decompresses Content-Encoding: gzip, so
        # response.content/.text is already plain text in that case.
        # Only manually decompress when the payload itself is a raw gzip
        # blob (indicated by Content-Type or a .gz URL suffix).
        needs_manual_decompress = (
            "application/gzip" in content_type or url.endswith(".gz")
        ) and "gzip" not in content_encoding
        if needs_manual_decompress:
            try:
                text = gzip.decompress(response.content).decode(
                    "utf-8", errors="replace"
                )
            except Exception as exc:
                logger.warning(
                    "async_fetch_log_content: failed to decompress gzip from %s: %s",
                    url,
                    exc,
                )
                return None
        else:
            text = response.text
        say(f"fetched {len(text):,} chars from {url}")
        return text
    except Exception as exc:
        logger.warning("async_fetch_log_content: failed to fetch %s: %s", url, exc)
        return None


def _call_api(method: str, endpoint: str, data: dict) -> Any:
    """Call a PanDA REST API endpoint and return the ``data`` field of the response.

    Raises ``RuntimeError`` on connection failure or a non-success server response.
    This is a blocking function; wrap with ``asyncio.to_thread`` in async contexts.
    """
    from pandaserver.api.v1.http_client import HttpClient, api_url_ssl  # noqa: PLC0415

    client = HttpClient()
    _apply_credentials(client)  # per-user OIDC override when bound (else env default)
    url = f"{api_url_ssl}/{endpoint}"
    if method == "get":
        status, response = client.get(url, data)
    else:
        status, response = client.post(url, data)

    if status != 0:
        raise RuntimeError(
            f"PanDA connection error for {endpoint!r}: {response}. "
            "Check PANDA_API_URL_SSL."
        )
    if not isinstance(response, dict) or not response.get("success"):
        message = response.get("message", "unknown error") if isinstance(response, dict) else str(response)
        raise RuntimeError(
            f"PanDA returned error for {endpoint!r}: {message}"
        )
    return response.get("data")


async def fetch_task_data(task_id: int | str, verbose: bool = False) -> dict[str, Any]:
    """Fetch full task details from PanDA and return them as a ``dict``.

    The blocking ``HttpClient`` call is offloaded to a thread pool so
    the event loop stays responsive.

    Args:
        task_id: The PanDA ``jediTaskID`` to look up (int or numeric string).
        verbose: If ``True``, sets the bamboo logger to DEBUG level, which
            is useful for diagnosing network or authentication issues.

    Returns:
        A ``dict`` containing the task details as returned by the PanDA
        server, ready to be passed as ``task_data`` to
        :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`
        or any extraction strategy.

    Raises:
        RuntimeError: If the PanDA server returns a non-zero status code or
                      an error response.
        ValueError:   If *task_id* cannot be converted to ``int``.
    """
    task_id_int = int(task_id)

    if verbose:
        logging.getLogger("bamboo").setLevel(logging.DEBUG)

    with thinking("Fetching task data from PanDA"):
        try:
            data = await asyncio.to_thread(
                _call_api, "get", "task/get_detailed_info", {"task_id": task_id_int}
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"PanDA returned error for task_id={task_id_int}: {exc}. "
                "Check PANDA_API_URL_SSL."
            ) from exc

    say(f"fetched task {task_id_int} details ({len(data)} fields)")
    return data


async def get_job_descriptions(
    task_id: int, unsuccessful_only: bool = False
) -> list[dict[str, Any]]:
    """Return job description dicts for *task_id*.

    Args:
        task_id: The PanDA ``jediTaskID``.
        unsuccessful_only: When ``True`` only failed, cancelled, and closed jobs
            are returned (server-side filter).

    Returns:
        List of job dicts.  Empty list on any error.
    """
    try:
        data = await asyncio.to_thread(
            _call_api,
            "get",
            "task/get_job_descriptions",
            {"task_id": task_id, "unsuccessful_only": unsuccessful_only},
        )
        return data if isinstance(data, list) else []
    except RuntimeError as exc:
        logger.warning("get_job_descriptions: failed for task_id=%s: %s", task_id, exc)
        return []


async def get_datasets_and_files(
    task_id: int,
    dataset_types: list[str] | None = None,
    dataset_only: bool = False,
) -> list[dict[str, Any]]:
    """Return dataset and file information for *task_id*.

    Each entry has the form ``{"dataset": {"name": ..., "id": ...}, "files": [...]}``.

    Args:
        task_id: The PanDA ``jediTaskID``.
        dataset_types: Dataset type filter (default: ``["input", "pseudo_input"]``).
        dataset_only: When ``True``, only return dataset information without associated files.

    Returns:
        List of dataset dicts.  Empty list on any error.
    """
    params: dict[str, Any] = {"task_id": task_id}
    if dataset_types:
        params["dataset_types"] = dataset_types
    if dataset_only:
        params["dataset_only"] = True
    try:
        data = await asyncio.to_thread(
            _call_api, "get", "task/get_datasets_and_files", params
        )
        return data if isinstance(data, list) else []
    except RuntimeError as exc:
        logger.warning("get_datasets_and_files: failed for task_id=%s: %s", task_id, exc)
        return []


def _parse_panda_datetime(value: Any) -> datetime | None:
    """Normalize a PanDA modificationTime-like value to a ``datetime``.

    Accepts a ``datetime`` (passthrough), a ``{"__datetime__": "<iso>"}`` wrapper
    dict (the encoding PanDA's wire protocol uses for datetime values when the
    receiving side has not applied ``pandaclient.Client.decode_special_cases``),
    or a string in either ISO 8601 form or the legacy ``YYYY-MM-DD HH:MM:SS``
    form. Returns ``None`` for anything else.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, dict):
        iso = value.get("__datetime__")
        if isinstance(iso, str):
            try:
                return datetime.fromisoformat(iso)
            except ValueError:
                return None
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


async def get_similar_successful_tasks(
    task_data: dict[str, Any],
    n_tasks: int = 50,
) -> list[dict[str, Any]]:
    """Find recently finished tasks similar to the failing task.

    Uses ``get_tasks_detailed_info_since`` with a ``filters`` dict.  Plain-value
    filters (``userName``, ``processingType``, ``transUses``, ``transHome``,
    ``architecture``) are pushed to SQL.  ``"finished|done"`` is applied via
    ``re.search`` server-side.

    Time window: the ``since`` cutoff is anchored on the failing task's
    ``modificationTime`` (``modificationTime - 14 days``) so the returned
    successful tasks are temporally comparable to the failure (same software
    release, queue conditions, etc.). If ``modificationTime`` is far enough
    in the past that the anchored cutoff would exceed the server's
    30-day-from-now hard limit, or if ``modificationTime`` is
    missing/unparseable, the function falls back to ``now - 30 days`` and
    logs a warning — the result in that case may not be temporally
    comparable.

    Args:
        task_data: The failing task's data dict.
        n_tasks: Maximum number of task IDs to retrieve before filtering.

    Returns:
        List of matching task detail dicts sorted by ``modificationTime`` descending.
        Empty list on any error.
    """
    WINDOW_DAYS = 14
    SERVER_LIMIT_DAYS = 30
    now = datetime.now()
    earliest_allowed = now - timedelta(days=SERVER_LIMIT_DAYS)

    raw_mtime = task_data.get("modificationTime")
    anchor = _parse_panda_datetime(raw_mtime)
    if raw_mtime and anchor is None:
        logger.warning(
            "get_similar_successful_tasks: cannot parse modificationTime=%r — "
            "falling back to now-anchored search (returned tasks may not be "
            "temporally comparable)",
            raw_mtime,
        )
    elif not raw_mtime:
        logger.warning(
            "get_similar_successful_tasks: task_data has no modificationTime — "
            "falling back to now-anchored search (returned tasks may not be "
            "temporally comparable)"
        )

    if anchor is None:
        since_dt = earliest_allowed
    else:
        desired = anchor - timedelta(days=WINDOW_DAYS)
        if desired < earliest_allowed:
            logger.warning(
                "get_similar_successful_tasks: failure modificationTime=%s is older "
                "than the server's %d-day window; falling back to now-anchored search "
                "(returned tasks may not be temporally comparable)",
                raw_mtime,
                SERVER_LIMIT_DAYS,
            )
            since_dt = earliest_allowed
        else:
            since_dt = desired
    since = since_dt.strftime("%Y-%m-%d %H:%M:%S")
    filters = {
        k: v
        for k, v in {
            "status": "finished|done",
            "userName": task_data.get("userName"),
            "prodSourceLabel": task_data.get("prodSourceLabel"),
            "processingType": task_data.get("processingType"),
            "transUses": task_data.get("transUses"),
            "transHome": task_data.get("transHome"),
            "architecture": task_data.get("architecture"),
        }.items()
        if v
    }
    params = {"since": since, "n_tasks": n_tasks, "filters": json.dumps(filters)}
    try:
        data = await asyncio.to_thread(
            _call_api, "get", "task/get_tasks_detailed_info_since", params
        )
        tasks = data if isinstance(data, list) else []
        return sorted(
            tasks,
            key=lambda t: _parse_panda_datetime(t.get("modificationTime")) or datetime.min,
            reverse=True,
        )
    except RuntimeError as exc:
        logger.warning("get_similar_successful_tasks: failed: %s", exc)
        return []


async def get_job_log(
    panda_id: int,
    filename: str = "payload.stdout",
) -> str | None:
    """Download a job log file from the PanDA monitor filebrowser.

    Args:
        panda_id: The ``PandaID`` of the job.
        filename: Log filename to fetch (default: ``payload.stdout``).

    Returns:
        Log content as a string, or ``None`` on any error.
    """
    base_url = os.getenv("PANDA_MONITOR_URL", "https://bigpanda.cern.ch")
    url = f"{base_url}/filebrowser/?pandaid={panda_id}&json&filename={filename}"
    return await async_fetch_log_content(url)


# ---------------------------------------------------------------------------
# Status histories and reads of the database
# ---------------------------------------------------------------------------
#
# job/get_status_history, task/get_status_history and
# async_process/submit_db_read_request answer with more than ``data``: a refusal
# says why in ``data`` ({"rejected": rule, ...} or {"error": code}) while
# ``success`` is false, and a poll says where the request is in a top-level
# ``async_meta``.  ``_call_api`` keeps neither, so these calls keep the whole
# response.

#: most job IDs one call of job/get_status_history takes (its max_keys, the
#: most one Oracle IN list holds)
STATUS_HISTORY_MAX_JOBS = 1000


def _call_api_full(method: str, endpoint: str, data: dict) -> dict[str, Any]:
    """Call a PanDA REST API endpoint and return the whole response.

    Unlike :func:`_call_api`, a response whose ``success`` is false is returned
    rather than raised: its ``data`` says why.  Raises ``RuntimeError`` only when
    there is no response to return.
    """
    from pandaserver.api.v1.http_client import HttpClient, api_url_ssl  # noqa: PLC0415

    client = HttpClient()
    _apply_credentials(client)  # per-user OIDC override when bound (else env default)
    url = f"{api_url_ssl}/{endpoint}"
    if method == "get":
        status, response = client.get(url, data)
    else:
        status, response = client.post(url, data)
    if status != 0:
        raise RuntimeError(
            f"PanDA connection error for {endpoint!r}: {response}. "
            "Check PANDA_API_URL_SSL."
        )
    if not isinstance(response, dict):
        raise RuntimeError(f"PanDA returned no response object for {endpoint!r}: {response!r}")
    return response


async def get_job_status_history(
    job_ids: list[int],
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """The status changes of *job_ids* in jobs_StatusLog, in as many calls as it takes.

    The server refuses a call naming more jobs than its probe budget allows for
    the window, and says how many partitions the window touches and what the
    budget is; the jobs are then split into calls of ``budget // partitions``.
    The IDs go in a JSON body (POST, which the server takes since panda-server
    79030c84): a GET puts each in the URL, and Apache refuses a request line
    past 8190 bytes, about 420 IDs, with an HTTP error rather than a refusal.

    Args:
        job_ids: PanDA IDs; one named twice is asked once.
        since, until: the window, in UTC (``2026-10-12 14:30:45``); the server
            widens it to whole days.

    Returns:
        On success ``{"success": True, "rows", "as_of", "retained_from",
        "since", "until", "calls"}``: the rows of every call in call order;
        as_of the earliest call's, since every change committed by then is in
        rows; retained_from the latest call's, or None when a call could not
        tell.  Otherwise the first response that failed, as it came, with
        ``calls`` added.  ``calls`` holds every response, for the record.
    """
    ids = list(dict.fromkeys(int(i) for i in job_ids))
    window = {k: v for k, v in (("since", since), ("until", until)) if v is not None}
    answers, calls = await _keyed_in_calls(
        "job/get_status_history", "job_ids", ids, window, STATUS_HISTORY_MAX_JOBS
    )
    if answers is None:
        return _failed(calls, "job_ids")
    return {
        "success": True,
        **_whole_span(answers),
        "since": answers[0].get("since"),
        "until": answers[0].get("until"),
        "calls": calls,
    }


async def _keyed_in_calls(
    endpoint: str, field: str, keys: list, extra: dict[str, Any], size: int
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Ask *endpoint* for *keys* in as many calls as its probe budget allows.

    The keyed reads (job/get_status_history, job/get_datasets_by_name) refuse a
    call naming more keys than ``budget // partitions`` of the table they read,
    and say both numbers; the keys are then split into calls of that size.  A
    read that looks each key up in more than one form (job/get_datasets_by_name
    from panda-server 100f79b4: a name with and without its scope) says how many
    in ``keys_per_name``, and both limits count those forms, so a call takes
    ``budget // (keys_per_name * partitions)`` or ``max_keys // keys_per_name``;
    a refusal without it counts one.

    Returns:
        ``(answers, calls)``: the ``data`` of every call in call order and
        every response as it came; answers is None when a call failed for
        another reason or not even one key fits, or when there was no key.
    """
    calls: list[dict[str, Any]] = []
    done = 0
    while done < len(keys):
        chunk = keys[done : done + size]
        response = await asyncio.to_thread(
            _call_api_full, "post", endpoint, {field: chunk, **extra}
        )
        calls.append(response)
        data = response.get("data") or {}
        if response.get("success"):
            done += len(chunk)
            continue
        fits = None
        per_key = data.get("keys_per_name") or 1
        if data.get("rejected") == "probe_budget" and data.get("partitions"):
            fits = data["budget"] // (per_key * data["partitions"])
        elif data.get("rejected") == "too_many_keys" and data.get("max_keys"):
            fits = data["max_keys"] // per_key
        if not fits or fits >= len(chunk):
            return None, calls
        size = fits
    if not calls:
        return None, calls
    return [c["data"] for c in calls if c.get("success")], calls


def _failed(calls: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """The response that ended a keyed read, as it came, with ``calls`` added."""
    if not calls:
        return {
            "success": False,
            "message": f"no {field}",
            "data": {"rejected": "no_keys"},
            "calls": calls,
        }
    return {**calls[-1], "calls": calls}


def _whole_span(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """The rows of every call, and the span all of them are whole for.

    as_of is the earliest call's, since every change committed by then is in
    rows; retained_from the latest call's, or None when a call could not tell.
    """
    retained = [a.get("retained_from") for a in answers]
    return {
        "rows": [row for a in answers for row in a["rows"]],
        "as_of": min((a["as_of"] for a in answers), key=_parse_panda_datetime),
        "retained_from": None
        if None in retained
        else max(retained, key=_parse_panda_datetime),
    }


async def get_task_status_history(task_id: int) -> dict[str, Any]:
    """The status changes of one task in TASKS_STATUSLOG: the whole response.

    On success ``data`` is ``{"rows", "as_of", "retained_from"}``.
    """
    return await asyncio.to_thread(
        _call_api_full, "get", "task/get_status_history", {"task_id": int(task_id)}
    )


# ---------------------------------------------------------------------------
# Rows read by key, without the async daemon
# ---------------------------------------------------------------------------
#
# A read by key goes through a plain endpoint rather than db_read, which waits
# behind the daemon's greps (up to ~250 s).  db_read keeps the aggregates and
# the reads by a time range.

#: most dataset names one call of job/get_datasets_by_name takes (its max_keys);
#: a refusal then says how many fit the probe budget
DATASETS_MAX_NAMES = 1000
#: job IDs per call of job/get_description_incl_archive.  It has no cost bound,
#: and a job in jobsArchived4 costs about 160 index lookups (panda-server,
#: 2026-10-14), so a call is kept to a few dozen
JOBS_PER_CALL = 30


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def get_datasets_by_name(names: list[str]) -> dict[str, Any]:
    """The rows of the server's datasets (ATLAS_PANDA.Datasets) named *names*.

    The table is partitioned by modificationdate, so each name is looked up in
    every partition; a call takes ``budget // partitions`` names, and the
    names are split into as many calls as that takes.  A server that looks a
    name up with and without its scope halves that (its ``keys_per_name``, see
    :func:`_keyed_in_calls`).  JEDI_Datasets is not read.

    Returns:
        On success ``{"success": True, "rows", "as_of", "retained_from",
        "calls"}``: every column of each row by its DatasetSpec name; a name
        with no row has none.  retained_from speaks of modificationdate: a
        dataset last modified before it may have been in the table and be
        gone.  Otherwise the response that failed, as it came, with ``calls``
        added.
    """
    keys = list(dict.fromkeys(str(n) for n in names))
    answers, calls = await _keyed_in_calls(
        "job/get_datasets_by_name", "names", keys, {}, DATASETS_MAX_NAMES
    )
    if answers is None:
        return _failed(calls, "names")
    return {"success": True, **_whole_span(answers), "calls": calls}


async def get_jobs_incl_archive(job_ids: list[int]) -> dict[str, Any]:
    """Each job's row and files: the live tables first, then the archive's last 90 days.

    job/get_description_incl_archive with ``report_errors``, so that a
    database error is told from a job not found.  The answer has no as_of: the
    server reads the tables in separate transactions.  ``asked_at`` is when the
    first call went, on this machine's clock (UTC).

    Returns:
        On success ``{"success": True, "jobs", "asked_at", "calls"}``: jobs maps
        each PanDA ID to the job's dict (every column by its JobSpec name, and
        ``Files``), to None when no live table has it and the archive's last 90
        days do not either, or to ``{"error": code}`` when the database failed
        for it (not retried).  Otherwise the response that failed, as it came,
        with ``asked_at`` and ``calls`` added.
    """
    ids = list(dict.fromkeys(int(i) for i in job_ids))
    asked_at = _utc_now()
    calls: list[dict[str, Any]] = []
    jobs: dict[int, Any] = {}
    for start in range(0, len(ids), JOBS_PER_CALL):
        chunk = ids[start : start + JOBS_PER_CALL]
        response = await asyncio.to_thread(
            _call_api_full,
            "post",
            "job/get_description_incl_archive",
            {"job_ids": chunk, "report_errors": True},
        )
        calls.append(response)
        data = response.get("data")
        if (
            not response.get("success")
            or not isinstance(data, list)
            or len(data) != len(chunk)
        ):
            return {**response, "asked_at": asked_at, "calls": calls}
        jobs.update(zip(chunk, data, strict=True))
    return {"success": True, "jobs": jobs, "asked_at": asked_at, "calls": calls}


async def get_task_detailed_info(task_id: int) -> dict[str, Any]:
    """One task's row and parameters: the whole response of task/get_detailed_info.

    ``data`` holds every JediTaskSpec column, with ``splitRule`` decoded, and
    keys that are not columns: ``split_rule_raw`` (the column as stored),
    ``job_execution_params``, ``task_creation_arguments``,
    ``raw_task_params`` and the input files' progress (``nFiles`` to
    ``pctFailed``).  The server answers alike for a task it has no row of and
    one it failed to read.  It has no as_of; ``asked_at`` is when the call
    went, on this machine's clock (UTC).
    """
    asked_at = _utc_now()
    response = await asyncio.to_thread(
        _call_api_full, "get", "task/get_detailed_info", {"task_id": int(task_id)}
    )
    return {**response, "asked_at": asked_at}


async def db_read(
    queries: list[dict[str, Any]],
    poll_seconds: float = 10.0,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Submit one batch of db_read queries and wait for its answers.

    The server's daemon takes requests every ~10 seconds and stops a db_read
    30 seconds before the daemon's own timeout or at 480 seconds, whichever
    comes first (270 s with the 300-s timeout it runs under).  A request that
    comes in while the daemon is running a grep waits for the next cycle, up
    to ~250 seconds, before it starts; hence the defaults.

    Returns:
        get_result's response once ``async_meta.status`` is ``done`` or
        ``failed``, with ``async_id`` added.  When done, ``data`` is
        ``{"as_of", "answers"}``, one answer per query (status done, rejected,
        deadline or error).  A refused submit comes back as it came.

    Raises:
        TimeoutError: past *timeout_seconds*; the message names the async_id,
            whose result can still be fetched.
    """
    submitted = await asyncio.to_thread(
        _call_api_full, "post", "async_process/submit_db_read_request", {"queries": queries}
    )
    if not submitted.get("success"):
        return submitted
    async_id = submitted["data"]["async_id"]
    waited = 0.0
    while True:
        result = await asyncio.to_thread(
            _call_api_full, "get", "async_process/get_result", {"async_id": async_id}
        )
        meta = result.get("async_meta")
        if meta is None:
            # no async_meta: this poll failed (not found, not authorized)
            return {**result, "async_id": async_id}
        if meta.get("status") in ("done", "failed"):
            return {**result, "async_id": async_id}
        if waited >= timeout_seconds:
            raise TimeoutError(f"db_read {async_id} still {meta.get('status')} after {waited:.0f} s")
        await asyncio.sleep(poll_seconds)
        waited += poll_seconds


# ---------------------------------------------------------------------------
# Log files over the async API (panda-server 67fff963 and later)
# ---------------------------------------------------------------------------
#
# submit_grep_request and submit_log_list_request are answered by every machine
# of a service, or by one machine: get_result reports one row per machine and
# says the request is "complete" once every expected machine has answered.
# A grep for whole records and a log list put JSON in each row's ``result``.

#: the machine groups one PanDA deployment runs as: the async API's service_name
LOG_SERVICES = ("server", "jedi")
#: most strings one grep takes (submit_grep_request's fixed_strings)
GREP_MAX_STRINGS = 1000
#: most log names one list takes (submit_log_list_request's log_names)
LIST_MAX_NAMES = 100


def _target(service: str | None, machine: str | None) -> dict[str, str]:
    if (service is None) == (machine is None):
        raise ValueError("name exactly one of service and machine")
    return (
        {"service_name": service} if service is not None else {"machine_name": machine}
    )


async def grep_log(
    log_filename: str,
    fixed_strings: list[str],
    *,
    service: str | None = None,
    machine: str | None = None,
    whole_records: bool = True,
    tail_bytes: int | None = None,
    max_matches: int | None = None,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Grep one log file for lines holding any of *fixed_strings*, and wait for the machines' answers.

    The strings are matched as they are (no escaping).  With *whole_records*
    each machine's result is ``{"records": [{"offset", "head", "lines",
    "cut"}], "window"}``: a matching line with the rest of its record, where
    ``head`` is false for a match inside a record whose first line did not
    match.  The daemon runs about one long grep per machine per cycle (300 s),
    so a wait may span many cycles when requests queue.

    Args:
        log_filename: a file under the server's logdir, ``panda-*.log`` or a
            rotated ``panda-*.log-YYYYMMDD.gz``.
        fixed_strings: 1 to :data:`GREP_MAX_STRINGS`; one named twice is sent once.
        service, machine: exactly one; a service means every machine alive in it.
        tail_bytes: search only the end of a plain file; the answer's
            ``window`` says where the search started.

    Returns:
        As :func:`wait_per_machine`; a refused submit as it came.
    """
    strings = list(dict.fromkeys(fixed_strings))
    if not 1 <= len(strings) <= GREP_MAX_STRINGS:
        raise ValueError(
            f"a grep takes 1 to {GREP_MAX_STRINGS} strings, not {len(strings)}"
        )
    params: dict[str, Any] = {
        "log_filename": log_filename,
        "fixed_strings": strings,
        "whole_records": whole_records,
        **_target(service, machine),
    }
    if tail_bytes is not None:
        params["tail_bytes"] = int(tail_bytes)
    if max_matches is not None:
        params["max_matches"] = int(max_matches)
    return await _per_machine_request(
        "async_process/submit_grep_request", params, poll_seconds, timeout_seconds
    )


async def list_logs(
    *,
    service: str | None = None,
    machine: str | None = None,
    log_names: list[str] | None = None,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """The log files of every machine of a service, or of one machine, and wait for the answers.

    Each machine's result is a list of ``{"name", "size", "mtime"}`` (mtime in
    UTC, ``YYYY-MM-DD HH:MM:SS``): every regular file under its logdir whose
    name starts with ``panda-``, in the order of their names, cut at 1 MB
    (``truncated``).  A rotated file's mtime is its last write before it was
    rotated, not the rotation: its processes go on writing to it until
    postrotate stops them (up to 146 s past its mtime on 2026-10-14), and a
    quiet log's mtime is hours before.  A jedi machine keeps 180 days of
    about 75 logs: its whole list never fits, so ask for the logs by name.

    Args:
        log_names: only these logs and their rotated files (a name ``n``
            matches ``n`` and ``n-*``), up to :data:`LIST_MAX_NAMES`.  A name
            with no file is not in the list.

    Returns:
        As :func:`wait_per_machine`; a refused submit as it came.
    """
    params = _target(service, machine)
    if log_names is not None:
        names = list(dict.fromkeys(log_names))
        if not 1 <= len(names) <= LIST_MAX_NAMES:
            raise ValueError(
                f"a list takes 1 to {LIST_MAX_NAMES} log names, not {len(names)}"
            )
        params["log_names"] = names
    return await _per_machine_request(
        "async_process/submit_log_list_request",
        params,
        poll_seconds,
        timeout_seconds,
    )


async def _per_machine_request(
    endpoint: str, params: dict[str, Any], poll_seconds: float, timeout_seconds: float
) -> dict[str, Any]:
    submitted = await asyncio.to_thread(_call_api_full, "post", endpoint, params)
    if not submitted.get("success"):
        return submitted
    return await wait_per_machine(
        submitted["data"]["async_id"], poll_seconds, timeout_seconds
    )


async def wait_per_machine(
    async_id: str, poll_seconds: float = 15.0, timeout_seconds: float = 3600.0
) -> dict[str, Any]:
    """Poll get_result for a per-machine request until every expected machine has answered.

    Returns:
        The last poll with ``async_id`` added.  Its ``data.overall_status`` is
        ``complete``, or ``pending`` when the wait ended first: the request
        goes on in the server, and this can be called again with the same id.
        A poll that failed (not found, not authorized) comes back as it came.
    """
    waited = 0.0
    while True:
        result = await asyncio.to_thread(
            _call_api_full, "get", "async_process/get_result", {"async_id": async_id}
        )
        data = result.get("data") or {}
        if (
            not result.get("success")
            or data.get("overall_status") == "complete"
            or waited >= timeout_seconds
        ):
            return {**result, "async_id": async_id}
        await asyncio.sleep(poll_seconds)
        waited += poll_seconds


def per_machine(
    response: dict[str, Any], json_result: bool
) -> dict[str, dict[str, Any]]:
    """Each machine's answer in a per-machine response, by machine name.

    Every expected machine has an entry: one that has not answered is
    ``{"status": "no result"}``, since silence is not "found nothing".  An
    answer is ``{"status", "truncated", "error", "return_code", "result"}``,
    with *result* parsed from JSON when *json_result* (a grep for whole
    records, a log list).  A file that is not there is a ``failed`` status
    whose error says "No such file or directory".
    """
    data = response.get("data") or {}
    rows = {row.get("machine_name"): row for row in data.get("results") or []}
    out: dict[str, dict[str, Any]] = {}
    for machine in dict.fromkeys([*(data.get("expected_machines") or []), *rows]):
        row = rows.get(machine)
        if row is None:
            out[machine] = {"status": "no result"}
            continue
        result = row.get("result")
        if json_result and result:
            result = json.loads(result)
        out[machine] = {
            "status": row.get("status"),
            "truncated": bool(row.get("truncated")),
            "error": row.get("error_msg"),
            "return_code": row.get("return_code"),
            "result": result,
        }
    return out
