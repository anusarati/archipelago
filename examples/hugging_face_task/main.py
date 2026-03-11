#!/usr/bin/env python3
"""
Run a task from the mercor/apex-agents HuggingFace dataset.

Usage:
    ./run.sh              # Run task index 0
    ./run.sh 42           # Run task index 42
    ./run.sh task_abc123  # Run task by ID
    ./run.sh world_abc123 # Run all tasks in a world
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

EXAMPLE_DIR = Path(os.environ.get("EXAMPLE_DIR", Path(__file__).parent))
ARCHIPELAGO_DIR = Path(os.environ.get("ARCHIPELAGO_DIR", EXAMPLE_DIR.parent.parent))
ENVIRONMENT_DIR = Path(
    os.environ.get("ENVIRONMENT_DIR", ARCHIPELAGO_DIR / "environment")
)
AGENTS_DIR = Path(os.environ.get("AGENTS_DIR", ARCHIPELAGO_DIR / "agents"))
GRADING_DIR = Path(os.environ.get("GRADING_DIR", ARCHIPELAGO_DIR / "grading"))

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8080")
HF_DATASET = "mercor/apex-agents"
INFRA_FAILURE_EXIT_CODE = 42
DEFAULT_MCP_CONFIG_FILE = "mcp_config_all_oss_servers.json"
WORLD_APP_TO_SERVER = {
    "calendar": "calendar_server",
    "chat": "chat_server",
    "code execution": "code_execution_server",
    "excel": "sheets_server",
    "filesystem": "filesystem_server",
    "mail": "mail_server",
    "pdfs": "pdf_server",
    "powerpoint": "slides_server",
    "word": "docs_server",
}

# Default task: Investment Banking World 221 - BBDC/TVPG accretion/dilution sensitivity analysis
DEFAULT_TASK = "task_9ba58a6197114140877a1df1754d2993"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_progress(scope: str, current: int, total: int, msg: str):
    pct = (current / total) * 100 if total > 0 else 0
    log(f"[{scope}] {current}/{total} ({pct:.0f}%) - {msg}")


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def wait_for_health(url: str, timeout: int = 120) -> bool:
    """Wait for environment to be healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    return False


def start_environment():
    """Start a fresh environment container (always restarts)."""
    env_file = ENVIRONMENT_DIR / ".env"
    env_example = ENVIRONMENT_DIR / ".env.example"
    if not env_file.exists() and env_example.exists():
        log("Creating .env from .env.example...")
        shutil.copy(env_example, env_file)
    elif not env_file.exists():
        log("Creating empty .env file...")
        env_file.touch()

    log("Stopping any existing environment containers...")
    subprocess.run(
        ["docker", "compose", "down", "-v"], cwd=ENVIRONMENT_DIR, capture_output=True
    )

    log("Building and starting environment container...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--build"], cwd=ENVIRONMENT_DIR
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to start environment")

    log("Waiting for environment to be healthy...")
    if not wait_for_health(ENV_URL):
        subprocess.run(["docker", "compose", "logs"], cwd=ENVIRONMENT_DIR)
        raise RuntimeError("Environment failed to start")

    log("Environment started")


def ensure_environment_ready():
    """Reuse a healthy environment, or start one if unavailable."""
    if wait_for_health(ENV_URL, timeout=5):
        log("Environment already healthy; reusing existing container")
        return
    log("Environment not healthy; starting container")
    start_environment()


def tar_gz_to_zip(tar_gz_path: Path) -> Path:
    """Convert tar.gz to zip for grading."""
    stem = tar_gz_path.stem
    if stem.endswith(".tar"):
        stem = stem[:-4]
    zip_path = tar_gz_path.parent / f"{stem}.zip"
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        zf.writestr(member.name, f.read())
    return zip_path


def _parse_existing_world_summary(
    summary_file: Path, world_tasks: list[dict[str, object]]
) -> tuple[dict[str, dict[str, object]], str | None]:
    """Load existing world summary entries keyed by task_id."""
    if not summary_file.exists():
        return {}, None

    try:
        with open(summary_file) as f:
            payload = json.load(f)
    except Exception as e:
        log(f"WARNING: Could not read existing summary at {summary_file}: {e}")
        return {}, None

    raw_summary = payload.get("summary", [])
    if not isinstance(raw_summary, list):
        return {}, None

    valid_task_ids = {
        str(t.get("task_id")) for t in world_tasks if isinstance(t.get("task_id"), str)
    }
    summary_by_task: dict[str, dict[str, object]] = {}
    for entry in raw_summary:
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("task_id")
        if not isinstance(task_id, str):
            continue
        if task_id not in valid_task_ids:
            continue
        summary_by_task[task_id] = dict(entry)

    old_stop_reason = payload.get("stop_reason")
    if old_stop_reason is not None and not isinstance(old_stop_reason, str):
        old_stop_reason = None

    return summary_by_task, old_stop_reason


def _build_world_summary(
    world_tasks: list[dict[str, object]], summary_by_task: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Build ordered world summary list matching world task order."""
    return [
        summary_by_task[t["task_id"]]
        for t in world_tasks
        if t["task_id"] in summary_by_task
    ]


def _write_world_summary(
    summary_file: Path,
    world_id: str,
    world_name: str,
    world_tasks: list[dict[str, object]],
    summary_by_task: dict[str, dict[str, object]],
    stop_reason: str | None,
) -> list[dict[str, object]]:
    """Persist world summary file and return ordered summary records."""
    summary = _build_world_summary(world_tasks, summary_by_task)
    with open(summary_file, "w") as f:
        json.dump(
            {
                "world_id": world_id,
                "world_name": world_name,
                "task_count": len(world_tasks),
                "tasks_executed": len(summary),
                "stopped_early": stop_reason is not None,
                "stop_reason": stop_reason,
                "summary": summary,
            },
            f,
            indent=2,
        )
    return summary


def _should_retry_task_on_resume(entry: dict[str, object] | None) -> bool:
    """Return True when a prior task outcome should be retried on world resume."""
    if not entry:
        return True

    return_code = entry.get("return_code")
    status = entry.get("agent_status")

    # Environment/bootstrap failure while configuring MCP servers.
    if return_code == INFRA_FAILURE_EXIT_CODE:
        return True

    # Transient/system errors (e.g., API rate limits/timeouts/provider issues).
    if status == "error":
        return True

    # User interruptions, cancellations, or missing/unknown status should rerun.
    if status in ("cancelled", "interrupted", None):
        return True

    # Non-zero script exit with non-terminal status should rerun.
    if isinstance(return_code, int) and return_code != 0 and status not in (
        "completed",
        "failed",
    ):
        return True

    # Terminal outcomes: completed or failed (e.g., max-steps/turn-limit).
    return False


def main():
    # Parse task selector from command line (index, task ID, or use default)
    task_selector = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    # Load task/world metadata (allow parent process to pass cached paths)
    tasks_path_env = os.environ.get("HF_TASKS_JSON_PATH")
    worlds_path_env = os.environ.get("HF_WORLDS_JSON_PATH")

    if tasks_path_env and worlds_path_env:
        tasks_path = tasks_path_env
        worlds_path = worlds_path_env
        log("Using cached task/world metadata from environment")
    else:
        log("Downloading task/world metadata from HuggingFace...")
        tasks_path = hf_hub_download(
            HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset"
        )
        worlds_path = hf_hub_download(
            HF_DATASET, "world_descriptions.json", repo_type="dataset"
        )

    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(worlds_path) as f:
        worlds = {w["world_id"]: w for w in json.load(f)}

    # World mode: run all tasks for a specific world sequentially
    if task_selector.startswith("world_"):
        world_id = task_selector
        world_tasks = [t for t in tasks if t.get("world_id") == world_id]
        if not world_tasks:
            log(f"ERROR: No tasks found for world: {world_id}")
            sys.exit(1)

        world_name = worlds.get(world_id, {}).get("world_name", "Unknown world")
        log("=" * 60)
        log(f"WORLD RUN: {world_name} ({world_id})")
        log(f"Tasks to run: {len(world_tasks)}")
        log("=" * 60)

        # Download this world's snapshot exactly once and reuse for all tasks.
        log("Preparing world snapshot cache...")
        world_zip_cached = hf_hub_download(
            HF_DATASET, f"world_files_zipped/{world_id}.zip", repo_type="dataset"
        )
        log(f"Using cached world snapshot: {world_zip_cached}")

        summary_dir = EXAMPLE_DIR / "output" / world_id
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / "summary.json"
        summary_by_task: dict[str, dict[str, object]] = {}
        script_path = Path(__file__).resolve()
        stop_reason: str | None = None
        continue_on_infra_failure = (
            os.environ.get("WORLD_CONTINUE_ON_INFRA_FAILURE", "false").lower()
            in ("1", "true", "yes")
        )
        world_resume = env_truthy("WORLD_RESUME", default=True)
        reuse_world_environment = env_truthy("WORLD_REUSE_ENVIRONMENT", default=True)

        if world_resume:
            summary_by_task, old_stop_reason = _parse_existing_world_summary(
                summary_file, world_tasks
            )
            skipped_task_ids = {
                task_id
                for task_id, entry in summary_by_task.items()
                if not _should_retry_task_on_resume(entry)
            }
            retry_task_ids = {
                task_id
                for task_id, entry in summary_by_task.items()
                if _should_retry_task_on_resume(entry)
            }
            if skipped_task_ids:
                log(
                    f"Resuming world run from existing summary: {len(skipped_task_ids)} terminal task(s) will be skipped"
                )
            if retry_task_ids:
                log(
                    f"Resuming world run from existing summary: {len(retry_task_ids)} retryable task(s) will be rerun"
                )
            if old_stop_reason:
                log(f"Previous world run stop reason: {old_stop_reason}")

        if reuse_world_environment:
            log("Ensuring shared environment for world run...")
            ensure_environment_ready()

        current_task_id: str | None = None
        task_name_by_id = {
            str(t["task_id"]): str(t.get("task_name", t["task_id"])) for t in world_tasks
        }

        try:
            for i, world_task in enumerate(world_tasks, start=1):
                task_id = world_task["task_id"]
                task_name = world_task.get("task_name", task_id)
                log("-" * 60)
                log_progress(
                    scope="WORLD",
                    current=i,
                    total=len(world_tasks),
                    msg=f"Running {task_id} ({task_name})",
                )

                existing_entry = summary_by_task.get(task_id)
                if world_resume and existing_entry and not _should_retry_task_on_resume(
                    existing_entry
                ):
                    log(f"Skipping terminal task from previous run: {task_id}")
                    continue
                if world_resume and existing_entry:
                    log(
                        "Retrying task from previous run due to retryable status: "
                        f"{task_id} (return_code={existing_entry.get('return_code')}, "
                        f"agent_status={existing_entry.get('agent_status')})"
                    )

                child_env = os.environ.copy()
                child_env["HF_TASKS_JSON_PATH"] = str(tasks_path)
                child_env["HF_WORLDS_JSON_PATH"] = str(worlds_path)
                child_env["HF_WORLD_ZIP_PATH"] = str(world_zip_cached)
                if reuse_world_environment:
                    child_env["HF_SKIP_START_ENVIRONMENT"] = "1"

                current_task_id = task_id
                result = subprocess.run(
                    [sys.executable, str(script_path), task_id], env=child_env
                )
                current_task_id = None

                task_output_dir = EXAMPLE_DIR / "output" / task_id
                trajectory_status = None
                final_score = None

                trajectory_file = task_output_dir / "trajectory.json"
                if trajectory_file.exists():
                    with open(trajectory_file) as f:
                        trajectory = json.load(f)
                    trajectory_status = trajectory.get("status")

                grades_file = task_output_dir / "grades.json"
                if grades_file.exists():
                    with open(grades_file) as f:
                        grades = json.load(f)
                    final_score = grades.get("scoring_results", {}).get("final_score")

                summary_by_task[task_id] = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "world_id": world_id,
                    "return_code": result.returncode,
                    "agent_status": trajectory_status,
                    "final_score": final_score,
                }

                # Persist progress after each task so resume works after interruption.
                _ = _write_world_summary(
                    summary_file=summary_file,
                    world_id=world_id,
                    world_name=world_name,
                    world_tasks=world_tasks,
                    summary_by_task=summary_by_task,
                    stop_reason=stop_reason,
                )

                if result.returncode == INFRA_FAILURE_EXIT_CODE:
                    stop_reason = (
                        f"Infrastructure failure while configuring MCP servers on task {task_id}"
                    )
                    log(f"ERROR: {stop_reason}")
                    if continue_on_infra_failure:
                        log(
                            "WORLD_CONTINUE_ON_INFRA_FAILURE=true, continuing despite infra failure"
                        )
                    else:
                        log("Stopping world run early due to infra failure")
                        break
        except KeyboardInterrupt:
            interrupted_task = current_task_id
            if interrupted_task:
                summary_by_task.setdefault(
                    interrupted_task,
                    {
                        "task_id": interrupted_task,
                        "task_name": task_name_by_id.get(
                            interrupted_task, interrupted_task
                        ),
                        "world_id": world_id,
                        "return_code": 130,
                        "agent_status": "interrupted",
                        "final_score": None,
                    },
                )
                stop_reason = f"Interrupted by user (Ctrl+C) while running task {interrupted_task}"
            else:
                stop_reason = "Interrupted by user (Ctrl+C)"
            log(f"Stopping world run: {stop_reason}")

        summary = _write_world_summary(
            summary_file=summary_file,
            world_id=world_id,
            world_name=world_name,
            world_tasks=world_tasks,
            summary_by_task=summary_by_task,
            stop_reason=stop_reason,
        )

        success_count = sum(1 for s in summary if s["return_code"] == 0)
        failure_count = len(summary) - success_count

        log("=" * 60)
        log("WORLD RUN SUMMARY")
        log("=" * 60)
        log(f"World: {world_name} ({world_id})")
        log(f"Total tasks: {len(world_tasks)}")
        log(f"Tasks executed: {len(summary)}")
        log(f"Succeeded: {success_count}")
        log(f"Failed: {failure_count}")
        if stop_reason:
            log(f"Stopped early: {stop_reason}")
        log(f"Summary file: {summary_file}")

        if failure_count > 0 or stop_reason:
            sys.exit(1)
        return

    # Find the task
    if task_selector.isdigit():
        task_index = int(task_selector)
        if task_index < 0 or task_index >= len(tasks):
            log(f"ERROR: Task index out of range (0-{len(tasks) - 1})")
            sys.exit(1)
        task = tasks[task_index]
    else:
        task = next((t for t in tasks if t["task_id"] == task_selector), None)
        if not task:
            log(f"ERROR: Task not found: {task_selector}")
            sys.exit(1)

    world_id = task["world_id"]
    world = worlds.get(world_id)
    if not world:
        log(f"ERROR: World not found: {world_id}")
        sys.exit(1)

    trajectory_id = f"hf_{task['task_id']}_{uuid.uuid4().hex[:8]}"
    grading_run_id = f"gr_{uuid.uuid4().hex[:8]}"
    output_dir = EXAMPLE_DIR / "output" / task["task_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log(f"Task: {task['task_name']}")
    log(f"Domain: {task['domain']}")
    log(f"World: {world['world_name']}")
    log(f"Prompt: {task['prompt'][:100]}...")
    log("=" * 60)
    step = 0
    total_steps = 7

    step += 1
    skip_start_environment = env_truthy("HF_SKIP_START_ENVIRONMENT", default=False)
    if skip_start_environment:
        log_progress(task["task_id"], step, total_steps, "Reusing environment")
        ensure_environment_ready()
    else:
        log_progress(task["task_id"], step, total_steps, "Starting environment")
        start_environment()

    # Download and extract world snapshot
    step += 1
    log_progress(task["task_id"], step, total_steps, "Preparing world snapshot")
    cached_world_zip = os.environ.get("HF_WORLD_ZIP_PATH")
    if cached_world_zip and Path(cached_world_zip).exists():
        cached_path = Path(cached_world_zip)
        if cached_path.name == f"{world_id}.zip":
            zip_path = str(cached_path)
            log(f"Using cached world snapshot from parent: {zip_path}")
        else:
            log(
                f"Cached world snapshot {cached_path.name} does not match {world_id}.zip, downloading correct world snapshot..."
            )
            zip_path = hf_hub_download(
                HF_DATASET, f"world_files_zipped/{world_id}.zip", repo_type="dataset"
            )
    else:
        log(f"Downloading world snapshot: {world_id}")
        zip_path = hf_hub_download(
            HF_DATASET, f"world_files_zipped/{world_id}.zip", repo_type="dataset"
        )
    world_zip = output_dir / f"{world_id}.zip"
    shutil.copy(zip_path, world_zip)

    step += 1
    log_progress(task["task_id"], step, total_steps, "Populating environment")
    log("Populating environment with world snapshot...")
    with zipfile.ZipFile(world_zip, "r") as zf:
        names = zf.namelist()

        for subsystem in ["filesystem", ".apps_data"]:
            subsystem_files = [n for n in names if n.startswith(f"{subsystem}/")]
            if not subsystem_files:
                continue

            log(f"  Populating {subsystem} ({len(subsystem_files)} files)...")
            subsystem_tar = output_dir / f"{subsystem}.tar.gz"

            with tarfile.open(subsystem_tar, "w:gz") as tar:
                for name in subsystem_files:
                    new_name = name[len(f"{subsystem}/") :]
                    if not new_name:
                        continue
                    info = tarfile.TarInfo(name=new_name)
                    if name.endswith("/"):
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        tar.addfile(info)
                    else:
                        data = zf.read(name)
                        info.size = len(data)
                        info.mode = 0o644
                        tar.addfile(info, io.BytesIO(data))

            with open(subsystem_tar, "rb") as f:
                resp = httpx.post(
                    f"{ENV_URL}/data/populate",
                    files={
                        "archive": (f"{subsystem}.tar.gz", f.read(), "application/gzip")
                    },
                    params={"subsystem": subsystem},
                    timeout=600.0,
                )
                if resp.status_code != 200:
                    log(f"ERROR: Failed to populate {subsystem}: {resp.text}")
                    sys.exit(1)
                log(f"  {subsystem}: {resp.json()}")

    # Configure MCP servers
    step += 1
    log_progress(task["task_id"], step, total_steps, "Configuring MCP servers")
    log("Configuring MCP servers...")
    mcp_config_override = os.environ.get("MCP_CONFIG_FILE")
    mcp_config: dict[str, object]
    if mcp_config_override:
        mcp_config_path = Path(mcp_config_override)
        if not mcp_config_path.is_absolute():
            mcp_config_path = EXAMPLE_DIR / mcp_config_override
        if not mcp_config_path.exists():
            log(f"ERROR: MCP config file not found: {mcp_config_path}")
            sys.exit(1)
        log(f"  MCP config: {mcp_config_path.name} (override)")
        with open(mcp_config_path) as f:
            mcp_config = json.load(f)
    else:
        base_config_path = EXAMPLE_DIR / DEFAULT_MCP_CONFIG_FILE
        with open(base_config_path) as f:
            full_config = json.load(f)

        configured_servers = full_config.get("mcpServers", {})
        world_id = task.get("world_id")
        world_info = worlds.get(world_id, {})
        world_apps = world_info.get("apps", [])
        selected_servers: list[str] = []
        unknown_services: list[str] = []

        if isinstance(world_apps, list) and world_apps:
            for app in world_apps:
                if not isinstance(app, dict):
                    continue
                service_name = str(app.get("service_name", "")).strip()
                if not service_name:
                    continue
                server_name = WORLD_APP_TO_SERVER.get(service_name.lower())
                if server_name is None:
                    unknown_services.append(service_name)
                    continue
                if server_name in configured_servers and server_name not in selected_servers:
                    selected_servers.append(server_name)

        if selected_servers:
            mcp_config = {
                "mcpServers": {
                    server_name: configured_servers[server_name]
                    for server_name in selected_servers
                }
            }
            log(f"  MCP config: {DEFAULT_MCP_CONFIG_FILE} (filtered by world apps)")
        else:
            mcp_config = full_config
            log(f"  MCP config: {DEFAULT_MCP_CONFIG_FILE} (no app filter)")

        if unknown_services:
            log(f"  Unknown world services (ignored): {unknown_services}")

    log(f"  Servers: {list(mcp_config['mcpServers'].keys())}")

    max_apps_attempts = int(os.environ.get("MCP_CONFIGURE_MAX_ATTEMPTS", "3"))
    apps_retry_delay = float(os.environ.get("MCP_CONFIGURE_RETRY_DELAY_SECONDS", "2"))
    for apps_attempt in range(1, max_apps_attempts + 1):
        log(f"  /apps attempt {apps_attempt}/{max_apps_attempts}...")
        try:
            resp = httpx.post(f"{ENV_URL}/apps", json=mcp_config, timeout=600.0)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            if status == 503:
                log("ERROR: MCP gateway readiness failed (503 Service Unavailable)")
                try:
                    payload = e.response.json()
                    detail = payload.get("detail", {}) if isinstance(payload, dict) else {}
                    failed_servers = detail.get("failed_servers", [])
                    server_details = detail.get("details", {})
                    if failed_servers:
                        log(f"Failed servers: {', '.join(failed_servers)}")
                        for server in failed_servers:
                            d = server_details.get(server, {})
                            err = d.get("error", "unknown error")
                            attempts = d.get("attempts", "unknown")
                            log(f"  - {server}: {err} (attempts={attempts})")
                    else:
                        log(f"Response body: {e.response.text}")
                except Exception:
                    log(
                        f"Response body: {e.response.text if e.response is not None else ''}"
                    )

                if apps_attempt < max_apps_attempts:
                    log(
                        f"Retrying MCP configuration in {apps_retry_delay:.1f}s (readiness timeout can be transient on cold start)..."
                    )
                    time.sleep(apps_retry_delay)
                    continue

                sys.exit(INFRA_FAILURE_EXIT_CODE)

            log(f"ERROR: Failed configuring MCP servers (status={status}): {e}")
            sys.exit(INFRA_FAILURE_EXIT_CODE)
        except Exception as e:
            log(f"ERROR: Unexpected MCP configuration failure: {e}")
            sys.exit(INFRA_FAILURE_EXIT_CODE)
    log("MCP servers configured")

    # Generate initial messages from HuggingFace task prompt
    # System prompt from agents/runner/agents/react_toolbelt_agent/README.md
    system_prompt = """You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

**Always Available (Meta-Tools):**
- `todo_write` - Task planning: create/update todos. Takes `todos` array [{id, content, status}] and `merge` boolean.
- `toolbelt_list_tools` / `toolbelt_inspect_tool` / `toolbelt_add_tool` / `toolbelt_remove_tool` - Tool management
- `final_answer` - Submit your answer (status: completed/blocked/failed)

**Domain Tools:** Use `toolbelt_list_tools` to discover, then `toolbelt_add_tool` to add them.

## Workflow

1. Plan: Use `todo_write` to create todos for complex tasks
2. Discover: Use `toolbelt_list_tools` to find relevant tools
3. Execute: Work through todos, use `todo_write` with `merge=true` to update status
4. Complete: Call `final_answer` (all todos must be completed/cancelled first)

## Rules

- Update todo status with `todo_write`: set `in_progress` when starting, `completed` when done
- Show your work for calculations
- `final_answer` is rejected if todos are incomplete
"""
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["prompt"]},
    ]
    with open(output_dir / "initial_messages.json", "w") as f:
        json.dump(initial_messages, f, indent=2)

    # Load orchestrator config
    with open(EXAMPLE_DIR / "orchestrator_config.json") as f:
        orchestrator_config = json.load(f)

    trajectory_file = output_dir / "trajectory.json"

    # Run agent
    step += 1
    log_progress(task["task_id"], step, total_steps, "Running agent")
    log("Running agent...")
    agent_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "runner.main",
        "--trajectory-id",
        trajectory_id,
        "--initial-messages",
        str(output_dir / "initial_messages.json"),
        "--mcp-gateway-url",
        f"{ENV_URL}/mcp/",
        "--agent-config",
        str(EXAMPLE_DIR / "agent_config.json"),
        "--orchestrator-model",
        orchestrator_config["model"],
        "--output",
        str(trajectory_file),
    ]

    # Add extra args if present
    if orchestrator_config.get("extra_args"):
        extra_args_file = output_dir / "orchestrator_extra_args.json"
        with open(extra_args_file, "w") as f:
            json.dump(orchestrator_config["extra_args"], f)
        agent_cmd.extend(["--orchestrator-extra-args", str(extra_args_file)])

    result = subprocess.run(agent_cmd, cwd=AGENTS_DIR)
    if result.returncode != 0:
        log(f"WARNING: Agent exited with code {result.returncode}")

    agent_status = None
    if trajectory_file.exists():
        with open(trajectory_file) as f:
            trajectory = json.load(f)
            agent_status = trajectory.get("status")
            log(f"Agent status: {agent_status}")

    # Save final snapshot
    step += 1
    log_progress(task["task_id"], step, total_steps, "Saving final snapshot")
    log("Saving final snapshot...")
    with httpx.stream("POST", f"{ENV_URL}/data/snapshot") as resp:
        resp.raise_for_status()
        final_tar_gz = output_dir / "final_snapshot.tar.gz"
        with open(final_tar_gz, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)

    final_zip = tar_gz_to_zip(final_tar_gz)
    log(f"Saved: {final_zip}")

    # Run grading if agent completed
    step += 1
    log_progress(task["task_id"], step, total_steps, "Running grading (if applicable)")
    if agent_status != "completed":
        log(f"Skipping grading (agent status: {agent_status})")
    else:
        log("Running grading...")

        # Generate verifiers from HuggingFace rubric
        verifiers = [
            {
                "verifier_id": c["verifier_id"],
                "verifier_version": 1,
                "world_id": world_id,
                "task_id": task["task_id"],
                "eval_config_id": "ec_output_llm",
                "verifier_values": {
                    "criteria": c["criteria"],
                    "is_primary_objective": i == 0,
                },
                "verifier_index": i,
                "verifier_dependencies": None,
            }
            for i, c in enumerate(task.get("rubric", []))
        ]
        with open(output_dir / "verifiers.json", "w") as f:
            json.dump(verifiers, f, indent=2)

        grades_file = output_dir / "grades.json"

        grading_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "runner.main",
            "--grading-run-id",
            grading_run_id,
            "--trajectory-id",
            trajectory_id,
            "--initial-snapshot",
            str(world_zip),
            "--final-snapshot",
            str(final_zip),
            "--trajectory",
            str(trajectory_file),
            "--grading-settings",
            str(EXAMPLE_DIR / "grading_settings.json"),
            "--verifiers",
            str(output_dir / "verifiers.json"),
            "--eval-configs",
            str(EXAMPLE_DIR / "eval_configs.json"),
            "--scoring-config",
            str(EXAMPLE_DIR / "scoring_config.json"),
            "--output",
            str(grades_file),
        ]

        result = subprocess.run(grading_cmd, cwd=GRADING_DIR)
        if result.returncode != 0:
            log(f"WARNING: Grading exited with code {result.returncode}")

        if grades_file.exists():
            with open(grades_file) as f:
                grades = json.load(f)
            log("=" * 60)
            log("GRADING RESULTS")
            log("=" * 60)
            log(f"Status: {grades.get('grading_run_status')}")
            log(f"Final Score: {grades.get('scoring_results', {}).get('final_score')}")
            for vr in grades.get("verifier_results", []):
                log(f"  - {vr.get('verifier_id')}: {vr.get('score')}")

    log("=" * 60)
    log("DONE")
    log(f"Output: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
