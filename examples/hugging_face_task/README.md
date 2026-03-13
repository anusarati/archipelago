# Hugging Face Task Example

Run tasks from the [mercor/apex-agents](https://huggingface.co/datasets/mercor/apex-agents) benchmark dataset, which contains 480 professional services tasks across investment banking, tax accounting, management consulting, and more.

## Task

The default task is an Investment Banking challenge from World 221. The prompt is:

> Calculate the accretion / dilution of both BBDC and TVPG shareholders, sensitized for different Cash consideration and Bid Premium.
>
> Edit the existing merger model and add two sensitivity analyses: one showing BBDC accretion/dilution and one showing TVPG accretion/dilution, each sensitized to bid premium (10% and 20%) and cash consideration (10% and 15%).
>
> Assume an increase of EBIT Synergies by 480bps and a 210bps decrease in post-deal bidder share price downside. All output values should be in %, rounded to 2 decimal places.




## Quick Start

```bash
cd archipelago/examples/hugging_face_task

# Set your LLM provider's API key
export GOOGLE_API_KEY=...      # or
export ANTHROPIC_API_KEY=...   # or
export OPENAI_API_KEY=...

./run.sh
```

The script will:
1. Download task data from HuggingFace
2. Start the environment container
3. Populate the environment with the world snapshot
4. Configure all MCP servers
5. Run the agent
6. Save the final snapshot
7. Run grading and display results

## Running Different Tasks

```bash
# Run default task (Investment Banking - BBDC/TVPG accretion/dilution)
./run.sh

# Run task at a specific index (0-479)
./run.sh 42

# Run task by ID
./run.sh task_9ba58a6197114140877a1df1754d2993

# Run all tasks for one world
./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

## Output

Results are saved to `output/<task_id>/`:

| File | Description |
|------|-------------|
| `trajectory.json` | Agent's conversation history and tool calls |
| `final_snapshot.zip` | Final state of the environment |
| `grades.json` | Grading results with scores and rationale |
| `initial_messages.json` | Task prompt (from HuggingFace) |
| `agent_config.json` | Agent configuration used |
| `verifiers.json` | Grading criteria (from HuggingFace rubric) |

## How It Works

Unlike `simple_task` which uses static pre-defined files, this example dynamically fetches everything from HuggingFace:

```
┌─────────────────────┐
│   HuggingFace       │
│   mercor/apex-agents│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  1. Download Task   │───▶│  2. Run Agent       │───▶│  3. Grade Results   │
│  - Task prompt      │    │  - All MCP servers  │    │  - Compare snapshots│
│  - World snapshot   │    │  - Execute task     │    │  - Evaluate rubric  │
│  - Rubric criteria  │    │  - Save trajectory  │    │  - Calculate score  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Configuration

### Changing the Model

Edit `orchestrator_config.json`:

```json
{
  "model": "anthropic/claude-sonnet-4-20250514"
}
```

### Using Fewer MCP Servers

By default, this runner loads `mcp_config_all_oss_servers.json` and then filters to the exact servers required by the world's `apps` field from HuggingFace metadata.

You can always override the config file:

```bash
MCP_CONFIG_FILE=mcp_config_all_oss_servers.json ./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

If MCP readiness is flaky on cold start, the runner retries `/apps` up to 3 times by default. You can tune:

```bash
MCP_CONFIGURE_MAX_ATTEMPTS=5 MCP_CONFIGURE_RETRY_DELAY_SECONDS=2 ./run.sh ...
```

Gateway-side readiness timeout defaults to 10 seconds. You can tune it in `environment/.env`:

```bash
MCP_READINESS_TIMEOUT_SECONDS=10
MCP_READINESS_RETRY_INTERVAL_SECONDS=1
```

### World Run Environment Reuse

World runs now reuse one environment container by default (`WORLD_REUSE_ENVIRONMENT=true`) and skip container restarts for each child task.

```bash
# default behavior (reuse one container for the whole world run)
./run.sh world_eec3883ca3c54c41a62d3f220a27736c

# force old behavior (restart container for each task)
WORLD_REUSE_ENVIRONMENT=false ./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

### Resume World Runs

World mode resumes by default (`WORLD_RESUME=true`) using `output/<world_id>/summary.json`:

- Terminal tasks (`agent_status` in `completed` or `failed`) are skipped
- Retryable tasks are re-run (for example `agent_status=error`, interruptions/cancellations, and infra startup failures)

```bash
# default: resume from existing summary
./run.sh world_eec3883ca3c54c41a62d3f220a27736c

# force a full rerun of all tasks
WORLD_RESUME=false ./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

### HSN Filesystem Mode

Enable Hierarchical Semantic Navigation (HSN) mode for the filesystem server:

```bash
USE_HSN_FILESYSTEM=true ./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

In HSN mode:

- The runner builds and caches per-world document embeddings and an HSN tree at `~/.cache/archipelago/hsn/<world_id>/<embedding_model>/hsn_index.json`
- Document extraction output is cached per world at `~/.cache/archipelago/hsn/<world_id>/_extraction/documents_with_text.json.gz`
- PDF extraction uses `pdf2image + pytesseract` OCR (MarkItDown is not used for PDFs)
- Extraction cache is invalidated automatically when the world snapshot hash, extraction settings (for example `HSN_MAX_EXTRACTED_TEXT_CHARS`), or MarkItDown version changes
- Cached HSN artifacts are reused across future runs for that same world
- Embedding batches are cached incrementally; if a batch fails, the runner does binary search to isolate failing inputs, caches successful vectors, then aborts
- A system message is injected before the user prompt, describing HSN semantics and listing top-level files with ID paths
- `filesystem_server.list_files` annotates file entries with HSN paths and accepts file paths (returns top-10 HSN children)
- `filesystem_server.search_files` also annotates matched files with HSN paths

Host prerequisites for PDF OCR:

- `tesseract` binary in `PATH`
- Poppler binary in `PATH` (`pdftoppm` or `pdftocairo`)

Embedding model is configurable:

```bash
USE_HSN_FILESYSTEM=true \
HSN_EMBEDDING_MODEL=openai/text-embedding-3-small \
HSN_EMBEDDING_BATCH_SIZE=32 \
./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

HSN embeddings use the `openai` Python client against an OpenAI-compatible endpoint. You can override auth + endpoint:

```bash
USE_HSN_FILESYSTEM=true \
HSN_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B \
HSN_EMBEDDING_API_KEY=sk-... \
HSN_EMBEDDING_BASE_URL=https://api.deepinfra.com/v1/openai \
./run.sh world_eec3883ca3c54c41a62d3f220a27736c
```

Optional extra embedding parameters can be passed as JSON:

```bash
HSN_EMBEDDING_EXTRA_ARGS='{"dimensions":1024}'
```

Optional embedding timeout (seconds):

```bash
HSN_EMBEDDING_TIMEOUT_SECONDS=180
```

Separate text limits for extraction vs embedding input:

```bash
HSN_MAX_EXTRACTED_TEXT_CHARS=131072
HSN_MAX_EMBEDDING_TEXT_CHARS=131072
```

Optional PDF OCR tuning:

```bash
HSN_PDF_OCR_DPI=150
HSN_PDF_OCR_MAX_PAGES=128
HSN_PDF_OCR_LANG=eng
HSN_PDF_OCR_CONFIG="--psm 6"
HSN_PDF_OCR_PAGE_TIMEOUT_SECONDS=20
HSN_PDF_OCR_THREAD_COUNT=2
```

## Available MCP Servers

| Server | Description |
|--------|-------------|
| `calendar_server` | Calendar and scheduling |
| `chat_server` | Chat/messaging |
| `code_execution_server` | Python code execution |
| `spreadsheets_server` | Spreadsheets/spreadsheet manipulation |
| `filesystem_server` | File operations |
| `mail_server` | Email |
| `pdfs_server` | PDF reading and manipulation |
| `presentations_server` | Presentations/slides |
| `documents_server` | Documents/document editing |

## Troubleshooting

### Task not found

The dataset contains 480 tasks indexed 0-479. Use `--task-index` for numeric indices or `--task-id` for specific task IDs.

### Environment fails to start

Check Docker is running and ports aren't in use:
```bash
docker ps
lsof -i :8080
```

### Agent timeout

For complex tasks, the agent may need more steps. Modify `max_steps` in `main.py`:
```python
agent_config = {
    "agent_config_values": {"timeout": 3600, "max_steps": 100},  # Increase from 50
    ...
}
```
