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
