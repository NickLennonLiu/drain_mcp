# Drain MCP

Drain MCP is a FastMCP server that exposes [Drain3](https://github.com/logpai/Drain3)
log template mining as MCP tools. It can learn templates from individual log
lines or whole log files, run read-only inference against the trained model,
inspect mined clusters, and manage Drain masking rules at runtime.

The server uses file persistence by default, so trained clusters survive process
restarts in `drain3_state.bin`.

## Features

- Train a Drain3 model from a local UTF-8 log file or a directly reachable
  HTTP/HTTPS log file.
- Train incrementally from one log line at a time.
- Match log lines or files against the current model without changing state.
- Inspect model statistics and page through mined clusters.
- List, add, and remove masking rules used before template extraction.
- Run as a Streamable HTTP MCP server on `http://localhost:8101/mcp`.

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) or another Python environment manager

Project dependencies are declared in `pyproject.toml`:

- `drain3`
- `fastmcp`

## Quick Start

Install dependencies:

```bash
uv sync
```

Start the MCP server:

```bash
uv run python server.py
```

By default the server listens on:

```text
http://0.0.0.0:8101/mcp
```

For local MCP clients, use:

```text
http://localhost:8101/mcp
```

## MCP Client Configuration

Configure your MCP client to use Streamable HTTP with the server URL:

```json
{
  "mcpServers": {
    "drain-mcp": {
      "url": "http://localhost:8101/mcp"
    }
  }
}
```

If your client requires an explicit transport field, choose Streamable HTTP.

## Tools

### `train_file`

Train the Drain model from a local file path or HTTP/HTTPS URL.

Input:

- `file_url`: absolute local path or directly reachable URL

Returns the number of processed lines, cluster counts before and after training,
new cluster count, and message counts for clusters touched during the run.

### `train_line`

Train the model with one log line.

Input:

- `line`: a single log message without a newline

Returns the mined template, extracted parameters, cluster id, cluster size, and
whether a cluster was created or its template changed.

### `inference_line`

Match one log line against the trained model without updating model state.

Input:

- `line`: a single log message without a newline

Returns whether the line matched, the cluster id, template, parameters, and
cluster size.

### `inference_file`

Run read-only inference over a local file path or HTTP/HTTPS URL.

Input:

- `file_url`: absolute local path or directly reachable URL

Returns total line count, matched count, unmatched count, match rate, and the
first 100 per-line inference results.

### `model_stats`

Return aggregate model statistics:

- total cluster count
- total trained message count

### `list_clusters`

List mined clusters with pagination.

Inputs:

- `page`: 1-based page number, default `1`
- `page_size`: clusters per page, default `20`

Returns cluster id, size, and template for each cluster on the page.

### `list_masking`

List the currently configured Drain masking rules.

### `add_masking`

Add a runtime masking rule.

Inputs:

- `regex_pattern`: regular expression to mask before template mining
- `mask_with`: placeholder name to use for matches

The model is saved and reinitialized so the new rule applies to future training
and inference. Existing persisted clusters are preserved.

### `remove_masking`

Remove a runtime masking rule by exact regex pattern.

Input:

- `regex_pattern`: exact pattern to remove

The model is saved and reinitialized so the updated masking configuration
applies to future training and inference. Existing persisted clusters are
preserved.

## Persistence and Configuration

The server loads Drain configuration from:

```text
drain_demos/drain3.ini
```

The default configuration includes masking rules for common identifiers such as
IP addresses, hexadecimal values, numeric values, and command strings. It also
sets the Drain similarity threshold, tree depth, child limit, max cluster count,
and extra delimiters.

Model state is stored at:

```text
drain3_state.bin
```

This file is ignored by git because it is runtime state. Delete it if you want to
start with a fresh model.

Runtime masking changes are kept in memory for the current server process and
applied after the model is reinitialized. To make a masking rule permanent, add
it to `drain_demos/drain3.ini`.

## Demo Scripts

The `drain_demos` directory contains standalone Drain3 examples:

- `drain_stdin_demo.py`: interactive training and inference from standard input
- `drain_bigfile_demo.py`: batch processing demo using the public SSH log data
  set from Zenodo
- `drain3.ini`: shared Drain3 configuration

Run a demo with:

```bash
uv run python drain_demos/drain_stdin_demo.py
```

## Development

Check that the Python files compile successfully:

```bash
uv run python -m py_compile server.py main.py drain_demos/*.py
```

The repository currently has no automated test suite. For functional testing,
start the server and call the MCP tools from a compatible MCP client.
