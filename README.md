# GORKHE Bridge

GORKHE Bridge is a Windows desktop agent that connects a local Ollama model directly to a live Unreal Engine MCP server.

```text
GORKHE Bridge
├── Ollama
│   └── Local model
└── Unreal MCP
    └── Unreal Engine
```

The bridge discovers MCP tools, converts their schemas into Ollama function tools, lets the model choose tools, executes calls against Unreal, and sends the results back to the model until the task is complete.

## Requirements

- Windows 10 or 11
- Python 3.10+
- Ollama running locally
- A tool-capable Ollama model
- Unreal Engine 5.8+ with Unreal MCP enabled
- Unreal MCP available at `http://127.0.0.1:8000/mcp`

The default model is `qwen3:8b`.

## Install

```powershell
git clone https://github.com/roshabbhandari/unreal_mcp_ollama.git
cd unreal_mcp_ollama
py -m pip install -r requirements.txt
```

## Run

Start Unreal Engine and start its MCP server from the Unreal console:

```text
ModelContextProtocol.StartServer
```

Make sure Ollama is running and the model exists:

```powershell
docker exec -it gorkhe-ollama ollama list
```

Start the bridge:

```powershell
py gorkhe_bridge.py
```

Default endpoints:

```text
Ollama:      http://127.0.0.1:11434
Unreal MCP:  http://127.0.0.1:8000/mcp
Model:       qwen3:8b
```

All three values can be changed in the application.

## Agent loop

1. Connect to Unreal MCP over Streamable HTTP.
2. Discover the available tools.
3. Check the selected Ollama model.
4. Send the request and tool schemas to Ollama.
5. Execute any requested MCP tool calls.
6. Return tool results to Ollama.
7. Continue until a final answer is produced or the step limit is reached.

## Example

```text
Inspect the current Unreal level and tell me its name.
```

For an editor task:

```text
Create one cube at the origin and save the current level.
```

The bridge will show the tool name, arguments, results, and final model response in the desktop window.

## Architecture

```text
Windows
  │
  ├── GORKHE Bridge
  │      ├── PySide6 UI
  │      ├── Ollama client
  │      └── MCP client
  │
  ├── Ollama
  │      └── qwen3:8b
  │
  └── Unreal Engine
         └── Unreal MCP :8000
```

## Troubleshooting

Check Ollama:

```powershell
curl http://127.0.0.1:11434/api/tags
```

Check the Unreal MCP endpoint is reachable from Windows, then use the bridge's `Test connections` button.

If the bridge reports that the model is missing, refresh the model list or install the model in Ollama.

Keep Unreal Editor open with its MCP server running when executing Unreal actions.

## Security

The bridge does not provide arbitrary shell access. It only executes tools exposed by the configured MCP server.

Review an MCP server's tools before connecting it to a model that can call tools automatically.

## License

MIT
