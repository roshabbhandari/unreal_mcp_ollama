from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import httpx
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_MAX_STEPS = 100


@dataclass
class Config:
    ollama_url: str = DEFAULT_OLLAMA_URL
    mcp_url: str = DEFAULT_MCP_URL
    model: str = DEFAULT_MODEL
    thinking: bool = False
    max_steps: int = DEFAULT_MAX_STEPS


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump())
        except Exception:
            pass
    return str(value)


def json_text(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2)


def tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = {"type": "object", "properties": {}}
    return {
        "name": str(tool.name),
        "description": str(getattr(tool, "description", "") or ""),
        "input_schema": jsonable(schema),
    }


def result_text(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json_text(structured)

    content = getattr(result, "content", None)
    if content is None:
        return json_text(result)

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        elif isinstance(item, dict) and item.get("text") is not None:
            parts.append(str(item["text"]))
        else:
            parts.append(json_text(item))
    return "\n".join(parts) if parts else json_text(result)


@asynccontextmanager
async def unreal_connection(url: str) -> AsyncIterator[Client]:
    async with streamable_http_client(url, terminate_on_close=False) as transport:
        async with Client(transport) as client:
            yield client


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def models(self) -> list[str]:
        response = await self.http.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [item["name"] for item in data.get("models", []) if item.get("name")]

    async def chat(self, config: Config, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": item["input_schema"],
                    },
                }
                for item in tools
            ]
        if config.thinking:
            payload["think"] = True
        response = await self.http.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.http.aclose()


class ConnectionWorker(QThread):
    success = Signal(str)
    error = Signal(str)

    def __init__(self, ollama_url: str, mcp_url: str):
        super().__init__()
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url

    def run(self) -> None:
        asyncio.run(self.work())

    async def work(self) -> None:
        ollama = OllamaClient(self.ollama_url)
        try:
            models = await ollama.models()
            async with unreal_connection(self.mcp_url) as mcp:
                tools = await mcp.list_tools()
                info = getattr(mcp, "server_info", None)
                name = getattr(info, "name", "Unreal MCP") if info else "Unreal MCP"
                protocol = getattr(mcp, "protocol_version", "unknown")
            self.success.emit(
                "\n".join(
                    [
                        "Ollama: connected",
                        f"Models: {', '.join(models) or 'none'}",
                        f"Unreal MCP: connected ({name})",
                        f"MCP protocol: {protocol}",
                        f"Tools discovered: {len(tools.tools)}",
                    ]
                )
            )
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            await ollama.close()


class ModelWorker(QThread):
    success = Signal(list)
    error = Signal(str)

    def __init__(self, ollama_url: str):
        super().__init__()
        self.ollama_url = ollama_url

    def run(self) -> None:
        asyncio.run(self.work())

    async def work(self) -> None:
        ollama = OllamaClient(self.ollama_url)
        try:
            self.success.emit(await ollama.models())
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            await ollama.close()


class AgentWorker(QThread):
    status = Signal(str)
    output = Signal(str)
    error = Signal(str)
    done = Signal()

    def __init__(self, config: Config, prompt: str):
        super().__init__()
        self.config = config
        self.prompt = prompt
        self.stop_requested = False

    def stop(self) -> None:
        self.stop_requested = True

    def run(self) -> None:
        try:
            asyncio.run(self.work())
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.done.emit()

    async def work(self) -> None:
        ollama = OllamaClient(self.config.ollama_url)
        try:
            self.status.emit("Connecting to Unreal MCP...")
            async with unreal_connection(self.config.mcp_url) as mcp:
                response = await mcp.list_tools()
                tools = [tool_schema(tool) for tool in response.tools]
                self.status.emit(f"Unreal MCP connected — {len(tools)} tool(s) available.")

                models = await ollama.models()
                if self.config.model not in models:
                    raise RuntimeError(
                        f"Model '{self.config.model}' is not installed. Available: {', '.join(models) or 'none'}"
                    )

                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": (
                            "You are GORKHE, an autonomous Unreal Engine game-development agent. "
                            "You have live access to Unreal Engine through MCP. When the user asks "
                            "you to inspect, create, modify, save, test, or otherwise act in Unreal, "
                            "use the available MCP tools. Do not stop at planning. Do not create a "
                            "task list unless explicitly asked. Never claim success without a tool "
                            "result. Inspect only when necessary, then act, verify, and continue."
                        ),
                    },
                    {"role": "user", "content": self.prompt},
                ]

                for step in range(1, max(1, self.config.max_steps) + 1):
                    if self.stop_requested:
                        self.status.emit("Stopped.")
                        return

                    self.status.emit(f"Thinking and acting — step {step}/{self.config.max_steps}")
                    result = await ollama.chat(self.config, messages, tools)
                    message = result.get("message", {}) or {}
                    content = str(message.get("content") or "")
                    tool_calls = message.get("tool_calls") or []

                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": content,
                    }
                    if message.get("thinking"):
                        assistant_message["thinking"] = message["thinking"]
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)

                    if content:
                        self.output.emit(content)

                    if not tool_calls:
                        self.status.emit("Finished.")
                        return

                    for call in tool_calls:
                        if self.stop_requested:
                            self.status.emit("Stopped.")
                            return

                        function = call.get("function", {}) or {}
                        name = function.get("name")
                        arguments = function.get("arguments") or {}
                        if not name:
                            self.output.emit("[Tool error]\nModel returned a tool call without a name.")
                            continue

                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError as exc:
                                self.output.emit(f"[Tool error]\nInvalid arguments for {name}: {exc}")
                                arguments = {}

                        self.output.emit(
                            "[Unreal MCP tool]\n"
                            f"{name}\n"
                            f"{json_text(arguments)}"
                        )
                        self.status.emit(f"Calling {name}...")

                        try:
                            tool_result = await mcp.call_tool(name, arguments=arguments)
                            text = result_text(tool_result)
                            if getattr(tool_result, "is_error", False) or getattr(tool_result, "isError", False):
                                text = f"TOOL_ERROR\n{text}"
                            self.output.emit(f"[Tool result]\n{text}")
                        except Exception as exc:
                            text = f"{type(exc).__name__}: {exc}"
                            self.output.emit(f"[Tool error]\n{text}")

                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "content": text,
                            }
                        )

                self.status.emit("Reached the maximum agent step limit.")
        finally:
            await ollama.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agent_worker: AgentWorker | None = None
        self.connection_worker: ConnectionWorker | None = None
        self.model_worker: ModelWorker | None = None
        self.setWindowTitle("GORKHE Bridge")
        self.resize(1100, 780)
        self.build_ui()
        self.refresh_models()

    def build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("GORKHE Bridge")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        subtitle = QLabel("Ollama + Unreal Engine MCP")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        connection_group = QGroupBox("Connections")
        form = QFormLayout(connection_group)
        self.ollama_url = QLineEdit(DEFAULT_OLLAMA_URL)
        self.mcp_url = QLineEdit(DEFAULT_MCP_URL)
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.setCurrentText(DEFAULT_MODEL)

        row = QHBoxLayout()
        self.test_button = QPushButton("Test connections")
        self.refresh_button = QPushButton("Refresh models")
        self.test_button.clicked.connect(self.test_connections)
        self.refresh_button.clicked.connect(self.refresh_models)
        row.addWidget(self.test_button)
        row.addWidget(self.refresh_button)

        form.addRow("Ollama API", self.ollama_url)
        form.addRow("Unreal MCP", self.mcp_url)
        form.addRow("Model", self.model_box)
        form.addRow("", row)
        layout.addWidget(connection_group)

        task_group = QGroupBox("Task")
        task_layout = QVBoxLayout(task_group)
        self.prompt = QPlainTextEdit()
        self.prompt.setMinimumHeight(130)
        self.prompt.setPlaceholderText(
            "Example: Create a ground, two houses, a light and a player start in the current Unreal level, then save it."
        )
        task_layout.addWidget(self.prompt)
        layout.addWidget(task_group)

        controls = QHBoxLayout()
        self.thinking = QCheckBox("Enable thinking")
        self.run_button = QPushButton("Run agent")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_agent)
        self.stop_button.clicked.connect(self.stop_agent)
        controls.addWidget(self.thinking)
        controls.addStretch(1)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.run_button)
        layout.addLayout(controls)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Ready")
        status_row.addWidget(self.status_label, 1)
        layout.addLayout(status_row)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, 1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QWidget { background: #101418; color: #eef2f5; font-family: \"Segoe UI\"; font-size: 10.5pt; }
            QGroupBox { border: 1px solid #2b333b; border-radius: 8px; margin-top: 9px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QPlainTextEdit, QComboBox { background: #171d22; border: 1px solid #323c45; border-radius: 6px; padding: 8px; }
            QPushButton { background: #242c33; border: 1px solid #3a4650; border-radius: 6px; padding: 8px 14px; }
            QPushButton:hover { background: #303a42; }
            QPushButton:disabled { color: #69747d; }
            QLabel#subtitle { color: #9ca8b1; }
            """
        )

    def refresh_models(self) -> None:
        if self.model_worker and self.model_worker.isRunning():
            return
        self.status_label.setText("Loading Ollama models...")
        self.model_worker = ModelWorker(self.ollama_url.text().strip())
        self.model_worker.success.connect(self.models_ok)
        self.model_worker.error.connect(self.models_error)
        self.model_worker.start()

    def models_ok(self, models: list[str]) -> None:
        current = self.model_box.currentText().strip() or DEFAULT_MODEL
        self.model_box.clear()
        self.model_box.addItems(models)
        if current in models:
            self.model_box.setCurrentText(current)
        elif DEFAULT_MODEL in models:
            self.model_box.setCurrentText(DEFAULT_MODEL)
        elif models:
            self.model_box.setCurrentIndex(0)
        else:
            self.model_box.setCurrentText(current)
        self.status_label.setText(f"{len(models)} Ollama model(s) found")

    def models_error(self, text: str) -> None:
        self.status_label.setText("Ollama unavailable")
        self.output.appendPlainText(f"[Model refresh]\n{text}")

    def test_connections(self) -> None:
        if self.connection_worker and self.connection_worker.isRunning():
            return
        self.status_label.setText("Testing connections...")
        self.connection_worker = ConnectionWorker(
            self.ollama_url.text().strip(),
            self.mcp_url.text().strip(),
        )
        self.connection_worker.success.connect(self.connection_ok)
        self.connection_worker.error.connect(self.connection_error)
        self.connection_worker.start()

    def connection_ok(self, text: str) -> None:
        self.status_label.setText("Connections ready")
        self.output.appendPlainText(text)

    def connection_error(self, text: str) -> None:
        self.status_label.setText("Connection failed")
        self.output.appendPlainText(f"[Connection error]\n{text}")

    def run_agent(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing task", "Enter a task first.")
            return
        if self.agent_worker and self.agent_worker.isRunning():
            return

        model = self.model_box.currentText().strip()
        if not model:
            QMessageBox.warning(self, "Missing model", "Select an Ollama model.")
            return

        config = Config(
            ollama_url=self.ollama_url.text().strip(),
            mcp_url=self.mcp_url.text().strip(),
            model=model,
            thinking=self.thinking.isChecked(),
            max_steps=DEFAULT_MAX_STEPS,
        )

        self.output.clear()
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.agent_worker = AgentWorker(config, prompt)
        self.agent_worker.status.connect(self.status_label.setText)
        self.agent_worker.output.connect(self.output.appendPlainText)
        self.agent_worker.error.connect(self.agent_error)
        self.agent_worker.done.connect(self.agent_done)
        self.agent_worker.start()

    def stop_agent(self) -> None:
        if self.agent_worker and self.agent_worker.isRunning():
            self.agent_worker.stop()
            self.status_label.setText("Stopping...")

    def agent_error(self, text: str) -> None:
        self.output.appendPlainText(f"[Agent error]\n{text}")
        self.status_label.setText("Error")

    def agent_done(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.status_label.text() not in {"Error", "Stopped."}:
            self.status_label.setText("Finished")

    def closeEvent(self, event) -> None:
        if self.agent_worker and self.agent_worker.isRunning():
            self.agent_worker.stop()
            self.agent_worker.wait(5000)
        if self.connection_worker and self.connection_worker.isRunning():
            self.connection_worker.wait(5000)
        if self.model_worker and self.model_worker.isRunning():
            self.model_worker.wait(5000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
