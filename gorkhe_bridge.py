from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from mcp import Client
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
MAX_STEPS = 12


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def json_text(value: Any) -> str:
    try:
        return json.dumps(jsonable(value), ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def result_text(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
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
        elif isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
        else:
            parts.append(json_text(item))
    return "\n".join(parts) if parts else json_text(result)


@dataclass
class Config:
    ollama_url: str = DEFAULT_OLLAMA_URL
    mcp_url: str = DEFAULT_MCP_URL
    model: str = DEFAULT_MODEL
    thinking: bool = False
    max_steps: int = MAX_STEPS


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))

    async def models(self) -> list[str]:
        response = await self.http.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [
            item["name"]
            for item in response.json().get("models", [])
            if item.get("name")
        ]

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        thinking: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": item["input_schema"],
                    },
                }
                for item in tools
            ],
            "stream": False,
            "keep_alive": "10m",
        }
        if thinking:
            payload["think"] = True

        response = await self.http.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.http.aclose()


class McpClient:
    def __init__(self, url: str):
        self.url = url

    async def inspect(self) -> list[dict[str, Any]]:
        async with Client(self.url) as client:
            response = await client.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": jsonable(
                        getattr(
                            tool,
                            "inputSchema",
                            {"type": "object", "properties": {}},
                        )
                    ),
                }
                for tool in response.tools
            ]

    async def run_agent(
        self,
        config: Config,
        prompt: str,
        status: Callable[[str], None],
        output: Callable[[str], None],
        stopped: Callable[[], bool],
    ) -> None:
        ollama = OllamaClient(config.ollama_url)
        try:
            status("Connecting to Unreal MCP...")
            async with Client(config.mcp_url) as mcp:
                tools_response = await mcp.list_tools()
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": jsonable(
                            getattr(
                                tool,
                                "inputSchema",
                                {"type": "object", "properties": {}},
                            )
                        ),
                    }
                    for tool in tools_response.tools
                ]
                status(f"Unreal MCP connected — {len(tools)} tool(s).")

                status("Checking Ollama...")
                models = await ollama.models()
                if config.model not in models:
                    raise RuntimeError(
                        f"Model '{config.model}' is not installed. "
                        f"Available: {', '.join(models) or 'none'}"
                    )

                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": (
                            "You are GORKHE, an Unreal Engine development agent. "
                            "You have access to a live Unreal Engine MCP server. "
                            "Use MCP tools for Unreal actions and inspection. "
                            "Never claim an action succeeded without a tool result. "
                            "Prefer small, verifiable changes. "
                            "Do not create a task list unless asked."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                for step in range(1, config.max_steps + 1):
                    if stopped():
                        status("Stopped.")
                        return

                    status(f"Thinking… {step}/{config.max_steps}")
                    response = await ollama.chat(
                        config.model,
                        messages,
                        tools,
                        config.thinking,
                    )
                    message = response.get("message", {})
                    content = message.get("content") or ""
                    tool_calls = message.get("tool_calls") or []

                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": content,
                    }
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)

                    if content:
                        output(content)

                    if not tool_calls:
                        status("Finished.")
                        return

                    for call in tool_calls:
                        if stopped():
                            status("Stopped.")
                            return

                        function = call.get("function", {})
                        name = function.get("name")
                        arguments = function.get("arguments") or {}
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}

                        if not name:
                            continue

                        output(
                            "\n[Unreal MCP tool]\n"
                            f"{name}\n"
                            f"{json_text(arguments)}"
                        )
                        status(f"Calling {name}…")

                        try:
                            result = await mcp.call_tool(name, arguments=arguments)
                            text = result_text(result)
                            output(f"[Tool result]\n{text}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": text,
                                }
                            )
                        except Exception as exc:
                            error_text = f"{type(exc).__name__}: {exc}"
                            output(f"[Tool error]\n{error_text}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": error_text,
                                }
                            )

                status("Stopped after reaching the agent step limit.")
        finally:
            await ollama.close()


class ConnectionWorker(QThread):
    success = Signal(str)
    error = Signal(str)

    def __init__(self, ollama_url: str, mcp_url: str):
        super().__init__()
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url

    def run(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        ollama = OllamaClient(self.ollama_url)
        try:
            models = await ollama.models()
            tools = await McpClient(self.mcp_url).inspect()
            self.success.emit(
                "Ollama: connected\n"
                f"Models: {', '.join(models) or 'none'}\n"
                f"Unreal MCP: connected\n"
                f"Tools: {len(tools)}"
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
        asyncio.run(self._run())

    async def _run(self) -> None:
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

    def stopped(self) -> bool:
        return self.stop_requested

    def run(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        try:
            await McpClient(self.config.mcp_url).run_agent(
                self.config,
                self.prompt,
                self.status.emit,
                self.output.emit,
                self.stopped,
            )
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.done.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agent_worker: AgentWorker | None = None
        self.connection_worker: ConnectionWorker | None = None
        self.model_worker: ModelWorker | None = None
        self.setWindowTitle("GORKHE Bridge")
        self.resize(1120, 780)
        self.build_ui()
        self.refresh_models()

    def build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("GORKHE Bridge")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        subtitle = QLabel("Ollama + Unreal Engine MCP desktop agent")
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

        buttons = QHBoxLayout()
        self.test_button = QPushButton("Test connections")
        self.refresh_button = QPushButton("Refresh models")
        self.test_button.clicked.connect(self.test_connections)
        self.refresh_button.clicked.connect(self.refresh_models)
        buttons.addWidget(self.test_button)
        buttons.addWidget(self.refresh_button)

        form.addRow("Ollama API", self.ollama_url)
        form.addRow("Unreal MCP", self.mcp_url)
        form.addRow("Model", self.model_box)
        form.addRow("", buttons)
        layout.addWidget(connection_group)

        task_group = QGroupBox("Task")
        task_layout = QVBoxLayout(task_group)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "Example: Inspect the current Unreal level and tell me its name."
        )
        self.prompt.setMinimumHeight(120)
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

        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status"))
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)
        layout.addLayout(status_layout)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Connection and agent output appear here.")
        layout.addWidget(self.output, 1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QWidget {
                background: #101418;
                color: #eef2f5;
                font-family: "Segoe UI";
                font-size: 10.5pt;
            }
            QGroupBox {
                border: 1px solid #2b333b;
                border-radius: 8px;
                margin-top: 9px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QPlainTextEdit, QComboBox {
                background: #171d22;
                border: 1px solid #323c45;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton {
                background: #242c33;
                border: 1px solid #3a4650;
                border-radius: 6px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #303b44;
            }
            QPushButton:disabled {
                color: #69747d;
            }
            QLabel#subtitle {
                color: #9ca8b1;
            }
            """
        )

    def test_connections(self) -> None:
        if self.connection_worker and self.connection_worker.isRunning():
            return
        self.status_label.setText("Testing...")
        self.connection_worker = ConnectionWorker(
            self.ollama_url.text().strip(),
            self.mcp_url.text().strip(),
        )
        self.connection_worker.success.connect(self.connection_ok)
        self.connection_worker.error.connect(self.connection_error)
        self.connection_worker.start()

    def connection_ok(self, text: str) -> None:
        self.status_label.setText("Connected")
        self.output.appendPlainText(text)

    def connection_error(self, text: str) -> None:
        self.status_label.setText("Connection failed")
        self.output.appendPlainText(text)

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

    def run_agent(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing task", "Enter a task first.")
            return
        if self.agent_worker and self.agent_worker.isRunning():
            return

        config = Config(
            ollama_url=self.ollama_url.text().strip(),
            mcp_url=self.mcp_url.text().strip(),
            model=self.model_box.currentText().strip(),
            thinking=self.thinking.isChecked(),
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
        self.output.appendPlainText(f"\n[Error]\n{text}")
        self.status_label.setText("Error")

    def agent_done(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.status_label.text() not in {"Error", "Stopped."}:
            self.status_label.setText("Finished")

    def closeEvent(self, event) -> None:
        if self.agent_worker and self.agent_worker.isRunning():
            self.agent_worker.stop()
            self.agent_worker.wait(3000)
        if self.connection_worker and self.connection_worker.isRunning():
            self.connection_worker.wait(3000)
        if self.model_worker and self.model_worker.isRunning():
            self.model_worker.wait(3000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
