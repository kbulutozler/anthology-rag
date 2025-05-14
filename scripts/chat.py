#!/usr/bin/env python3
"""
Simple CLI for chatting with the NAACL-2025 research assistant.
"""

import requests
from rich.console import Console
from rich.text import Text

def main():
    console = Console()
    url = "http://localhost:9001/ask"

    welcome_message = Text("Chat with the NAACL-2025 Agent", style="bold magenta")
    console.print(welcome_message)
    console.print("Type your message or send an empty input to quit.", style="italic dim")
    console.print("-" * 50, style="dim")

    while True:
        try:
            question = console.input("[b bright_blue]You:[/b bright_blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not question.strip():
            break
        try:
            response = requests.post(url, json={"question": question})
            response.raise_for_status()
            answer = response.json().get("answer", "")
            console.print(f"[b green]Agent:[/b green] {answer}\n")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            break

if __name__ == "__main__":
    main() 