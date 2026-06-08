from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


SYSTEM_PROMPT = """
You are a helpful assistant with tools.

Use the web search tool when the user asks for facts, current information,
people, companies, news, prices, docs, or anything that may need lookup.
Use the datetime tool when the user asks for the current date, time, today, or now.
Use the calculator for arithmetic.

After a tool call, answer clearly and briefly. Include sources when the web search
tool returns them. Do not invent facts when the tools fail.
""".strip()


@tool
def serper_web_search(query: str) -> str:
    """Search the web with Serper for current or factual information."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Serper search is not configured. Add SERPER_API_KEY to your .env file."

    payload = json.dumps({"q": query, "num": 5}).encode("utf-8")
    request = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return f"Search failed: {exc}"

    direct_answer = _extract_direct_answer(data)
    organic = data.get("organic", [])

    lines = []
    if direct_answer:
        lines.append(f"Direct answer: {direct_answer}")

    if organic:
        lines.append("Search results:")
        for index, item in enumerate(organic[:5], start=1):
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "No snippet available.")
            link = item.get("link", "")
            lines.append(f"{index}. {title}\n   {snippet}\n   Source: {link}")

    return "\n".join(lines) if lines else "No search results found."


@tool
def get_system_datetime() -> str:
    """Get the current system date and time."""
    now = datetime.now().astimezone()
    utc_now = datetime.now(timezone.utc)
    return (
        f"Local system date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"UTC date and time: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


@tool
def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression."""
    return calculate_safely(expression)


def build_graph():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")

    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    model = ChatOpenAI(model=model_name, temperature=0)
    tools = [serper_web_search, get_system_datetime, calculate]
    return create_react_agent(model, tools=tools, prompt=SYSTEM_PROMPT)


def run_agent(user_input: str) -> str:
    app = build_graph()
    state = app.invoke({"messages": [{"role": "user", "content": user_input}]})
    final_message = state["messages"][-1]
    return final_message.content


def calculate_safely(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_math(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"Could not calculate that expression: {exc}"

    return f"{expression} = {value}"


def _eval_math(node: ast.AST) -> Any:
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
        return binary_ops[type(node.op)](_eval_math(node.left), _eval_math(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
        return unary_ops[type(node.op)](_eval_math(node.operand))

    raise ValueError("only numbers and basic arithmetic operators are allowed")


def _extract_direct_answer(data: dict[str, Any]) -> str:
    answer_box = data.get("answerBox") or {}
    knowledge_graph = data.get("knowledgeGraph") or {}

    for key in ("answer", "snippet", "description"):
        value = answer_box.get(key) or knowledge_graph.get(key)
        if value:
            return str(value)

    title = knowledge_graph.get("title")
    description = knowledge_graph.get("description")
    if title and description:
        return f"{title}: {description}"

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI LangGraph agent.")
    parser.add_argument("query", nargs="+", help="Query to send to the agent")
    args = parser.parse_args()

    print(run_agent(" ".join(args.query)))


if __name__ == "__main__":
    main()
