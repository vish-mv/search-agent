# Simple LangGraph Agent

A tiny OpenAI-powered LangGraph agent that can:

- decide when to call tools using an OpenAI chat model
- search the web with [Serper](https://serper.dev/)
- answer date/time questions from the system clock
- calculate simple arithmetic

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e .
```

For web search, create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
SERPER_API_KEY=your-serper-key
```

`OPENAI_API_KEY` is required. `SERPER_API_KEY` is only needed when the agent chooses web search.

## Run

CLI:

```bash
simple-agent "Who is the CEO of OpenAI?"
simple-agent "What date and time is it?"
simple-agent "calculate 12 * (4 + 3)"
```

You can also run it directly:

```bash
python3 -m simple_langgraph_agent.agent "search OpenAI Codex"
```

API server:

```bash
uvicorn simple_langgraph_agent.api:app --reload
```

Test the endpoints:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What date and time is it?"}'
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"Who is the CEO of OpenAI?"}'
```

Open the interactive API docs at:

```text
http://127.0.0.1:8000/docs
```

## How It Works

The graph is a LangGraph tool-calling agent:

1. OpenAI receives the user query and the tool descriptions.
2. The model decides whether to call `serper_web_search`, `get_system_datetime`, or `calculate`.
3. LangGraph executes the tool call and returns the tool result to the model.
4. The model writes the final answer.
