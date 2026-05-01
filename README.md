# Retail AI Assistant 🛍️

A fun little side project that turned into a full‑on local AI agent.  
It's a Personal Shopper and a Customer Support bot rolled into one, running entirely on your machine with [Ollama](https://ollama.com/) and Python. No cloud, no APIs, no nonsense.

## What it does

You can ask it things like:

- *“I need a modest evening gown under $300 in size 8, on sale”*
- *“Can I return order O0001?”*
- *“Show me clearance items in size 8”*

The agent figures out what tool to use, pulls real data from CSV files, and applies the store’s return policy **in code** (not from an LLM’s memory). That means zero hallucination on returns—every decision is deterministic.

## Why I built it

I wanted to get my hands dirty with function‑calling, Ollama, and real‑world constraints like stock data, size availability, and a multi‑step return policy. The result is a small but robust agent that actually works under the hood, not just with fancy prompts.

## The fun parts

- 🔧 **Tools chosen by the model** – it decides when to call `search_products`, `get_order`, etc. on its own.
- 🧠 **Policy in Python, not in the LLM** – the `evaluate_return` function computes whether you can return something, the LLM just reads you the answer.
- 🛡️ **Anti‑hallucination tricks** – tags are checked against the real inventory, results are slimmed down, and the model is told to shut up about shoes.
- ⚡ **Streaming** – answers appear gradually, which feels way nicer than waiting for a big block of text.

## Running it

1. **Get Ollama** from [ollama.com](https://ollama.com) and pull the model:
   ```bash
   ollama pull qwen2.5:3b
   
2. Install Python requirements:

pip install openai pandas

3. Clone this repo:

python agent.py

Start chatting. Type exit to quit.
