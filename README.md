# vLLM Couroutines Package
![vLLM](https://img.shields.io/badge/vLLM-GPU_Inference-FF6F00?style=flat)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-lightgrey) 

An installable asynchronous chat client and dataset runner for a running vLLM server's OpenAI-compatible API. It accepts user-provided chat prompts, writes every response in input order, and records a run manifest. It does not start vLLM or score model answers.

## Requirements

- Python 3.10 or newer
- A running vLLM server with the requested model loaded
- An input dataset in JSON Lines (`.jsonl`) format

Install the package in the client environment; install and configure vLLM separately in the server environment.

```bash
python -m pip install .
```

For editable development installs, use `python -m pip install -e .` from this directory.

## Test Dataset format (not required)

Each non-empty line is one JSON object with a unique, non-empty string `id` and a non-empty list of text chat `messages`. Each message must have string `role` and `content` fields.

```jsonl
{"id":"item-001","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}]}
{"id":"item-002","messages":[{"role":"user","content":"What is the capital of Japan?"}]}
```

Create a JSONL file for any dataset by serializing one record per prompt. Keep the IDs stable so outputs can be joined reliably to source examples and labels. Blank lines are ignored. Duplicate IDs and malformed rows are rejected before requests are sent.

## Run

Follow these steps from the package directory (`code/vllm-async-vr` in this repository). The commands use `meta-llama/Llama-3.1-8B-Instruct` as an example; replace it with a model you can access, using the same model ID when starting vLLM and running the client.

1. Install the client package and create a small JSONL dataset:

```bash
python -m pip install -e .
mkdir -p data results
cat > data/test.jsonl <<'EOF'
{"id":"item-001","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}]}
{"id":"item-002","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of Japan?"}]}
EOF
```

Replace these example records with one JSON object per line from your own dataset. Each record needs a unique `id` and a `messages` list.

2. In terminal 1, start the vLLM server with the model to evaluate:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 8000
```

Wait for the server to report that it is ready.

3. In terminal 2, from the package directory, run the dataset:

```bash
vllm-async-vr \
    --input data/test.jsonl \
    --output results/run-001.jsonl \
    --base-url http://127.0.0.1:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-tokens 128 \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --concurrency 8 \
    --num-chunks 30
```

The command prints the paths of the results and run manifest. Predictions are written to `results/run-001.jsonl`, one JSON record per input with its `id`, `messages`, and generated `response`. The manifest is written to `results/run-001.jsonl.manifest.json` and records the input SHA-256, model, endpoint, generation settings, runtime versions, and execution settings. For an authenticated server, set `OPENAI_API_KEY` in terminal 2 before running the command; the key is not saved in the manifest. Use `vllm-async-vr --help` to see every option.

`--timeout` defaults to 30 seconds, `--max-retries` defaults to 5, `--concurrency` defaults to 8, and `--num-chunks` defaults to 30; these execution settings are recorded in the manifest. The runner uses one batch when the dataset has at most `--num-chunks` records, and splits into non-empty batches only when it has more. Requests run concurrently within each batch, subject to the concurrency limit, and batches run in order. A fixed seed helps reproducibility but does not guarantee identical output across model/server versions, hardware, or kernels.

## Python API

```python
import asyncio

from vllm_async_vr import VLLMChatClient


async def main() -> None:
    messages = [{"role": "user", "content": "Summarize this text: ..."}]
    async with VLLMChatClient(
        base_url="http://127.0.0.1:8000/v1",
        model="your-served-model-name",
        max_concurrency=4,
    ) as client:
        response = await client.complete(
            messages,
            max_tokens=128,
            temperature=0,
            top_p=1,
            seed=42,
        )
        print(response)


asyncio.run(main())
```

`complete_many()` accepts a sequence of message lists and returns responses in matching order, bounded by `max_concurrency`. Its `num_chunks` argument (default 30) is only used when the prompt count is greater than that value. The async context manager closes the HTTP client. API failures are raised to the caller.

## Reproducibility notes

Keep the input dataset and generated manifest with the output. The manifest hashes the exact input file bytes, so even changes to whitespace produce a different hash. Record the vLLM version with `--server-version`; also preserve model revision, server launch arguments, hardware, and any custom chat template or server-side configuration externally. The package records the requested model name and decoding parameters but cannot verify which model weights or server configuration the endpoint actually loaded. It does not calculate accuracy or other evaluation metrics.
