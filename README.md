# Prompting Couroutines Package for Accelerated Inference
![vLLM](https://img.shields.io/badge/vLLM-GPU_Inference-FF6F00?style=flat)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-lightgrey) 

An installable asynchronous chat client and dataset runner for a running vLLM server's OpenAI-compatible API. It accepts user-provided chat prompts, writes every response in input order, and records a run manifest. It does not start vLLM or score model answers.

## Requirements

- Python 3.10 or newer
- A running vLLM server with the requested model loaded
- An input dataset in JSON Lines (`.jsonl`) format

## Environment setup

Set up the client environment from the package directory (`code/vllm-async-vr` in this repository):

```bash
conda create -n vllm-async-vr python=3.11 -y
conda activate vllm-async-vr
python -m pip install --upgrade pip
python -m pip install -e .
vllm-async-vr --help
```

The client environment installs the OpenAI client; it does not install vLLM or CUDA components.

Create a separate Conda environment on the GPU/server machine, then install vLLM into it using the official installation command for your operating system, Python version, CUDA/ROCm setup, and PyTorch requirements. These requirements vary by vLLM release, so use the matching [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

```bash
conda create -n vllm-server python=3.11 -y
conda activate vllm-server
# Install vLLM here using the command for your platform from its installation guide.
```

If the selected model requires Hugging Face authentication, configure `HF_TOKEN` securely in the server environment before launching vLLM.

## Dataset format

Each non-empty line is one JSON object with a unique, non-empty string `id`. You can provide ready-made `messages`, or keep your own dataset fields and use `--user-template` to turn them into a prompt. A message is an object with string `role` and `content` fields.

```jsonl
{"id":"item-001","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}]}
{"id":"item-002","messages":[{"role":"user","content":"What is the capital of Japan?"}]}
```

For example, an existing review dataset can keep its own schema:

```jsonl
{"id":"review-001","product":"Notebook","review":{"title":"Good paper","text":"The pages are smooth."},"rating":5}
{"id":"review-002","product":"Pen","review":{"title":"Leaks","text":"Ink leaked in my bag."},"rating":1}
```

Keep IDs stable so results can be joined to source examples and labels. Blank lines are ignored, and duplicate or missing IDs are rejected before requests are sent. The CLI currently reads JSONL; convert CSV or other source formats to one JSON object per line before running it.

## Run

Follow these steps from the package directory (`code/vllm-async-vr` in this repository). The commands use `meta-llama/Llama-3.1-8B-Instruct` as an example; replace it with a model you can access, using the same model ID when starting vLLM and running the client.

1. In the activated `vllm-async-vr` Conda environment, create a small JSONL dataset:

```bash
mkdir -p data results
cat > data/test.jsonl <<'EOF'
{"id":"item-001","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of France?"}]}
{"id":"item-002","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is the capital of Japan?"}]}
EOF
```

Replace these example records with one JSON object per line from your own dataset. Each record needs a unique `id`; you can either provide a `messages` list or use templates as shown below.

2. In terminal 1, activate the `vllm-server` Conda environment and start vLLM with the model to evaluate:

```bash
conda activate vllm-server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 8000
```

Wait for the server to report that it is ready.

3. In terminal 2, from the package directory, activate the client environment and run the dataset:

```bash
conda activate vllm-async-vr
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

### Make prompts from existing fields

For the review JSONL above, run the same command with templates instead of prebuilt `messages`:

```bash
vllm-async-vr \
    --input data/reviews.jsonl \
    --output results/reviews.jsonl \
    --base-url http://127.0.0.1:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-tokens 128 \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --user-template 'Product: {product}. Review title: {review[title]}. Review: {review[text]}. Rating: {rating}/5. Classify the sentiment.' \
    --system-template 'Classify the sentiment of the supplied review.'
```

Templates use Python format-string syntax: `{field}` selects a top-level field, `{review[text]}` selects a nested object field, and `{items[0]}` selects a list item. Structured values inserted as a whole are rendered as JSON. Literal braces are written as `{{` and `}}`. Missing or invalid template fields produce an error with the input line number before any requests are sent. Output records retain all original data fields and add the rendered `messages` and model `response`. The manifest records the templates used.

An optional assistant prefill can be added after the user message with `--assistant-template`. For example, add this flag to the command above to give the model an assistant-role prefix:

```bash
--assistant-template 'Sentiment:'
```

The option is omitted by default and requires `--user-template`. It is useful when the model should continue from an assistant prefix; check that the model's chat template supports assistant-prefill messages.

## Python API

`PromptSerializer` can also be used directly when loading data in your own Python pipeline:

```python
from vllm_async_vr import PromptSerializer

serializer = PromptSerializer(
    user_template="Review: {review[text]}. Rating: {rating}/5.",
    system_template="Classify the sentiment.",
    assistant_template="Sentiment:",  # Optional assistant-role prefill
)
messages = serializer.serialize(
    {"review": {"text": "The pages are smooth."}, "rating": 5}
)
```

The resulting `messages` can be passed to `VLLMChatClient.complete()` or `complete_many()` as shown below. Omit `assistant_template` when the model should generate directly after the user message.

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

