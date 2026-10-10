"""Dataset-driven command-line runner for the async vLLM client."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import cast

from .client import ChatPrompt, VLLMChatClient


def _load_dataset(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            record_id = record.get("id")
            messages = record.get("messages")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"Line {line_number} must have a non-empty string 'id'")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate id {record_id!r} on line {line_number}")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Line {line_number} must have a non-empty 'messages' list")
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(
                        f"Each message on line {line_number} must have string 'role' and 'content'"
                    )

            seen_ids.add(record_id)
            records.append({"id": record_id, "messages": messages})

    if not records:
        raise ValueError("The input dataset contains no records")
    return records


def _package_version() -> str:
    try:
        return importlib.metadata.version("vllm-async-vr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


async def _run(args: argparse.Namespace, records: list[dict[str, object]]) -> list[str]:
    prompts = [cast(ChatPrompt, record["messages"]) for record in records]
    async with VLLMChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
    ) as client:
        return await client.complete_many(
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            num_chunks=args.num_chunks,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a JSONL chat dataset against a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input chat dataset in JSONL format")
    parser.add_argument("--output", required=True, type=Path, help="Output predictions JSONL path")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model name as exposed by the server")
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--server-version",
        default=os.getenv("VLLM_SERVER_VERSION"),
        help="Optional vLLM/server build identifier to record in the manifest",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.num_chunks < 1:
        parser.error("--num-chunks must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")

    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if args.input.resolve() in {args.output.resolve(), manifest_path.resolve()}:
        parser.error("input, output, and manifest paths must be different")
    if not args.overwrite and (args.output.exists() or manifest_path.exists()):
        parser.error("output already exists; choose another path or pass --overwrite")

    try:
        records = _load_dataset(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    started_at = datetime.now(timezone.utc)
    try:
        input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    except OSError as error:
        parser.error(f"Unable to read input dataset: {error}")

    try:
        responses = asyncio.run(_run(args, records))
    except Exception as error:
        parser.exit(1, f"Request failed: {error}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record, response in zip(records, responses, strict=True):
            output_record = {**record, "response": response}
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    manifest = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": args.input.name,
        "input_sha256": input_sha256,
        "record_count": len(records),
        "model": args.model,
        "base_url": args.base_url,
        "server_version": args.server_version,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "execution": {
            "concurrency": args.concurrency,
            "num_chunks": args.num_chunks,
            "timeout_seconds": args.timeout,
            "max_retries": args.max_retries,
        },
        "runtime": {
            "python": platform.python_version(),
            "package": _package_version(),
            "openai": importlib.metadata.version("openai"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(responses)} predictions to {args.output}")
    print(f"Wrote run manifest to {manifest_path}")
