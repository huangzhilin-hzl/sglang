#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class GrammarCase:
    name: str
    key_type: str
    key_string: str
    description: str


@dataclass
class CaseResult:
    name: str
    key_type: str
    payload_bytes: int
    cold_ms: list[float]
    cache_hit_ms: list[float]
    errors: list[str]


def simple_object_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
        "additionalProperties": False,
    }


def nested_object_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 80},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "assignees": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "email": {"type": "string", "format": "email"},
                        "role": {"type": "string", "enum": ["owner", "reviewer"]},
                    },
                    "required": ["id", "email", "role"],
                    "additionalProperties": False,
                },
            },
            "metadata": {
                "type": "object",
                "additionalProperties": {"type": ["string", "number", "boolean"]},
            },
        },
        "required": ["title", "priority", "assignees"],
        "additionalProperties": False,
    }


def union_object_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "event": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["click"]},
                            "target": {"type": "string"},
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                        },
                        "required": ["type", "target", "x", "y"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["navigate"]},
                            "url": {"type": "string"},
                            "method": {"type": "string", "enum": ["push", "replace"]},
                        },
                        "required": ["type", "url"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["submit"]},
                            "form_id": {"type": "string"},
                            "valid": {"type": "boolean"},
                        },
                        "required": ["type", "form_id", "valid"],
                        "additionalProperties": False,
                    },
                ]
            },
            "timestamp": {"type": "string"},
        },
        "required": ["event", "timestamp"],
        "additionalProperties": False,
    }


def large_enum_schema(enum_count: int = 256) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [f"action_{i:03d}" for i in range(enum_count)]},
            "reason": {"type": "string", "maxLength": 128},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["action", "confidence"],
        "additionalProperties": False,
    }


def tool_parameters_schema(index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,64}$"},
            "name": {"type": "string", "minLength": 1, "maxLength": 128},
            "mode": {"type": "string", "enum": ["create", "update", "delete"]},
            "enabled": {"type": "boolean"},
            "rank": {"type": "integer", "minimum": 0, "maximum": 10000},
            "labels": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "maxLength": 32},
            },
            "config": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "visible": {"type": "boolean"},
                    "variant": {"type": "string", "enum": [f"v{index}", "default"]},
                },
                "additionalProperties": False,
            },
        },
        "required": ["node_id", "name"],
        "additionalProperties": False,
    }


def tool_choice_required_json_schema(tool_count: int) -> dict[str, Any]:
    any_of = []
    for i in range(tool_count):
        name = f"tool_{i:02d}"
        any_of.append(
            {
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "parameters": tool_parameters_schema(i),
                },
                "required": ["name", "parameters"],
            }
        )

    return {
        "type": "array",
        "minItems": 1,
        "items": {"type": "object", "anyOf": any_of},
    }


def deepseek_v4_structural_tag(tool_count: int) -> dict[str, Any]:
    tags = []
    for i in range(tool_count):
        name = f"tool_{i:02d}"
        tags.append(
            {
                "type": "tag",
                "begin": f'<\uff5cDSML\uff5cinvoke name="{name}">\\n',
                "content": {
                    "type": "json_schema",
                    "json_schema": tool_parameters_schema(i),
                    "style": "deepseek_xml",
                },
                "end": "</\uff5cDSML\uff5cinvoke>\\n",
            }
        )

    return {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "const_string",
                    "value": "\\n\\n<\uff5cDSML\uff5ctool_calls>\\n",
                },
                {
                    "type": "tags_with_separator",
                    "tags": tags,
                    "separator": "\\n",
                    "at_least_one": True,
                    "stop_after_first": False,
                },
                {"type": "const_string", "value": "</\uff5cDSML\uff5ctool_calls>"},
            ],
        },
    }


def legacy_structural_tag_tools(tool_count: int) -> dict[str, Any]:
    structures = []
    triggers = []
    for i in range(tool_count):
        name = f"legacy_tool_{i:02d}"
        begin = f'<tool_call name="{name}">\\n'
        structures.append(
            {
                "begin": begin,
                "schema": tool_parameters_schema(i),
                "end": "\\n</tool_call>",
            }
        )
        triggers.append(begin)
    return {
        "type": "structural_tag",
        "structures": structures,
        "triggers": triggers,
        "at_least_one": True,
    }


def legacy_structural_tag() -> dict[str, Any]:
    return {
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<tool_call>\\n",
                "schema": simple_object_schema(),
                "end": "\\n</tool_call>",
            }
        ],
        "triggers": ["<tool_call>"],
        "at_least_one": True,
    }


def regex_alternation(count: int = 64) -> str:
    choices = "|".join(f"cmd_{i:02d}" for i in range(count))
    return rf"^({choices}):[a-zA-Z0-9_-]{{1,32}}$"


def build_cases(tool_count: int) -> list[GrammarCase]:
    medium_tool_count = min(max(2, tool_count // 4), 8)
    return [
        GrammarCase(
            "json_object_openai",
            "json",
            '{"type":"object"}',
            'response_format={"type":"json_object"}',
        ),
        GrammarCase(
            "json_any_builtin",
            "json",
            "$$ANY$$",
            "builtin any JSON grammar",
        ),
        GrammarCase(
            "json_schema_simple",
            "json",
            json_dumps(simple_object_schema()),
            "small object JSON schema",
        ),
        GrammarCase(
            "json_schema_nested",
            "json",
            json_dumps(nested_object_schema()),
            "nested object and array JSON schema",
        ),
        GrammarCase(
            "json_schema_anyof_union",
            "json",
            json_dumps(union_object_schema()),
            "anyOf union JSON schema",
        ),
        GrammarCase(
            "json_schema_large_enum",
            "json",
            json_dumps(large_enum_schema()),
            "large enum JSON schema",
        ),
        GrammarCase(
            "tool_choice_required_json_schema_1",
            "json",
            json_dumps(tool_choice_required_json_schema(1)),
            "fallback schema for tool_choice=required with 1 tool",
        ),
        GrammarCase(
            f"tool_choice_required_json_schema_{medium_tool_count}",
            "json",
            json_dumps(tool_choice_required_json_schema(medium_tool_count)),
            f"fallback schema for tool_choice=required with {medium_tool_count} tools",
        ),
        GrammarCase(
            f"tool_choice_required_json_schema_{tool_count}",
            "json",
            json_dumps(tool_choice_required_json_schema(tool_count)),
            f"fallback schema for tool_choice=required with {tool_count} tools",
        ),
        GrammarCase(
            "regex_email",
            "regex",
            r"^user@example\.com$",
            "simple regex",
        ),
        GrammarCase(
            "regex_phone",
            "regex",
            r"^\(\d{3}\) \d{3}-\d{4}$",
            "phone regex",
        ),
        GrammarCase(
            "regex_date",
            "regex",
            r"^2024-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$",
            "date regex",
        ),
        GrammarCase(
            "regex_large_alternation",
            "regex",
            regex_alternation(),
            "large alternation regex",
        ),
        GrammarCase(
            "regex_complex_json",
            "regex",
            r'^\{\s*"name"\s*:\s*"[a-zA-Z0-9 ]+"\s*,\s*"age"\s*:\s*[1-9][0-9]*\s*,\s*"city"\s*:\s*"[a-zA-Z0-9 ]+"\s*\}$',
            "JSON-looking regex",
        ),
        GrammarCase(
            "ebnf_literal",
            "ebnf",
            'root ::= "hello"',
            "literal EBNF",
        ),
        GrammarCase(
            "ebnf_greeting_choice",
            "ebnf",
            'root ::= "Hello" | "Hi" | "Hey"',
            "choice EBNF",
        ),
        GrammarCase(
            "ebnf_phone",
            "ebnf",
            """
root ::= "(" area ")" " " prefix "-" line
area ::= [0-9] [0-9] [0-9]
prefix ::= [0-9] [0-9] [0-9]
line ::= [0-9] [0-9] [0-9] [0-9]
""".strip(),
            "phone EBNF",
        ),
        GrammarCase(
            "ebnf_jsonish",
            "ebnf",
            """
root ::= object
object ::= "{" ws pair (ws "," ws pair)* ws "}"
pair ::= "\\"name\\"" ws ":" ws value |
         "\\"age\\"" ws ":" ws number |
         "\\"city\\"" ws ":" ws string
value ::= string | number
string ::= "\\"" [a-zA-Z0-9 ]+ "\\""
number ::= [1-9] [0-9]*
ws ::= [ ]*
""".strip(),
            "JSON-like EBNF",
        ),
        GrammarCase(
            "ebnf_tool_call_optional_params",
            "ebnf",
            """
root ::= function_call
function_call ::= "{" "\\"name\\"" ":" "\\"config_service\\"" ", " "\\"arguments\\"" ":" arguments "}"
arguments ::= "{" ( "\\"theme\\"" ":" ("\\"light\\"" | "\\"dark\\"") ( "," "\\"language\\"" ":" ("\\"en\\"" | "\\"es\\"" | "\\"fr\\"") )? ( "," "\\"notifications\\"" ":" ("true" | "false") )? | "\\"language\\"" ":" ("\\"en\\"" | "\\"es\\"" | "\\"fr\\"") ( "," "\\"notifications\\"" ":" ("true" | "false") )? | "\\"notifications\\"" ":" ("true" | "false") )? "}"
""".strip(),
            "tool-call-like EBNF with optional parameters",
        ),
        GrammarCase(
            "structural_tag_legacy",
            "structural_tag",
            json_dumps(legacy_structural_tag()),
            "legacy structural tag",
        ),
        GrammarCase(
            f"structural_tag_legacy_tools_{medium_tool_count}",
            "structural_tag",
            json_dumps(legacy_structural_tag_tools(medium_tool_count)),
            f"legacy structural tag with {medium_tool_count} tools",
        ),
        GrammarCase(
            "structural_tag_deepseek_v4_tools_1",
            "structural_tag",
            json_dumps(deepseek_v4_structural_tag(1)),
            "DeepSeek V4 style tool structural tag with 1 tool",
        ),
        GrammarCase(
            f"structural_tag_deepseek_v4_tools_{medium_tool_count}",
            "structural_tag",
            json_dumps(deepseek_v4_structural_tag(medium_tool_count)),
            f"DeepSeek V4 style tool structural tag with {medium_tool_count} tools",
        ),
        GrammarCase(
            f"structural_tag_deepseek_v4_tools_{tool_count}",
            "structural_tag",
            json_dumps(deepseek_v4_structural_tag(tool_count)),
            f"DeepSeek V4 style tool structural tag with {tool_count} tools",
        ),
    ]


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def get_eos_token_ids(tokenizer) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, list):
        return {int(x) for x in eos_token_id if x is not None}
    return {int(eos_token_id)}


def get_vocab_size(tokenizer, tokenizer_path: str, trust_remote_code: bool) -> int:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            tokenizer_path, trust_remote_code=trust_remote_code
        )
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
    except Exception:
        pass

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    return len(tokenizer)


def create_backend(args: argparse.Namespace):
    from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
    from sglang.srt.utils.hf_transformers import get_tokenizer

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = get_tokenizer(
        tokenizer_path,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        tokenizer_revision=args.revision,
        tokenizer_backend=args.tokenizer_backend,
    )
    vocab_size = get_vocab_size(tokenizer, tokenizer_path, args.trust_remote_code)
    eos_token_ids = get_eos_token_ids(tokenizer)

    server_args = SimpleNamespace(
        grammar_backend=args.grammar_backend,
        constrained_json_whitespace_pattern=args.constrained_json_whitespace_pattern,
        constrained_json_disable_any_whitespace=(
            args.constrained_json_disable_any_whitespace
        ),
        enable_strict_thinking=False,
        reasoning_parser=None,
    )
    backend = create_grammar_backend(
        server_args,
        tokenizer,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
        think_end_id=None,
    )
    if backend is None:
        raise RuntimeError(f"Grammar backend {args.grammar_backend!r} is not available")
    return backend


def compile_once(backend, case: GrammarCase, require_reasoning: bool) -> tuple[float, bool]:
    key = (case.key_type, case.key_string)
    start = time.perf_counter()
    value, cache_hit = backend.get_cached_or_future_value(key, require_reasoning)
    if not cache_hit:
        value = value.result()
        if value is not None and hasattr(value, "copy"):
            backend.set_cache(key, value.copy())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, cache_hit


def run_case(
    backend,
    case: GrammarCase,
    repeat: int,
    warmup: int,
    require_reasoning: bool,
) -> CaseResult:
    result = CaseResult(
        name=case.name,
        key_type=case.key_type,
        payload_bytes=len(case.key_string.encode("utf-8")),
        cold_ms=[],
        cache_hit_ms=[],
        errors=[],
    )

    for _ in range(warmup):
        backend.reset()
        try:
            compile_once(backend, case, require_reasoning)
        except Exception:
            pass

    for _ in range(repeat):
        backend.reset()
        try:
            cold_ms, cold_hit = compile_once(backend, case, require_reasoning)
            if cold_hit:
                result.errors.append("unexpected cache hit during cold compile")
            result.cold_ms.append(cold_ms)

            hit_ms, cache_hit = compile_once(backend, case, require_reasoning)
            if not cache_hit:
                result.errors.append("unexpected cache miss during cache-hit pass")
            result.cache_hit_ms.append(hit_ms)
        except Exception as e:
            result.errors.append(f"{type(e).__name__}: {e}")

    return result


def print_results(results: list[CaseResult]) -> None:
    headers = [
        "case",
        "kind",
        "bytes",
        "cold_avg",
        "cold_p50",
        "cold_p95",
        "cold_min",
        "cold_max",
        "hit_avg",
        "status",
    ]
    rows = []
    for result in results:
        cold = result.cold_ms
        hit = result.cache_hit_ms
        status = "ok" if not result.errors else result.errors[0]
        rows.append(
            [
                result.name,
                result.key_type,
                str(result.payload_bytes),
                fmt_ms(statistics.mean(cold) if cold else None),
                fmt_ms(percentile(cold, 0.50)),
                fmt_ms(percentile(cold, 0.95)),
                fmt_ms(min(cold) if cold else None),
                fmt_ms(max(cold) if cold else None),
                fmt_ms(statistics.mean(hit) if hit else None),
                status,
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang grammar compilation latency without starting a server."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "SGLANG_BENCH_MODEL", "meta-llama/Llama-3.2-1B-Instruct"
        ),
        help="Model path used as tokenizer fallback.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get("SGLANG_BENCH_TOKENIZER"),
        help="Tokenizer path. Defaults to --model-path.",
    )
    parser.add_argument(
        "--grammar-backend",
        default="xgrammar",
        choices=["xgrammar", "llguidance", "outlines"],
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tool-count", type=int, default=32)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only cases whose name contains this substring. Can be repeated.",
    )
    parser.add_argument("--require-reasoning", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer-mode", default="auto", choices=["auto", "slow"])
    parser.add_argument("--tokenizer-backend", default="huggingface")
    parser.add_argument("--constrained-json-whitespace-pattern")
    parser.add_argument("--constrained-json-disable-any-whitespace", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = build_cases(args.tool_count)
    if args.case:
        cases = [
            case
            for case in cases
            if any(pattern in case.name for pattern in args.case)
        ]
    if not cases:
        raise SystemExit("No cases matched --case filters")
    if args.list_cases:
        for case in cases:
            print(
                f"{case.name}\t{case.key_type}\t"
                f"{len(case.key_string.encode('utf-8'))} bytes\t{case.description}"
            )
        return 0

    backend = create_backend(args)
    print(
        f"backend={args.grammar_backend} tokenizer={args.tokenizer_path or args.model_path} "
        f"repeat={args.repeat} warmup={args.warmup} tool_count={args.tool_count}",
        flush=True,
    )
    results = []
    for case in cases:
        print(f"running {case.name}: {case.description}", flush=True)
        results.append(
            run_case(
                backend,
                case,
                repeat=args.repeat,
                warmup=args.warmup,
                require_reasoning=args.require_reasoning,
            )
        )

    if args.json_output:
        print(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "key_type": r.key_type,
                        "payload_bytes": r.payload_bytes,
                        "cold_ms": r.cold_ms,
                        "cache_hit_ms": r.cache_hit_ms,
                        "errors": r.errors,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        print_results(results)

    backend.executor.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
