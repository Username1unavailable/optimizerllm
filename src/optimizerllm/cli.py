"""
CLI entry point — optimizerllm

Commands:
  optimize   Optimize a prompt from stdin or file and print results
  inspect    Show block-by-block breakdown of a prompt before/after
  presets    List all available presets

Usage examples:
  echo '{"messages":[{"role":"user","content":"Hello"}]}' | optimizerllm optimize
  optimizerllm optimize --file prompt.json --preset rag --provider openai
  optimizerllm inspect --file prompt.json
  optimizerllm presets
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _load_input(path: str | None) -> dict[str, Any]:
    """Read JSON from a file or stdin."""
    if path:
        with open(path) as f:
            return json.load(f)
    if not sys.stdin.isatty():
        return json.load(sys.stdin)
    print("Error: provide --file or pipe JSON to stdin.", file=sys.stderr)
    sys.exit(1)


def _build_optimizer(args: argparse.Namespace) -> Any:
    from optimizerllm.presets import preset
    from optimizerllm.optimizer import ContextOptimizer
    from optimizerllm.core.block import CompressionMode

    if args.preset:
        return preset(
            args.preset,
            provider=args.provider,
            model=args.model,
            max_input_tokens=args.max_tokens,
            output_format=args.format,
        )

    return ContextOptimizer(
        provider=args.provider,
        model=args.model,
        max_input_tokens=args.max_tokens,
        output_budget=args.output_budget,
        mode=CompressionMode(args.mode),
        output_format=args.format,
    )


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

def cmd_optimize(args: argparse.Namespace) -> None:
    data = _load_input(args.file)
    opt = _build_optimizer(args)

    result = opt.optimize(
        messages=data.get("messages"),
        system=data.get("system"),
        tools=data.get("tools"),
        documents=data.get("documents"),
        examples=data.get("examples"),
        history=data.get("history"),
        structured=data.get("structured"),
        query=data.get("query"),
    )

    t = result.telemetry

    if args.json:
        output = {
            "telemetry": t.as_dict(),
            "rendered": result.rendered,
        }
        print(json.dumps(output, indent=2))
        return

    # Human-readable
    print(f"\n{'─'*60}")
    print(f"  optimizerllm  run {t.run_id[:8]}")
    print(f"{'─'*60}")
    print(f"  provider   : {t.provider} / {t.model}")
    print(f"  mode       : {t.compression_mode}")
    print(f"  tokens     : {t.tokens_before:,} → {t.tokens_after:,}  ({t.reduction_pct:.1f}% reduction)")
    print(f"  blocks     : {t.blocks_before} → {t.blocks_after}  ({len(t.blocks_removed)} removed)")
    print(f"  cache pfx  : {t.cacheable_prefix_tokens:,} tokens")
    print(f"  risk       : {t.risk_level}")
    if t.cost_before_usd is not None:
        saved = (t.cost_before_usd or 0) - (t.cost_after_usd or 0)
        print(f"  cost       : ${t.cost_before_usd:.4f} → ${t.cost_after_usd:.4f}  (save ${saved:.4f})")

    if t.blocks_removed:
        print(f"\n  removed blocks:")
        for r in t.blocks_removed:
            print(f"    ✗ [{r.block_type}]  {r.reason}  ({r.token_count} tok)")

    if args.show_output:
        print(f"\n  rendered output:")
        rendered = result.rendered
        if isinstance(rendered, list):
            for msg in rendered:
                print(f"    [{msg['role']}] {str(msg['content'])[:120]}")
        elif isinstance(rendered, dict):
            if "system" in rendered:
                print(f"    [system] {rendered['system'][:120]}")
            for msg in rendered.get("messages", []):
                print(f"    [{msg['role']}] {str(msg['content'])[:120]}")
        else:
            print(f"    {str(rendered)[:300]}")

    print()


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> None:
    """Show a block-by-block breakdown before and after optimization."""
    from optimizerllm.ingestion.ingester import Ingester
    from optimizerllm.normalization.normalizer import Normalizer
    from optimizerllm.token_accounting.counter import TokenCounter

    data = _load_input(args.file)
    opt = _build_optimizer(args)

    # Run full optimization to get before/after
    result = opt.optimize(
        messages=data.get("messages"),
        system=data.get("system"),
        tools=data.get("tools"),
        documents=data.get("documents"),
        history=data.get("history"),
        query=data.get("query"),
    )

    print(f"\n{'─'*60}")
    print(f"  BEFORE  ({result.telemetry.blocks_before} blocks, {result.telemetry.tokens_before:,} tokens)")
    print(f"{'─'*60}")

    # Re-ingest raw to show original blocks
    ingester = Ingester()
    raw_blocks = ingester.ingest(
        messages=data.get("messages"),
        system=data.get("system"),
        tools=data.get("tools"),
        documents=data.get("documents"),
        history=data.get("history"),
    )
    counter = TokenCounter()
    raw_blocks = counter.count_blocks(raw_blocks, provider=args.provider, model=args.model)
    for b in raw_blocks:
        preview = b.content[:70].replace("\n", " ")
        print(f"  {b.token_count:>5} tok  [{b.block_type.value:22s}]  {preview}")

    print(f"\n{'─'*60}")
    print(f"  AFTER   ({result.telemetry.blocks_after} blocks, {result.telemetry.tokens_after:,} tokens)")
    print(f"{'─'*60}")
    for b in result.blocks:
        preview = b.content[:70].replace("\n", " ")
        score_str = f"  score={b.score:.2f}" if b.score is not None else ""
        print(f"  {b.token_count:>5} tok  [{b.block_type.value:22s}]{score_str}  {preview}")

    if result.telemetry.blocks_removed:
        print(f"\n{'─'*60}")
        print(f"  DROPPED ({len(result.telemetry.blocks_removed)} blocks)")
        print(f"{'─'*60}")
        for r in result.telemetry.blocks_removed:
            print(f"  {r.token_count:>5} tok  [{r.block_type:22s}]  {r.reason}")
    print()


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def cmd_presets(_args: argparse.Namespace) -> None:
    from optimizerllm.presets import list_presets

    presets = list_presets()
    print(f"\n{'─'*60}")
    print(f"  Available presets")
    print(f"{'─'*60}")
    for p in presets:
        print(f"  {p['name']:20s}  [{p['mode']:13s}]  {p['description']}")
    print()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimizerllm",
        description="Deterministic context optimization engine for LLM prompts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--file", "-f", default=None, help="Input JSON file (default: stdin)")
    shared.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"])
    shared.add_argument("--model", default="gpt-4o")
    shared.add_argument("--max-tokens", type=int, default=128_000, dest="max_tokens")
    shared.add_argument("--output-budget", type=int, default=4_096, dest="output_budget")
    shared.add_argument("--mode", default="conservative", choices=["lossless", "conservative", "aggressive"])
    shared.add_argument("--preset", default=None, choices=["rag", "chat", "agent", "long_context_qa", "summarization"])
    shared.add_argument("--format", default="openai", choices=["openai", "anthropic", "text"])

    # optimize
    p_opt = sub.add_parser("optimize", parents=[shared], help="Optimize and summarise a prompt")
    p_opt.add_argument("--json", action="store_true", help="Output full JSON (telemetry + rendered)")
    p_opt.add_argument("--show-output", action="store_true", dest="show_output", help="Print rendered messages")
    p_opt.set_defaults(func=cmd_optimize)

    # inspect
    p_ins = sub.add_parser("inspect", parents=[shared], help="Block-by-block before/after breakdown")
    p_ins.set_defaults(func=cmd_inspect)

    # presets
    p_pre = sub.add_parser("presets", help="List available presets")
    p_pre.set_defaults(func=cmd_presets)

    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
