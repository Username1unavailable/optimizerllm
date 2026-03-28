"""
demo.py — real-world prompt scenarios for optimizerllm
Run: python demo.py
"""

from optimizerllm import ContextOptimizer, CompressionMode


# ── helpers ──────────────────────────────────────────────────────────────────

def header(title: str) -> None:
    print("\n" + "━" * 72)
    print(f"  {title}")
    print("━" * 72)


def show_result(result, show_messages: bool = True, max_content: int = 90) -> None:
    t = result.telemetry
    print(f"  tokens   : {t.tokens_before:>6,} → {t.tokens_after:>6,}  ({t.reduction_pct:.1f}% reduction)")
    print(f"  blocks   : {t.blocks_before:>6}  → {t.blocks_after:>6}  ({len(t.blocks_removed)} removed)")
    print(f"  cache pfx: {t.cacheable_prefix_tokens:>6,} tokens")
    print(f"  risk     : {t.risk_level}")
    if t.cost_before_usd is not None:
        saved = (t.cost_before_usd or 0) - (t.cost_after_usd or 0)
        print(f"  cost     : ${t.cost_before_usd:.4f} → ${t.cost_after_usd:.4f}  (save ${saved:.4f})")

    if t.blocks_removed:
        print(f"\n  dropped blocks:")
        for r in t.blocks_removed:
            score_str = f"score={r.score:.3f}" if r.score is not None else "           "
            print(f"    ✗ [{r.block_type:22s}] {score_str}  {r.reason}")

    if show_messages:
        rendered = result.rendered
        messages = rendered if isinstance(rendered, list) else rendered.get("messages", [])
        print(f"\n  final messages ({len(messages)}):")
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                text = content[0].get("text", "")
            else:
                text = content
            preview = text[:max_content].replace("\n", " ")
            ellipsis = "…" if len(text) > max_content else ""
            print(f"    [{msg['role']:9s}] {preview}{ellipsis}")


# ── scenario 1: RAG with noisy retrieval ─────────────────────────────────────

header("SCENARIO 1 · RAG — noisy retrieval, query-aware selection")
print("""
  Situation: a user asks about Python asyncio. The retrieval system
  returns 8 docs, but only 3 are actually relevant. The optimizer should
  keep the relevant ones and drop the noise.
""")

opt = ContextOptimizer(
    provider="openai", model="gpt-4o",
    max_input_tokens=8192, output_budget=1024,
    mode=CompressionMode.CONSERVATIVE,
    selector_threshold=0.15,
)

result = opt.optimize(
    system="You are a senior Python engineer. Answer with working code examples.",
    documents=[
        # Relevant — asyncio / concurrency
        "Python asyncio provides infrastructure for writing single-threaded concurrent code using coroutines. The event loop runs asynchronous tasks and callbacks.",
        "async def and await keywords are the foundation of Python asyncio. Use 'async def' to define a coroutine and 'await' to pause execution until a result is ready.",
        "asyncio.gather() runs multiple coroutines concurrently and returns their results as a list. It is the most common way to run tasks in parallel with asyncio.",
        # Noise — unrelated topics retrieved by broad Python keyword match
        "Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Use plt.plot() for line charts.",
        "The Python packaging ecosystem uses pyproject.toml as the standard configuration file. Hatchling and setuptools are common build backends.",
        "SQLAlchemy provides a full suite of ORM tools for database access. The Session object manages transactions and identity maps.",
        "Regular expressions in Python are handled by the re module. Use re.compile() for patterns you reuse frequently to avoid recompilation overhead.",
        "The Python csv module reads and writes tabular data in comma-separated format. DictReader maps each row to a dictionary keyed by column headers.",
    ],
    messages=[{"role": "user", "content": "How do I run multiple async tasks concurrently in Python?"}],
    query="asyncio gather concurrent coroutine event loop",
)

show_result(result)


# ── scenario 2: long conversation compaction ──────────────────────────────────

header("SCENARIO 2 · Long conversation — 20-turn history compaction")
print("""
  Situation: a coding assistant mid-session. 20 turns of history cover
  project setup, decisions made, constraints established. The optimizer
  extracts the state and keeps only the recent tail verbatim.
""")

history = [
    {"role": "user",      "content": "I want to build a REST API for a task management app."},
    {"role": "assistant", "content": "Great! What tech stack are you thinking? I'd suggest FastAPI for Python."},
    {"role": "user",      "content": "Yes, let's use FastAPI. I also need a database."},
    {"role": "assistant", "content": "PostgreSQL would pair well with FastAPI. We can use SQLAlchemy as the ORM."},
    {"role": "user",      "content": "I decided to go with PostgreSQL and SQLAlchemy. Can we add authentication?"},
    {"role": "assistant", "content": "Sure — JWT tokens are the standard for REST APIs. We'll use python-jose."},
    {"role": "user",      "content": "The requirement is that we must use OAuth2 with JWT. No sessions."},
    {"role": "assistant", "content": "Understood. FastAPI has built-in OAuth2 support. I'll set that up."},
    {"role": "user",      "content": "We also need rate limiting. The API must not allow more than 100 req/min per user."},
    {"role": "assistant", "content": "I'll add slowapi for rate limiting. It integrates cleanly with FastAPI."},
    {"role": "user",      "content": "Good. We agreed to use slowapi. What about error handling?"},
    {"role": "assistant", "content": "FastAPI's HTTPException is standard. We'll add a global exception handler."},
    {"role": "user",      "content": "The unresolved question is how to structure the task priority system."},
    {"role": "assistant", "content": "We could use an integer 1–5 priority field, or an enum. What do you prefer?"},
    {"role": "user",      "content": "Let's go with an integer priority field, 1 being highest priority."},
    {"role": "assistant", "content": "Done. Priority will be integer 1–5, 1 = highest. I'll add validation."},
    {"role": "user",      "content": "Also note that the client requires all timestamps to be UTC ISO-8601."},
    {"role": "assistant", "content": "Noted. All datetimes will be stored and returned as UTC ISO-8601 strings."},
    {"role": "user",      "content": "The deadline for the MVP is end of month. We need to keep scope tight."},
    {"role": "assistant", "content": "Understood — MVP scope: tasks CRUD, auth, rate limiting. No extras."},
]

opt2 = ContextOptimizer(
    provider="openai", model="gpt-4o",
    max_input_tokens=8192, output_budget=1024,
    mode=CompressionMode.CONSERVATIVE,
    compactor_recency_tail=4,
)

result2 = opt2.optimize(
    system="You are an expert Python backend engineer helping build a REST API.",
    history=history,
    messages=[{"role": "user", "content": "Can you write the SQLAlchemy model for the Task entity?"}],
)

show_result(result2, show_messages=False)
# Show the compacted state block
state_blocks = [b for b in result2.blocks if b.metadata.source == "compacted_history"]
if state_blocks:
    print(f"\n  compacted state block ({state_blocks[0].token_count} tokens):")
    for line in state_blocks[0].content.split("\n"):
        print(f"    {line}")

print(f"\n  final block count: {len(result2.blocks)}  (was {len(history) + 2} before compaction)")


# ── scenario 3: tool-use / agent prompt ───────────────────────────────────────

header("SCENARIO 3 · Agent prompt — tool schemas + system instructions")
print("""
  Situation: an AI agent with 4 tools and a verbose system prompt.
  Tool schemas have null fields that inflate token count. The optimizer
  canonicalizes JSON, strips nulls, and anchors schemas in the cache prefix.
""")

opt3 = ContextOptimizer(
    provider="openai", model="gpt-4o",
    max_input_tokens=8192, output_budget=2048,
    mode=CompressionMode.CONSERVATIVE,
)

result3 = opt3.optimize(
    system=(
        "You are an intelligent assistant with access to tools.\n\n"
        "Rules:\n"
        "- Always use the most specific tool available\n"
        "- Never make up information not returned by a tool\n"
        "- If a tool call fails, explain why and suggest alternatives\n"
        "- Always confirm destructive actions before executing\n\n"
        "You have access to: web search, calculator, code execution, and file I/O."
    ),
    tools=[
        {
            "name": "search_web",
            "description": "Search the internet for up-to-date information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results", "default": 5},
                    "language":    {"type": "string",  "description": "Language code", "default": "en"},
                    "deprecated":  None,
                    "legacy_mode": None,
                },
                "required": ["query"],
            },
        },
        {
            "name": "run_python",
            "description": "Execute Python code in a sandboxed environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                    "unused":  None,
                    "old_api": "",
                },
                "required": ["code"],
            },
        },
        {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":     {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                    "binary":   None,
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode":    {"type": "string", "description": "Write mode", "default": "w"},
                    "legacy":  None,
                    "v1_compat": "",
                },
                "required": ["path", "content"],
            },
        },
    ],
    messages=[{"role": "user", "content": "Search for the latest Python 3.13 release notes and save a summary to release_notes.md"}],
    query="python release notes",
)

show_result(result3)
t3 = result3.telemetry
cache_pct = (t3.cacheable_prefix_tokens / t3.tokens_after * 100) if t3.tokens_after else 0
print(f"\n  cache prefix is {cache_pct:.0f}% of total output — schema tokens locked in prefix")


# ── scenario 4: modes comparison ─────────────────────────────────────────────

header("SCENARIO 4 · Mode comparison — lossless vs conservative vs aggressive")
print("""
  Same input (system + 6 docs + 10-turn history + user query) run
  through all three compression modes side by side.
""")

kwargs = dict(
    system=(
        "You are a medical information assistant. Provide accurate, evidence-based "
        "information. Always recommend consulting a qualified healthcare provider for "
        "personal medical decisions. Do not diagnose. Cite sources when possible."
    ),
    documents=[
        "Metformin is the first-line treatment for type 2 diabetes. It works by decreasing hepatic glucose production and improving insulin sensitivity. Common side effects include nausea and diarrhea.",
        "SGLT2 inhibitors like empagliflozin reduce cardiovascular mortality in patients with type 2 diabetes and established cardiovascular disease. They also have renoprotective effects.",
        "GLP-1 receptor agonists such as semaglutide promote weight loss and lower blood glucose. They are increasingly used in obesity management independent of diabetes status.",
        "Blood pressure targets for diabetic patients are generally below 130/80 mmHg per ADA guidelines. ACE inhibitors or ARBs are preferred agents due to renoprotective effects.",
        "HbA1c monitoring every 3 months is recommended for patients not at goal; every 6 months for those who are stable. Target is typically below 7% for most non-pregnant adults.",
        "The Mediterranean diet has strong evidence for cardiovascular benefit in diabetic patients. Key components: olive oil, vegetables, legumes, whole grains, fish, moderate wine.",
        "Diabetic nephropathy affects approximately 40% of diabetic patients and is the leading cause of end-stage renal disease. Early detection via urine albumin-to-creatinine ratio is critical.",
    ],
    history=[
        {"role": "user",      "content": "I have type 2 diabetes and want to understand my medication options."},
        {"role": "assistant", "content": "Happy to help explain the main medication classes for type 2 diabetes."},
        {"role": "user",      "content": "My doctor mentioned metformin. Is it safe?"},
        {"role": "assistant", "content": "Metformin is generally very safe and is the most widely prescribed first-line diabetes medication."},
        {"role": "user",      "content": "What about the newer drugs I keep hearing about?"},
        {"role": "assistant", "content": "You may be thinking of SGLT2 inhibitors or GLP-1 agonists — both are newer and have additional cardiovascular and weight benefits."},
        {"role": "user",      "content": "I also have high blood pressure. Does that change things?"},
        {"role": "assistant", "content": "Yes — having both diabetes and hypertension means ACE inhibitors or ARBs are often preferred as they protect the kidneys."},
        {"role": "user",      "content": "My HbA1c is 8.2%. Is that bad?"},
        {"role": "assistant", "content": "An HbA1c of 8.2% is above the typical target of below 7%. Your doctor will likely want to adjust your treatment plan."},
    ],
    messages=[{"role": "user", "content": "What medication would be best if I have diabetes, high blood pressure, and I am overweight?"}],
    query="diabetes medication high blood pressure overweight treatment",
)

for mode in [CompressionMode.LOSSLESS, CompressionMode.CONSERVATIVE, CompressionMode.AGGRESSIVE]:
    opt_m = ContextOptimizer(
        provider="openai", model="gpt-4o",
        max_input_tokens=8192, output_budget=1024,
        mode=mode,
    )
    r = opt_m.optimize(**kwargs)
    t = r.telemetry
    removed_types = [rec.block_type for rec in t.blocks_removed]
    print(
        f"  {mode.value:13s}  {t.tokens_before:>5,} → {t.tokens_after:>5,} tok"
        f"  ({t.reduction_pct:5.1f}%)  blocks {t.blocks_before}→{t.blocks_after}"
        f"  dropped={len(t.blocks_removed)}"
        f"  risk={t.risk_level}"
    )


# ── scenario 5: exact duplicate injection ─────────────────────────────────────

header("SCENARIO 5 · Duplicate injection — same chunk retrieved N times")
print("""
  Situation: a RAG pipeline retrieves from 3 overlapping indexes.
  The same chunks appear multiple times. Deduplication runs first
  (lossless) before any other compression step.
""")

chunk_a = "The transformer architecture introduced in 'Attention is All You Need' (Vaswani et al., 2017) uses self-attention mechanisms to process sequences in parallel rather than recurrently."
chunk_b = "BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations by conditioning on both left and right context in all layers."

opt5 = ContextOptimizer(provider="openai", model="gpt-4o", mode=CompressionMode.LOSSLESS)
result5 = opt5.optimize(
    system="You are an NLP research assistant.",
    documents=[
        chunk_a,          # from index 1
        chunk_b,          # from index 1
        chunk_a,          # from index 2 (duplicate)
        chunk_b,          # from index 2 (duplicate)
        chunk_a,          # from index 3 (duplicate)
        "GPT models are autoregressive language models that predict the next token given all previous tokens. GPT-3 has 175 billion parameters.",
    ],
    messages=[{"role": "user", "content": "Explain the transformer architecture."}],
)

dedup_removed = [r for r in result5.telemetry.blocks_removed if "duplicate" in r.reason]
print(f"  input docs    : 6")
print(f"  deduped away  : {len(dedup_removed)}")
print(f"  unique kept   : {sum(1 for b in result5.blocks if b.block_type.value == 'retrieved_doc')}")
print(f"  token savings : {result5.telemetry.tokens_before - result5.telemetry.tokens_after} tokens  ({result5.telemetry.reduction_pct:.1f}%)")


print("\n" + "━" * 72)
print("  All scenarios complete.")
print("━" * 72 + "\n")


# ── PHASE 2 SCENARIOS ────────────────────────────────────────────────────────

header("SCENARIO 6 · Granular trimming — section-level selection inside a doc")
print("""
  Situation: a long structured document (with headings). The query is specific.
  Phase 2 trims irrelevant *sections* inside the block rather than dropping
  the whole block as Phase 1 would.
""")

LONG_DOC = """\
# Executive Summary
This quarterly report covers financial performance across all business units.
Revenue grew across the board driven by strong cloud adoption.

# Cloud Services
Cloud revenue increased 35% year-over-year to $11.4 billion.
Enterprise contract renewals were at an all-time high of 94%.
New product launches in AI infrastructure drove incremental demand.
Three new data center regions went live in Q3.

# Advertising Revenue
Digital advertising revenue declined 4% due to macroeconomic headwinds.
CPM rates fell in North America while APAC remained stable.
Brand safety initiatives reduced inventory but improved pricing power.

# Hardware Division
Hardware sales were flat at $3.2 billion.
Supply chain normalisation reduced component costs by 8%.
Next-generation device lineup is on track for Q1 launch.

# Outlook
Management expects cloud growth to accelerate to 40% in Q4.
Advertising recovery is expected in H1 of next year.
Operating margin guidance raised to 28-30%.
"""

from optimizerllm.presets import rag_optimizer

opt6 = rag_optimizer(provider="openai", model="gpt-4o", max_input_tokens=8192)
result6 = opt6.optimize(
    system="You are a financial analyst assistant.",
    documents=[LONG_DOC],
    messages=[{"role": "user", "content": "How is cloud revenue performing and what is the outlook?"}],
    query="cloud revenue growth outlook",
)

print(result6.summary())
print(f"\nOriginal doc: {len(LONG_DOC)} chars / {result6.telemetry.tokens_before} tokens")
trimmed_block = next((b for b in result6.blocks if "cloud" in b.content.lower()), None)
if trimmed_block:
    print(f"Trimmed block: {len(trimmed_block.content)} chars")
    print(f"\nTrimmed content:\n")
    for line in trimmed_block.content.split("\n"):
        print(f"  {line}")


header("SCENARIO 7 · Presets — same prompt, three different presets")
print("""
  Using the preset API with rag / chat / long_context_qa on the same input.
""")

from optimizerllm.presets import preset, list_presets

print("  Available presets:")
for p in list_presets():
    print(f"    {p['name']:20s}  {p['description']}")

print()

history = [
    {"role": "user",      "content": "I'm researching renewable energy storage."},
    {"role": "assistant", "content": "I can help. Are you focused on grid-scale or residential?"},
    {"role": "user",      "content": "Grid-scale. Specifically lithium iron phosphate batteries."},
    {"role": "assistant", "content": "LFP batteries are a strong choice for grid storage due to cycle life and safety."},
    {"role": "user",      "content": "What about vanadium flow batteries as an alternative?"},
    {"role": "assistant", "content": "Vanadium flow batteries offer longer duration storage and better cycle life."},
]
documents = [
    "Lithium iron phosphate (LFP) batteries offer 3000-6000 charge cycles and excellent thermal stability. They are widely deployed in grid-scale energy storage systems due to low cost and long operational life.",
    "Vanadium redox flow batteries can store energy for 4-12 hours and have virtually unlimited cycle life since the electrolyte does not degrade. Capital costs are higher but lifetime costs are competitive.",
    "Pumped hydro accounts for over 90% of global grid storage capacity. It requires specific geography but offers multi-hour to multi-day storage duration at low operating cost.",
    "The levelised cost of storage (LCOS) for grid-scale lithium-ion systems fell below $200/MWh in 2024, a 60% decrease over five years.",
]
messages = [{"role": "user", "content": "What is the best battery chemistry for 8-hour grid storage at lowest lifetime cost?"}]

for preset_name in ["rag", "chat", "long_context_qa"]:
    opt = preset(preset_name, provider="openai", model="gpt-4o", max_input_tokens=8192)
    r = opt.optimize(system="You are an energy storage expert.", history=history, documents=documents, messages=messages, query="battery grid storage 8 hour lifetime cost")
    t = r.telemetry
    print(f"  [{preset_name:15s}]  {t.tokens_before:>4} → {t.tokens_after:>4} tok  ({t.reduction_pct:5.1f}%)  blocks {t.blocks_before}→{t.blocks_after}  risk={t.risk_level}")


header("SCENARIO 8 · CLI — optimizerllm inspect")
print("  (runs the inspect command on a JSON prompt file)\n")

import json, subprocess, sys, tempfile, os

prompt_data = {
    "system": "You are a helpful assistant.",
    "documents": [
        "The Amazon River is the largest river by discharge in the world.",
        "The Nile River is generally considered the longest river in the world.",
        "The Mississippi River flows through the central United States.",
    ],
    "messages": [{"role": "user", "content": "Which river has the highest water discharge?"}],
    "query": "river discharge water flow volume",
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(prompt_data, f)
    tmp_path = f.name

try:
    proc = subprocess.run(
        [sys.executable, "-m", "optimizerllm.cli", "inspect", "--file", tmp_path,
         "--mode", "conservative", "--model", "gpt-4o"],
        capture_output=True, text=True
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("stderr:", proc.stderr)
finally:
    os.unlink(tmp_path)

