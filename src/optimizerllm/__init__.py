"""
optimizerllm — deterministic context optimization engine for LLM prompts.

Quickstart::

    from optimizerllm import ContextOptimizer, CompressionMode

    optimizer = ContextOptimizer(
        provider="openai",
        model="gpt-4o",
        max_input_tokens=128_000,
        output_budget=4_096,
        mode=CompressionMode.CONSERVATIVE,
    )

    result = optimizer.optimize(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Summarize the following docs."}],
        documents=["Doc 1 content...", "Doc 2 content..."],
        query="Summarize the docs",
    )

    # OpenAI-compatible messages list
    messages = result.rendered

    # Audit trail
    print(result.summary())
"""

from optimizerllm.presets import (
    preset,
    rag_optimizer,
    chat_optimizer,
    agent_optimizer,
    long_context_qa_optimizer,
    summarization_optimizer,
    list_presets,
)
from optimizerllm.core.block import (
    Block,
    BlockMetadata,
    BlockType,
    CompressionMode,
    CompressionPolicy,
    Mutability,
    make_example_block,
    make_message_block,
    make_retrieved_doc_block,
    make_system_block,
    make_tool_schema_block,
)
from optimizerllm.core.result import OptimizationResult, TelemetryReport
from optimizerllm.optimizer import ContextOptimizer
from optimizerllm.slot_guard.guard import Slot, SlotGuard, SlotIndex, SlotType
from optimizerllm.task_synth.synth import TaskDescriptor, TaskSynth, TaskType
from optimizerllm.recover.recover import Recover, RecoveryRecord
from optimizerllm.risk_controller.controller import RiskController, RiskAssessment
from optimizerllm.reranking.reorder import SemanticReorder
from optimizerllm.filtering.adaptive import AdaptiveEvidenceFilter
from optimizerllm.budget.controller import BudgetController, BudgetPlannerV2

__all__ = [
    # Main API
    "ContextOptimizer",
    # Presets
    "preset",
    "rag_optimizer",
    "chat_optimizer",
    "agent_optimizer",
    "long_context_qa_optimizer",
    "summarization_optimizer",
    "list_presets",
    # Core types
    "Block",
    "BlockMetadata",
    "BlockType",
    "CompressionMode",
    "CompressionPolicy",
    "Mutability",
    # Convenience constructors
    "make_system_block",
    "make_message_block",
    "make_tool_schema_block",
    "make_retrieved_doc_block",
    "make_example_block",
    # Result types
    "OptimizationResult",
    "TelemetryReport",
    # Slot Guard
    "Slot",
    "SlotGuard",
    "SlotIndex",
    "SlotType",
    # Task Synthesis
    "TaskDescriptor",
    "TaskSynth",
    "TaskType",
    # Recovery
    "Recover",
    "RecoveryRecord",
    # v2: Risk Controller
    "RiskController",
    "RiskAssessment",
    # v2: Semantic Reorder
    "SemanticReorder",
    # v2: Adaptive Evidence Filter
    "AdaptiveEvidenceFilter",
    # v2: Budget Planner
    "BudgetController",
    "BudgetPlannerV2",
]

__version__ = "0.2.0"
