# optimizerllm

Deterministic context optimization engine for LLM prompts.

```python
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
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    documents=["Paris is the capital of France."],
    query="capital of France",
)

print(result.rendered)   # OpenAI-compatible messages list
print(result.summary())  # Audit report
```
