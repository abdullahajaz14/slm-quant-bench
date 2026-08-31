"""slmbench.prompts: task instruction templates and per-model chat formats.

Template strings are protocol data, fixed at the pilot freeze.

Design notes carried from the spec:
  - backend.generate takes a `stop: list[str]` parameter, fed from STOP
    below per model.
  - chat formats are rendered MANUALLY here, keeping the raw completion
    API in the backend: one code path serves both generation and
    choice_loglik, token counting stays exact, and the template applied
    is visible in the results record rather than hidden in the GGUF.
  - none of the templates includes a BOS token: llama.cpp adds BOS at
    tokenisation, and doubling it measurably shifts outputs. Eyeball
    one rendered prompt per model on first run (the pilot report does
    this).
"""

from __future__ import annotations

from .adapters.base import Item

# --- Task instruction templates (fixed protocol data) -------------------

TASK_TEMPLATES: dict[str, str] = {
    "extractive_qa": (
        "Read the contract excerpt and answer the question by quoting the "
        "exact text from the excerpt. If the answer is a span, reply with "
        "the span only, no explanation.\n\n"
        "Contract excerpt:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "multihop_qa": (
        "Read the passages and answer the question. Reply with the answer "
        "only, as briefly as possible, no explanation.\n\n"
        "Passages:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "summarise": (
        "Summarise the following article in 3 to 4 sentences covering the "
        "key facts. Reply with the summary only.\n\n"
        "Article:\n{context}\n\n"
        "Summary:"
    ),
    "grounded_qa": (
        "Read the passage and answer the question using only the "
        "passage. Reply with the exact answer text from the passage, "
        "no explanation.\n\nPassage:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "document_summarise": (
        "Summarise the following public-sector document section in 3 "
        "to 5 sentences covering the key points. Reply with the "
        "summary only.\n\nDocument section:\n{context}\n\nSummary:"
    ),
    # TruthfulQA MC does not render through generate(); choice_loglik
    # consumes prompt+option pairs built by mc_prompt() below.
    "mc_question": "Q: {question}\nA:",
}

# --- Per-model chat wrappers (fixed protocol data, no BOS tokens) --------

CHAT_FORMATS: dict[str, str] = {
    "phi3-mini":  "<|user|>\n{user}<|end|>\n<|assistant|>\n",
    "gemma2-2b":  "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n",
    "llama32-3b": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                  "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "mistral7b":  "[INST] {user} [/INST]",
}

STOP: dict[str, list[str]] = {
    "phi3-mini":  ["<|end|>", "<|user|>"],
    "gemma2-2b":  ["<end_of_turn>"],
    "llama32-3b": ["<|eot_id|>"],
    "mistral7b":  ["</s>", "[INST]"],
}


def render(item: Item, template_name: str, model_name: str) -> str:
    """Instruction template -> user message -> model chat wrapper.
    Raises KeyError loudly on unknown names, never silently defaults."""
    if template_name not in TASK_TEMPLATES:
        raise KeyError(f"unknown task template {template_name!r}")
    if model_name not in CHAT_FORMATS:
        raise KeyError(f"unknown model chat format {model_name!r}")
    user = TASK_TEMPLATES[template_name].format(
        context=item.context, question=item.question)
    return CHAT_FORMATS[model_name].format(user=user)


def mc_prompt(item: Item, model_name: str) -> str:
    """TruthfulQA choice prompt: the part BEFORE the option continuation.
    choice_loglik(prompt, " " + option) then scores each option; the
    leading space matters for tokenisation."""
    if model_name not in CHAT_FORMATS:
        raise KeyError(f"unknown model chat format {model_name!r}")
    user = TASK_TEMPLATES["mc_question"].format(question=item.question)
    return CHAT_FORMATS[model_name].format(user=user)
