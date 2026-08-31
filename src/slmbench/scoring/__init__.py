"""slmbench.scoring: qa (EM/F1), rouge (ROUGE-1/2/L), choice (MC1/MC2).

rouge is intentionally NOT imported here: it pulls in the external
rouge_score package, which quality runs need but efficiency runs on the
device do not. Import it explicitly where required.
"""

from . import qa, choice  # noqa: F401
