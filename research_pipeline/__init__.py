"""Evidence-first literature research pipeline for small local models.

Orchestration is deterministic code. The language model is confined to narrow, schema-constrained
sub-tasks (decompose, extract, judge), and nothing it writes reaches the final answer without a
retrieved passage and a verification verdict behind it. See docs/research-pipeline.md.
"""

__version__ = "0.2.0"
