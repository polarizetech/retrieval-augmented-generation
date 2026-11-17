"""An example research-rag extension: one tool and one ask step. Copy this directory and edit.

An extension ADDS tools and steps. It never changes evidence, retrieval, verification, grading or
another step's results; its output lives under its own key. See docs/EXTENSIONS.md.
"""

from __future__ import annotations

from typing import Any

from mcp.types import Tool, ToolAnnotations

from research_mcp.extensions import (
    Context,
    StepInput,
    StepOutput,
    StepSpec,
    ToolSpec,
    declare,
)


def say_hello(arguments: dict[str, Any], context: Context) -> dict[str, Any]:
    return {"hello": arguments.get("name", "world")}


HELLO = ToolSpec(
    tool=Tool(
        name="example__hello",
        inputSchema={"type": "object", "properties": {"name": {"type": "string"}}},
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    ),
    handler=say_hello,
    label="example hello",
)


async def hello_step(step_input: StepInput) -> StepOutput:
    return StepOutput(result={"hello": "world"}, progress=["example: said hello"])


HELLO_STEP = StepSpec(name="hello", run=hello_step, triggers=("example extension",))

EXTENSION = declare(
    name="rag-ext-example",
    module=__name__,
    version="0.1.0",
    core_requires=">=0.3.0,<0.4",
    job="Shows the extension API: one tool and one ask step.",
    prefix="example",
    tools=(HELLO,),
    steps=(HELLO_STEP,),
)
