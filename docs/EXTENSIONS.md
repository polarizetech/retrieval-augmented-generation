# Writing an extension

An **extension** is a separately installed Python package that adds MCP tools and `ask` steps to
research-rag's gateway. You build your own research-rag by installing this repository's package
and your extensions in the same environment. Nothing in this repository has to be edited.

The core in `research_mcp/` and `research_pipeline/` knows how to answer, retrieve, verify and grade.
It does not know what your tool does or what your step looks for. That is what an extension
supplies, and the code that loads it lives in `research_mcp/extensions.py`.

## The one rule

**An extension adds tools and steps. It never changes evidence, retrieval, verification, grading,
coverage or another step's results.** Its output lives only under its own key,
`results["extensions"][<extension name>]`, and reaches clients as `brief.tool_results.extensions`.

This is the same rule as for domains ([DOMAINS.md](DOMAINS.md)), and for the same reason: a run
has to stay auditable. An answer's evidence, grades and checks come from the core alone, so an
extension that is installed, removed or broken cannot change what the core found. If your
extension fails, the ask still answers: the failure is named in the progress list and in its own
key, and nothing else moves.

## Anatomy

```
rag-ext-<prefix>/
├── pyproject.toml            # distribution + the entry point that makes it discoverable
└── rag_<prefix>/
    └── __init__.py           # the tools, the steps, and one declare() call at the bottom
```

Copy `docs/extension-template/` and rename it. The template adds one tool, `example__hello`, and
one ask step, `hello`. Its `pyproject.toml` registers the package under the `rag.extensions` entry
point, which is how the gateway finds it. The package is never imported by name from the core.

The template's `__init__.py` is the whole pattern:

```python
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


HELLO = ToolSpec(tool=Tool(name="example__hello", ...), handler=say_hello, label="example hello")


async def hello_step(step_input: StepInput) -> StepOutput:
    return StepOutput(result={"hello": "world"}, progress=["example: said hello"])


HELLO_STEP = StepSpec(name="hello", run=hello_step, triggers=("example extension",))

EXTENSION = declare(
    name="rag-ext-example",
    module=__name__,
    version="0.1.0",
    core_requires=">=0.3.0,<0.5",
    job="Shows the extension API: one tool and one ask step.",
    prefix="example",
    tools=(HELLO,),
    steps=(HELLO_STEP,),
)
```

The template's `Tool` carries the input schema and the annotations (see its file for both). The
module-level names are the convention: `say_hello` is the tool's handler, `HELLO` its `ToolSpec`,
`hello_step` the step's function, `HELLO_STEP` its `StepSpec`, and `EXTENSION` the value
`declare()` returns.

## `declare()`

`declare()` runs when the package is imported. It checks the declaration and registers it:

| Field | What it is |
|---|---|
| `name` | Distribution-style, starting `rag-ext-`. |
| `module` | The importable package name, `__name__` in its `__init__.py`. |
| `version` | Dotted integers, the extension's own version. |
| `core_requires` | The research-rag version range it was written for, e.g. `>=0.3.0,<0.5`. |
| `job` | One line: the one thing this adds. It may not be blank. |
| `prefix` | Lowercase letters, digits and underscores, starting with a letter. Its tools are named `<prefix>__<tool>`. It may not be one of the reserved core prefixes. |
| `tools` | The `ToolSpec`s this adds. |
| `steps` | The `StepSpec`s this adds. Step names are unique. |
| `instructions` | Optional. Added to the server's instructions when the server sets none of its own. |

A declaration that fails any check raises at import. Two extensions may not claim the same prefix,
and a declared name may not come from two modules. The version gate works as it does for domains:
an engine outside `core_requires` raises `IncompatibleCore`, and discovery reports the extension
as `incompatible` instead of loading it.

## Tools

A `ToolSpec` has the MCP `Tool`, a `handler`, a `label` and a `component`:

- The tool's name must start with the extension's prefix and two underscores, so `example__hello`
  for prefix `example`. A tool with any other name is refused at declaration.
- Set its annotations (`readOnlyHint`, `idempotentHint`, `openWorldHint`) honestly: clients use
  them to decide on approval prompts.
- The `handler` is called with the arguments dict and a `Context`. It may be a plain function or
  an `async` one. A plain function runs in a worker thread, so it does not block the gateway.
- `Context` gives the handler `call_tool` (an async call to a tool the gateway already has, by
  name and arguments), `has_tool` (whether one is available), `data_dir` (a folder the gateway
  creates for this extension), and `config` (the extension's section of `mcp-gateway.json`, or an
  empty dict), and `config_path` (the path of that `mcp-gateway.json`, for an extension that reads
  another section of it).
- If the handler raises, the client gets a tool error that names the label. The gateway keeps
  running.

An extension tool whose prefix is also held by an upstream server is not offered. A call to it is
refused, and that extension's `gateway_status` row then says why.

## Steps

A `StepSpec` is an ask step. It has a `name`, an async `run`, its `triggers` and an `always` flag:

- A step runs for a request when one of its `triggers` occurs in the question as whole words, or
  when `always` is true. Matching ignores case, and the question is the one left after any routing
  prefix is removed.
- `run` receives a `StepInput`: the request's `question`, its `mode`, `project`, `repo`, the
  `context` and a deep copy of `results` so far. Changing that copy changes nothing: the core's
  results are not affected.
- `run` returns a `StepOutput`, or `None` to say nothing. Its fields are:
  - `result`: a dict, stored under `results["extensions"][<extension name>][<step name>]`.
  - `progress`: lines for the progress list. When it is empty, the reply shows `<prefix>: ran`.
  - `found`, `next`, `for_you`: one-sentence items added to the reply's summary lists: what came
    back, what the assistant does next, and what the person needs to do.
  - `relay`: one markdown block, added to the full reply after the sun-sight section.
- A step that raises does not fail the ask. The result under its key becomes
  `{"error": "<ExceptionType>: <message>"}` and the progress list names the failure.

Within each stage, steps run in the order their extensions were loaded.

## Stages, shared prefixes and brief keys

A step's `stage` says when it runs. There are three values:

- `early` runs once the request is routed, before any retrieval. Its result also goes into a
  pending reply, when the request outlasts its deadline.
- `before_retrieval` runs just before retrieval, and may ask for retrieval to be skipped.
- `after` runs where steps ran before this version, after retrieval. It is the default.

A `before_retrieval` step sets `skip_retrieval` on its `StepOutput` to a reason. That skips
passage retrieval only in quick mode, and only when no specialist and no research words call for
passages. The reason shows in the progress list as `Retrieval: skipped (<reason>)`. Its `limits`
field replaces the closing line of the reply's summary when nothing was retrieved in quick mode.
Any step's `limits` may be used, but the first one set in a request wins.

Some core prefixes are shared, so an extension may add tools under them. `SHARED_PREFIXES` holds
`math` for now, and a tool there is named `math__<tool>`. A core tool's name can't be taken: a name
the core already uses under a shared prefix is refused at declaration.

A step's `StepOutput` may also set `brief`, and its extension may declare a `brief_key` and a
`brief_note`. The step's `brief` value then goes into the reply's `brief` under that key, and the
`brief_note` text is put at the start of the reply's `answer_format`. A brief key is the one
exception to the rule above: the value is not under `results["extensions"]`. A brief key must be
lowercase letters, digits and underscores, may not be a key the core's brief already uses, and only
one extension may hold it.

Version 0.4.0 is the first core version with stages, shared prefixes and brief keys. An extension
that uses any of them declares `core_requires=">=0.4.0"`.

## Routing words

A step's trigger words can be extended by the person, in the same way the core's routing is. In
`config/routing.toml` (gitignored, next to `config/routing.default.toml`):

```toml
[extensions."rag-ext-example"]
hello = ["extra words"]
```

The words are added to the step's own triggers and never remove them. The default file has the
same table, commented out, as an example.

## Per-extension config and `disabled`

In `mcp-gateway.json` (copied from `config/mcp-gateway.example.json`):

```json
"extensions": {
  "disabled": ["rag-ext-example"],
  "rag-ext-example": {}
}
```

- Every installed extension loads by itself. Naming it in `disabled` switches it off: its tools
  are not offered, its steps do not run, and its `gateway_status` row says `disabled`.
- A section named after the extension is passed to its tools and steps as `context.config`.

## Status rows

`gateway_status` reports every installed extension under `extensions`, one row each:

| Status | Meaning |
|---|---|
| `ok` | Declared and loaded. The row lists its `name`, `version`, `prefix`, `job`, `tools` and `steps`. |
| `disabled` | Named in `extensions.disabled`. |
| `clash` | Its prefix is also held by an upstream server, so its tools are refused. The row gives the reason. |
| `undeclared` | Its module was imported but never called `declare()`. |
| `incompatible` | Its `core_requires` does not match the installed research-rag. |
| `error` | Importing it failed. The reason is the exception. |

A broken extension is reported, never silently dropped, and it cannot stop the gateway from
starting.

## Building your own research-rag

Your research-rag is this repository's package plus the extensions you want, installed in the same
environment as one set of packages. The gateway finds each extension through its entry point when
it starts.

Keep the extension's `core_requires` in step with the research-rag you run. When the core moves to
a new minor version, an extension that declares the old range is reported as `incompatible` until
it is rebuilt against the new one.

## Why plugins

Scientific software tends to grow faster than its core can be reviewed, and the people who use it
need to add their own analyses without forking it. Gewaltig and Cannon (2014) argue that such
software "may require scripting or plugin support for extensions" (DOI
10.1371/journal.pcbi.1003376). The rule above is what makes that safe here: an extension can add
to what the research-rag does, and it cannot alter what the research-rag found.
