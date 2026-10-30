<!--
How replies look. To change them, copy this file to config/replies.md (not tracked by git) and
edit it there; any template you leave out of your copy keeps the default below. Edits show on the
next reply, with no restart.

  {{name}}               the block called name (empty when there is nothing to show)
  {{#name}} ... {{/name}}  shown only when name has something in it (good for headings)
  {{^name}} ... {{/name}}  shown only when name is empty
  <!- - note - ->          a note like this one is dropped from replies

Code still writes every block, so passages, quotes, evidence ids and sources stay exactly as
retrieved. Templates choose which blocks appear, their order and their headings. Leaving out
{{limits}} or {{problems}} hides what went wrong or what an answer can't support, so keep them.
-->

<!-- template: ask -->
<!--
The quick reply from `ask` (passages, a research run started, or coder context).
Blocks:
  problems     what failed or degraded, and the fallback used (first, so it isn't missed)
  route        quick, research or code
  question     the question as asked
  seconds      how long the gateway took
  steps        what ran, as a bullet list
  specialist   the specialist's name, if one was picked
  guidance     the specialist's guidance (not shown for research runs)
  passages     numbered verbatim passages, each with its source and evidence id
  passages_none  "none matched." when retrieval ran and found nothing
  excluded     how many passages the safety or retraction check took out
  search       scholarly search titles (no passages; not citable)
  research     the research run started, or the run this request attached to
  coder        the coder note, and any overlap with other sessions' work
  limits       what this reply can't support
  request      the request id
  usage        provider balance and usage
  footer       request id and usage together, as one line
-->
{{problems}}

**What the research tool did** ({{seconds}} s)
{{steps}}
{{#guidance}}

**{{specialist}} guidance:** {{guidance}}
{{/guidance}}
{{#passages}}

**Passages**
{{passages}}
{{excluded}}
{{/passages}}
{{#passages_none}}

**Passages:** {{passages_none}}
{{excluded}}
{{/passages_none}}
{{#search}}

**Scholarly search results** (titles only: no verbatim passages, not citable)
{{search}}
{{/search}}
{{#research}}

{{research}}
{{/research}}
{{#coder}}

{{coder}}
{{/coder}}

**Limits:** {{limits}}

{{#footer}}_{{footer}}_{{/footer}}

<!-- template: pending -->
<!--
The reply at the 45-second deadline, while the request is queued or still working.
Blocks: problems (queued, with position and wait), working (the "still working" line when not
queued), seconds, steps, request, usage, footer.
-->
{{problems}}
{{working}}

**What the research tool did so far** ({{seconds}} s)
{{steps}}

**Limits:** Nothing is concluded yet; the answer comes through `hub__result`.

{{#footer}}_{{footer}}_{{/footer}}

<!-- template: answer -->
<!--
The research pipeline's finished answer.
Blocks:
  question     the question
  bottom_line  the verified summary, or why there is none
  thin         why the answer is thin, cause by cause with what would change it (a bullet list;
               empty when the answer is not thin)
  risk         strictness, verifier and claim counts, risk flags (a bullet list)
  findings     verified claims under each sub-question, with [n] source marks
  conflicts    disagreeing sources and the search for null results
  tools        dataset and patent companion records, each under its own heading
  limits       what the answer can't support (a bullet list)
  sources      the numbered bibliography with verbatim quotes
  footer       which models wrote and checked it, and the counts
-->
# {{question}}

## Bottom line

{{bottom_line}}
{{#thin}}

## Why this answer is thin

{{thin}}
{{/thin}}
{{#risk}}

## Verification risk

{{risk}}
{{/risk}}

## Findings

{{findings}}

## Conflicting or negative evidence

{{conflicts}}
{{#tools}}

{{tools}}
{{/tools}}

## Limits of this answer

{{limits}}

## Sources

{{sources}}

{{footer}}
