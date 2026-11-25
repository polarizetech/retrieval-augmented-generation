# Example: what the research-rag tool sends back

Placeholder data in the exact layout the code builds (`research_mcp/ask.py`,
`research_pipeline/render.py`, `research_pipeline/mcp_server.py`). Everything below is what the
**calling model** (Claude/ChatGPT) receives. The person only sees what that model chooses to show.

One question, with every part of the server in play:

> research: Do two tones produce a cubic distortion product in the cochlea, and how large is it
> compared to the primaries?

That takes three tool calls from the client model:

1. `ask` → routes to the research pipeline, picks a specialist, starts a run.
2. `pipeline__research_continue` (called repeatedly) → "working… working…"
3. `pipeline__research_continue` → the finished answer, which also pulls in the companion
   tools (datasets, patents, math/BioNumbers lookups) and the retraction checks.

A quick question (no `research:` prefix) only does call 1, and returns passages instead.
That version is shown at the end (Call 1b).

---

## Call 1: `ask` (research route)

```json
{
  "request_id": "a1b2c3d4e5f6",
  "status": "working",
  "route": {
    "mode": "research",
    "reason": "prefix 'research:'",
    "matched": ["research:"]
  },
  "specialist": {
    "name": "nonlinear-signal-mixing",
    "why": "anchor terms: 'distortion product', 'two tones'",
    "roles": ["researcher", "nonlinear-signal-mixing"],
    "brief": "<agents-dir>/nonlinear-signal-mixing.md",
    "repository": "",
    "guidance": "Name the order of each product (2f1-f2 is third order). Say whether a level is relative to the primaries or absolute. Keep acoustic, mechanical and electrical measurements apart."
  },
  "progress": [
    "Route: research (prefix 'research:')",
    "Specialist: nonlinear-signal-mixing (anchor terms: 'distortion product', 'two tones')",
    "Started research run r-20261105-0912 (running)"
  ],
  "seconds": 1.4,
  "results": {
    "research": {
      "run_id": "r-20261105-0912",
      "status": "running",
      "model_backend": "openrouter:near-ai/qwen3-235b"
    }
  },
  "relay": "**Results**\n- Started research run `r-20261105-0912` (running).\n- Nothing is concluded until the run finishes.\n\n**Next**\n- The run carries on; the assistant fetches its answer with `hub__result` (`a1b2c3d4e5f6`).",
  "relay_full": "**What the research tool did** (1.4 s)\n- Route: research (prefix 'research:')\n- Specialist: nonlinear-signal-mixing (anchor terms: 'distortion product', 'two tones')\n- Started research run r-20261105-0912 (running)\n\n**Research run** `r-20261105-0912`: running. It runs on the gateway (model: openrouter:near-ai/qwen3-235b); continue with `pipeline__research_continue` until it is done, then show its answer with its Limits.\n\n**Limits:** The answer arrives when the run is done; until then nothing is concluded.\n\n_Request `a1b2c3d4e5f6` · Usage: OpenRouter: $XX.XX left, $X.XX/day, ~N days · NEAR AI: needs a reporting token · Ollama: ok, loaded: bge-m3:latest_",
  "relay_instructions": "Show `relay` to the person exactly as written. It is short on purpose: show `relay_full` (passages, steps, limits) only when the person asks for more. The passages are in `results` either way, for grounding your answer. Add anything of your own only after it, under a heading that says it is yours. Do not add facts from memory."
}
```

The person sees (the `relay`):

> **Results**
> - Started research run `r-20261105-0912` (running).
> - Nothing is concluded until the run finishes.
>
> **Next**
> - The run carries on; the assistant fetches its answer with `hub__result` (`a1b2c3d4e5f6`).

---

## Call 2: `pipeline__research_continue` (while running, repeated)

```json
{
  "run_id": "r-20261105-0912",
  "status": "working",
  "stage": "verify",
  "progress": "planned 4 sub-questions · 9 searches · 14 papers read · 31 passages · 11 claims drafted · verifying 6/11",
  "relay": "**Results**\n- Still working: verifying claims (6 of 11).\n\n**Next**\n- The assistant calls `pipeline__research_continue` again."
}
```

---

## Call 3: `pipeline__research_continue` (done)

The JSON envelope:

```json
{
  "run_id": "r-20261105-0912",
  "status": "done",
  "relay": "**Results**\n- Two tones at f1 and f2 make the cochlea produce distortion products, the largest at the cubic difference frequency 2f1–f2 [1][2]. In isolated cochlear segments these reached about 13% of the primaries' magnitude [3].\n- 6 of 11 claim(s) passed verification, from 4 source(s).\n- Findings, sources and limits are in the full answer.\n\n**Next**\n- Nothing else runs; this research request is done.",
  "relay_instructions": "Show `relay` to the person exactly as written. Show `answer` (the full answer, with its findings, sources and Limits) only when the person asks for more.",
  "answer": "<the whole markdown document below>",
  "run_dir": "~/.local/share/research-runs/r-20261105-0912",
  "stats": {
    "seconds": 412.6,
    "model_backend": "openrouter:near-ai/qwen3-235b",
    "searches": 9,
    "evidence": 31,
    "claims": 11,
    "claims_kept": 6,
    "strictness": "grounded",
    "independent_verifier": true
  }
}
```

The `answer` field, rendered:

> # Do two tones produce a cubic distortion product in the cochlea, and how large is it compared to the primaries?
>
> ## Bottom line
>
> Two tones at f1 and f2 make the cochlea produce distortion products, the largest at the cubic
> difference frequency 2f1–f2 [1][2]. In isolated cochlear segments these reached about 13% of the
> primaries' magnitude [3]. In human ears the emission measured in the ear canal is a mix of two
> components from different places along the basilar membrane [1].
>
> ## Verification risk
>
> - Strictness: **grounded** — every claim needs a retrieved passage that states it; secondary reports count.
> - Independent verifier: **yes**
> - Claims: 6 kept; 5 removed; 1 disputed.
> - Risk flag: one source is a 2025 study on ex vivo tissue, not intact ears.
>
> ## Findings
>
> ### Do two tones produce a cubic distortion product in the cochlea?
>
> - Two tones at f1 and f2 (f2 > f1) produce intermodulation products, including the cubic difference tone 2f1–f2 and the quadratic difference tone f2–f1. [2]
>   *Evidence: direct statement — review-style introduction; mouse; laser vibrometry*
> - In humans the largest distortion-product otoacoustic emission is at 2f1–f2. [1]
>   *Evidence: direct statement — human; DPOAE; reporting other work*
>
> ### How large is it relative to the primaries?
>
> - The first products to appear as level rose were 2f1–f2 and 2f2–f1, at about 15% of the primaries' magnitude. [3]
>   *Evidence: measured — ex vivo cochlear segments; cochlear microphonic*
> - Averaged over preparations, cubic products were 12.8 ± 7.8% of the primaries (n = 4). [3]
>   *Evidence: measured — ex vivo; four preparations; small sample*
> - Near a Hopf bifurcation, distortion products are predicted to grow at a roughly fixed ratio to the primaries. [3] *(verifiers disagreed)*
>   *Evidence: model prediction — theory, not measurement*
>
> ### Where along the cochlea are they generated?
>
> - The emission at 2f1–f2 has a nonlinear-distortion component from near the f2 place and a coherent-reflection component from the 2f1–f2 place. [1]
>   *Evidence: direct statement — human; reporting other work*
>
> ### Does the same mechanism appear outside the mammalian cochlea?
>
> - *Insufficient evidence: no claim for this sub-question survived verification.*
>
> ## Conflicting or negative evidence
>
> - No opposing result was retrieved. Searches run to find one: “absence of 2f1-f2 distortion product”; “distortion product not detected cochlea”; “failed replication DPOAE level ratio”. An empty result is not evidence that the finding is uncontested.
>
> ## Datasets that could test this
>
> - `openneuro/ds004xxx@1.0.2` — DPOAE recordings, 48 adults, two-tone sweeps (licence CC0)
> - `physionet/xxxx@1.0` — otoacoustic emission screening set (licence: none stated; not permission)
> - `zenodo/1234567@v2` — mouse cochlear vibrometry raw files (licence CC BY 4.0)
>
> ## Existing patents on this
>
> - US 10,XXX,XXX B2 — “Method for measuring distortion product otoacoustic emissions with …” (2019)
> - EP 3XXXXXX A1 — “Hearing screening device using two-tone stimulation” (2021)
>
> ## Constants and numbers looked up
>
> - BNID 10XXXX — human cochlear length ≈ 35 mm (BioNumbers; source: …)
>
> ## Limits of this answer
>
> - Dataset entries are catalogue pointers, not evidence; none was downloaded or analysed.
> - Patent entries are search hits, not legal or freedom-to-operate advice.
> - 4 drafted claim(s) were removed because their cited passage did not support them: “Distortion products are always below 20% of …”; …
> - 1 drafted claim(s) were removed because they stated a number their cited passage does not contain: “… about −40 dB re primaries”
> - Relevant-looking papers that could not be read (no open-access copy), so they contributed nothing: Robles & Ruggero, Mechanics of the mammalian cochlea (2001, doi:10.1152/physrev.2001.81.3.1305); …
> - 3 further candidate paper(s) were not fetched (fetch budget).
> - Search providers that did not answer at least once: core. Their silence is not zero results.
> - 1 passage(s) were excluded for containing hidden characters.
> - Evidence is text passages only. Results that appear only in figures, tables or supplements were not seen.
> - Searched as grounded: peer-reviewed and preprint literature in English … If the question falls outside that scope, these rules for what counts as evidence may not fit it.
> - Every quoted passage was retrieved and machine-checked, not read by a person. The evidence labels are computed from the retrieved set and are not formal evidence grades.
>
> ## Sources
>
> 1. Zelle et al., 2017. Input-output functions of the nonlinear-distortion component of distortion-product otoacoustic emissions in normal and hearing-impaired human ears. JASA. doi:10.1121/1.4982923 — *published; no retraction or concern found in Crossref; full text via pmc-s3:jats-xml*
>    - [E04, reporting other work, passage chars 1329–2708] “In humans, the most pronounced DPOAE is found at the cubic difference frequency fDP = 2f1–f2 …”
> 2. Dewey, 2022. Cubic and quadratic distortion products in vibrations of the mouse cochlear apex. JASA. doi:10.1121/10.0015244 — *published; no retraction or concern found in Crossref; full text via pmc-s3:jats-xml*
>    - [E11, experimental, passage chars 0–934] “… the cochlea generates significant intermodulation distortion products (DPs) at frequencies such as the “cubic” difference tone 2f1–f2 …”
> 3. Alonso et al., 2025. Amplification through local critical behavior in the mammalian cochlea. PNAS. doi:10.1073/pnas.2503389122 — *published; no retraction or concern found in Crossref; full text via pmc-s3:jats-xml*
>    - [E17, experimental, passage chars 27140–28261] “… whose magnitudes were about 15% of those of the primary tones …”
>    - [E18, experimental, passage chars 27140–28261] “… cubic combination tones whose magnitudes averaged 12.8 ± 7.8% relative to the primary tones …”
>    - [E19, modelling, passage chars 34258–35424] “… the magnitude of these distortion products is predicted to grow at an approximately fixed ratio …”
> 4. …
>
> *Generated by qwen3-235b; verified by qwen3-235b, glm-4.6; 9 searches, 31 evidence passages, 6/11 claims kept.*

---

## Call 1b: `ask` (quick route, no `research:` prefix)

This is the one you've been seeing most. Five passages; `results` carries each passage's full
text (up to 1,800 characters), and `relay_full` repeats the first 320 characters of each.

```json
{
  "request_id": "6f8e5653a3ff",
  "status": "done",
  "route": {"mode": "quick", "reason": "no research trigger words", "matched": []},
  "specialist": {
    "name": "nonlinear-signal-mixing",
    "why": "anchor terms: 'difference tone'",
    "roles": ["researcher", "nonlinear-signal-mixing"],
    "brief": "<agents-dir>/nonlinear-signal-mixing.md",
    "repository": "",
    "guidance": "Name the order of each product (2f1-f2 is third order). Say whether a level is relative to the primaries or absolute."
  },
  "progress": [
    "Route: quick (no research trigger words)",
    "Specialist: nonlinear-signal-mixing (anchor terms: 'difference tone')",
    "Retrieved 4 passage(s) (hybrid) in 9.0 s",
    "Left out 1 passage(s) under the relevance floor (reranker score < -3; best -4.10)"
  ],
  "seconds": 9.1,
  "results": {
    "evidence": {
      "query": "How do two tones mix nonlinearly in the cochlea to create a difference tone, and how big is it compared to the primaries?",
      "retrieval": "hybrid",
      "reranker": "onnx:onnx-community/bge-reranker-v2-m3-ONNX/onnx/model_int8.onnx",
      "n": 4,
      "results": [
        {
          "evidence_id": "W4309756930#p0.1a6c1ea9",
          "work": "W4309756930",
          "start": 0,
          "end": 934,
          "text": "<full passage, up to 1,800 characters>",
          "truncated": false,
          "title": "Cubic and quadratic distortion products in vibrations of the mouse cochlear apex",
          "doi": "10.1121/10.0015244",
          "pmid": "36456371",
          "year": 2022,
          "authors": ["James B. Dewey"],
          "bm25_rank": 3,
          "dense_rank": 2,
          "score": 1.10
        },
        "… 3 more of the same shape …"
      ],
      "excluded": [{"evidence_id": "W2xxxx#p12.ab12cd34", "flags": ["retracted"]}],
      "index": {
        "papers": 4620,
        "passages": 221514,
        "with_vectors": 221514,
        "stale": 1,
        "embedding_model": "bge-m3",
        "path": "~/.local/share/paper-fetch/passages.sqlite"
      },
      "below_floor": {"n": 1, "floor": -3.0, "best_score": -4.1}
    }
  },
  "relay": "**Results**\n- Found 4 passage(s) from 3 paper(s); the closest is from Cubic and quadratic distortion products in vibrations of the mouse cochlear apex, 2022.\n- 1 weaker passage(s) under the relevance floor were left out.\n- 1 passage(s) were left out by the safety or retraction check.\n- Specialist: nonlinear-signal-mixing.\n- These are passages, not a checked answer.\n\n**Next**\n- The assistant answers from these passages and cites their evidence ids.",
  "relay_full": "**What the research tool did** (9.1 s)\n- Route: quick …\n- Specialist: …\n- Retrieved 4 passage(s) …\n- Left out 1 passage(s) …\n\n**nonlinear-signal-mixing guidance:** Name the order of each product …\n\n**Passages**\n1. “In the mammalian cochlea, sound-evoked waves … (first 320 chars)…” (Cubic and quadratic …, 2022, doi:10.1121/10.0015244) `W4309756930#p0.1a6c1ea9`\n2. …\n3. …\n4. …\n\n(1 passage(s) excluded by the safety or retraction check.)\n\n**Limits:** These are retrieved passages, not a checked answer: nothing here judged whether they answer the question. Start the request with `research:` for a checked answer.\n\n_Request `6f8e5653a3ff` · Usage: OpenRouter: $XX.XX left … · Ollama: ok, loaded: bge-m3:latest_",
  "relay_instructions": "Show `relay` to the person exactly as written. It is short on purpose: … Do not add facts from memory."
}
```

The person sees (the `relay`):

> **Results**
> - Found 4 passage(s) from 3 paper(s); the closest is from Cubic and quadratic distortion products in vibrations of the mouse cochlear apex, 2022.
> - 1 weaker passage(s) under the relevance floor were left out.
> - 1 passage(s) were left out by the safety or retraction check.
> - Specialist: nonlinear-signal-mixing.
> - These are passages, not a checked answer.
>
> **Next**
> - The assistant answers from these passages and cites their evidence ids.

Then the client model writes its own answer underneath, from the passages in `results`.

---

## Where the length comes from

- **The same content three times.** In Call 1b the passages sit in `results` (full text), again
  in `relay_full` (excerpts), and are summarised in `relay`. In Call 3, `relay` repeats the
  answer's bottom line.
- **The relay reports process, not findings.** In the quick route, `relay` tells the person
  how many passages came back, what was left out, and which specialist was used, but not what
  the passages say. The useful part only shows up if the client model writes an answer after it.
- **Fixed boilerplate.** "These are passages, not a checked answer", the usage footer and
  `relay_instructions` appear on every call. The research answer's Limits section always lists
  the same three or four standard caveats.
- **Bookkeeping fields.** `index`, `route`, `specialist.brief`, `bm25_rank` and `dense_rank`
  help with debugging but not with answering.
