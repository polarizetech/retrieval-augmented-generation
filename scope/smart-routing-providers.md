# Scope: smart-routing-providers

Goal: a request runs only the pipeline steps it needs, and the person can choose per request where
each heavy step runs (the client's model, OpenRouter, Modal, the Mac or the droplet).

Tier: explore

Features:

F1. Research mode runs only when evidence is asked for: a `research:` or `deep:` prefix, or an
evidence-asking phrase. Bare nouns such as "paper", "study" or "cite" route to a quick lookup, and
the server instructions stop telling assistants to start research runs on their own.
Source: build

F2. In the explore tier, research runs check each claim once and skip the verifier calls on the
summary and the specialist views.
Source: build

F3. `ask` takes an optional `providers` map that chooses, for one request, which provider runs each
model step (planning, extraction, writing, verify), including "client" for the assistant's own
model.
Source: build

F4. The same `providers` map chooses where embeddings and reranking run for one request: modal,
mac, droplet, or off (rerank only).
Source: build

F5. Reranking can use the pipeline's text model ("llm") instead of a cross-encoder.
Source: build
