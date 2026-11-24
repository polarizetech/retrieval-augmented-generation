# The research pipeline for one-off batches on a rented server (see docs/cloud.md).
#
# One image holds everything a command-line run needs: the pipeline, paper-fetch (the paper
# library, reading and writing the same DigitalOcean Spaces bucket as the Mac), Ollama for the
# two local models (bge-m3 embeddings, the bespoke-minicheck verifier) and the public agent briefs.
# Nothing secret is built in: keys arrive at run time in an env file, the private agent briefs as
# a mounted folder, and the passage index is downloaded from Spaces when the container starts.
#
#   docker build -f cloud/Dockerfile -t research-pipeline-cloud .

FROM ollama/ollama:0.40.11

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git sqlite3 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --python 3.12 /opt/venv

# paper-fetch at a pinned commit, with S3 (Spaces), dense retrieval and the ONNX reranker.
ARG PAPER_FETCH_REF=d733c253717d39f290f4b0d9ce0341b148deb85f
RUN uv pip install "paper-fetch[mcp,s3,retrieval,rerank] @ git+https://github.com/polarizetech/paper-fetch@${PAPER_FETCH_REF}"

# The public agent briefs, pinned. Private extensions are mounted at /agents/private.
ARG AGENTS_REF=5ed292c058a640bd48afee3fa377247700942097
RUN git clone --quiet https://github.com/polarizetech/scientific-research-agents /opt/agents \
    && git -C /opt/agents checkout --quiet "${AGENTS_REF}"

# The pipeline itself, from this checkout.
COPY pyproject.toml README.md /opt/rag/
COPY research_pipeline /opt/rag/research_pipeline
COPY research_mcp /opt/rag/research_mcp
COPY domains /opt/rag/domains
COPY cloud /opt/rag/cloud
RUN uv pip install /opt/rag && for d in /opt/rag/domains/*/; do uv pip install "$d"; done

# The cloud defaults (no secrets). config/rag.env is never copied into the image.
ENV PIPELINE_LLM=endpoint \
    PIPELINE_ENDPOINT_URL=https://openrouter.ai/api/v1 \
    PIPELINE_ENDPOINT_MODEL=qwen/qwen3.8-27b \
    PIPELINE_ENDPOINT_REASONING=off \
    PIPELINE_ENDPOINT_ZDR=1 \
    PIPELINE_ENDPOINT_SORT=price \
    PIPELINE_VERIFIER_MODEL=bespoke-minicheck:latest \
    PIPELINE_STRICTNESS=grounded \
    PIPELINE_NUM_CTX=16384 \
    PIPELINE_DATA_DIR=/work/data \
    PIPELINE_RUNS_DIR=/work/runs \
    MCP_CONFIG=/opt/rag/cloud/gateway.json \
    PIPELINE_AGENT_PROFILES_DIR=/opt/agents \
    PIPELINE_AGENT_EXTENSIONS_DIR=/agents/private \
    OLLAMA_HOST=127.0.0.1:11434 \
    PAPER_FETCH_ENV_DIR=/nonexistent \
    PAPER_FETCH_STORE=s3 \
    PAPER_FETCH_INDEX=/work/index/passages.sqlite \
    PAPER_FETCH_EMBED_MODEL=bge-m3 \
    PAPER_FETCH_RERANK_MODEL=onnx-community/bge-reranker-v2-m3-ONNX::onnx/model_int8.onnx \
    HF_HOME=/work/hf

WORKDIR /work
ENTRYPOINT ["/opt/rag/cloud/entrypoint.sh"]
