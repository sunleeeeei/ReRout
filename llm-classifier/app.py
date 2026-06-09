"""
LLM Classifier Sidecar - Lightweight intent classification via external LLM API.

Receives requests in ReRout's intent classifier format:
  POST /classify {"text": "user message"}
  Response: {"label": "code_generation", "confidence": 0.92, "used": "llm"}

Supports two modes:
  - mock: Uses keyword rules (for testing the pipeline without a model)
  - llm:  Calls an external OpenAI-compatible API (e.g., Higress + Qwen) for classification
"""

from __future__ import annotations

import os
import time
import logging
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import httpx

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

MODE: str = os.getenv("LLM_CLASSIFIER_MODE", "mock")  # "mock" | "llm"
LLM_URL: str = os.getenv("LLM_CLASSIFIER_URL", "")  # e.g. http://higress:8080/v1/chat/completions
LLM_MODEL: str = os.getenv("LLM_CLASSIFIER_MODEL", "qwen2.5-7b")
LLM_API_KEY: str = os.getenv("LLM_CLASSIFIER_API_KEY", "")
LLM_TIMEOUT: int = int(os.getenv("LLM_CLASSIFIER_TIMEOUT", "10"))
LLM_PROMPT: str = os.getenv("LLM_CLASSIFIER_PROMPT", "")
LLM_LABELS: List[str] = [
    lbl.strip()
    for lbl in os.getenv(
        "LLM_CLASSIFIER_LABELS",
        "code_generation,reasoning,summarization,brainstorming,open_qa,chatbot",
    ).split(",")
    if lbl.strip()
]

DEFAULT_PROMPT = (
    "你是一个意图分类器。根据用户输入，判断其意图属于以下哪个类别：\n"
    "{labels}\n\n"
    "只返回类别名称的英文，不要返回其他内容。\n\n"
    "用户输入：{text}"
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RouteLLM - LLM Classifier Sidecar", version="0.1.0")

logger = logging.getLogger("llm-classifier")
if not logger.handlers:
    from pythonjsonlogger import jsonlogger

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Prometheus metrics
REQ_COUNTER = Counter("llm_classifier_requests_total", "Total classify requests")
REQ_ERRORS = Counter("llm_classifier_request_errors_total", "Classify request errors")
LATENCY = Histogram(
    "llm_classifier_latency_seconds", "Classify latency seconds"
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    used: str


# ---------------------------------------------------------------------------
# Mock mode: keyword-based heuristic (mirrors ReRout's intent classifier)
# ---------------------------------------------------------------------------

KEYWORD_MAP: List[Dict[str, Any]] = [
    {
        "label": "code_generation",
        "keywords": ["```", "def ", "class ", "import ", "function", "代码", "编程", "算法", "写代码", "code"],
        "confidence": 0.85,
    },
    {
        "label": "reasoning",
        "keywords": ["prove", "reason", "why", "explain step", "logic", "推理", "分析", "证明", "逻辑", "为什么"],
        "confidence": 0.80,
    },
    {
        "label": "summarization",
        "keywords": ["summarize", "tl;dr", "summary", "总结", "摘要", "概括", "归纳"],
        "confidence": 0.80,
    },
    {
        "label": "brainstorming",
        "keywords": ["brainstorm", "ideas", "story", "poem", "creative", "创意", "头脑风暴", "点子", "想法", "写一个故事"],
        "confidence": 0.75,
    },
    {
        "label": "open_qa",
        "keywords": ["qa", "question", "what is", "who is", "how to", "什么是", "谁是", "怎么", "如何", "为什么"],
        "confidence": 0.70,
    },
]


def classify_mock(text: str) -> ClassifyResponse:
    """Keyword-based classification for mock mode (no LLM call)."""
    t = (text or "").lower()
    for rule in KEYWORD_MAP:
        if any(kw in t for kw in rule["keywords"]):
            return ClassifyResponse(
                label=rule["label"], confidence=rule["confidence"], used="mock"
            )
    return ClassifyResponse(label="chatbot", confidence=0.60, used="mock")


# ---------------------------------------------------------------------------
# LLM mode: call external OpenAI-compatible API
# ---------------------------------------------------------------------------

VALID_LABELS_SET = set(LLM_LABELS)


async def classify_llm(text: str) -> ClassifyResponse:
    """Call external LLM API for intent classification."""
    if not LLM_URL:
        logger.warning("LLM_CLASSIFIER_URL not set, falling back to mock")
        return classify_mock(text)

    prompt_template = LLM_PROMPT if LLM_PROMPT else DEFAULT_PROMPT
    prompt = prompt_template.format(
        labels=", ".join(LLM_LABELS), text=text
    )

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=float(LLM_TIMEOUT)) as client:
            resp = await client.post(LLM_URL, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        content = result["choices"][0]["message"]["content"].strip()
        logger.info({"event": "llm_response", "raw": content})

        # Extract label from response — handle cases where LLM adds extra text
        label = _extract_label(content)
        confidence = 0.90 if label in VALID_LABELS_SET else 0.50

        if label not in VALID_LABELS_SET:
            logger.warning(
                {"event": "unknown_label", "label": label, "fallback": "chatbot"}
            )
            label = "chatbot"
            confidence = 0.40

        return ClassifyResponse(label=label, confidence=confidence, used="llm")

    except Exception as e:
        REQ_ERRORS.inc()
        logger.error({"event": "llm_call_failed", "error": str(e)})
        # Fallback to mock on failure — don't break the pipeline
        fallback = classify_mock(text)
        fallback.used = "llm_fallback_mock"
        return fallback


def _extract_label(content: str) -> str:
    """Extract the classification label from LLM response text."""
    content = content.strip()

    # Try direct match first
    if content.lower() in VALID_LABELS_SET:
        return content.lower()

    # Try to find a label anywhere in the response
    content_lower = content.lower()
    for label in VALID_LABELS_SET:
        if label in content_lower:
            return label

    # Try parsing as JSON (e.g. {"route": "code_generation"})
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            for key in ("route", "label", "category", "intent"):
                if key in parsed:
                    return str(parsed[key]).lower()
    except (json.JSONDecodeError, ValueError):
        pass

    # Return raw content stripped, caller will validate
    return content_lower


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mode": MODE,
        "llm_url": LLM_URL or "(not configured)",
        "llm_model": LLM_MODEL,
        "labels": LLM_LABELS,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    REQ_COUNTER.inc()
    start = time.perf_counter()
    try:
        if MODE == "llm":
            res = await classify_llm(req.text)
        else:
            res = classify_mock(req.text)

        logger.info({
            "event": "classified",
            "mode": MODE,
            "label": res.label,
            "used": res.used,
            "text_preview": req.text[:100],
        })
        return res
    except Exception as e:
        REQ_ERRORS.inc()
        logger.exception({"event": "classify_error", "error": str(e)})
        raise
    finally:
        LATENCY.observe(time.perf_counter() - start)


if __name__ == "__main__":
    import uvicorn

    logger.info({"event": "starting", "mode": MODE, "llm_url": LLM_URL, "model": LLM_MODEL})
    uvicorn.run(app, host="0.0.0.0", port=9000)
