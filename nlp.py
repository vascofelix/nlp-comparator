import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional
import urllib.request
import urllib.error


@dataclass
class NlpResult:
    entities: list
    model_ms: Optional[float]
    postprocess_ms: Optional[float]
    total_ms: float
    status: int
    error: Optional[str] = None

    @property
    def nlp_server_total_ms(self) -> Optional[float]:
        if self.model_ms is not None and self.postprocess_ms is not None:
            return self.model_ms + self.postprocess_ms
        return None

    def to_dict(self) -> dict:
        return {
            "entities": self.entities,
            "model_ms": self.model_ms,
            "postprocess_ms": self.postprocess_ms,
            "nlp_server_total_ms": self.nlp_server_total_ms,
            "total_ms": self.total_ms,
            "entity_count": len(self.entities),
            "status": self.status,
            "error": self.error,
        }


def _build_multipart(text: str, filename: str = "input.txt") -> tuple:
    boundary = uuid.uuid4().hex
    text_bytes = text.encode("utf-8")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + text_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def call_nlp(url: str, text: str) -> NlpResult:
    body, content_type = _build_multipart(text)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            elapsed = (time.monotonic() - start) * 1000
            data = resp.read().decode("utf-8")
            parsed = json.loads(data)

            if isinstance(parsed, dict) and "entities" in parsed:
                entities = parsed["entities"]
                body_timing = parsed.get("timing", {})
            else:
                entities = parsed
                body_timing = {}

            model_ms = resp.headers.get("X-Model-Ms")
            postprocess_ms = resp.headers.get("X-PostProcess-Ms")

            return NlpResult(
                entities=entities,
                model_ms=float(model_ms) if model_ms else body_timing.get("model_ms"),
                postprocess_ms=float(postprocess_ms) if postprocess_ms else body_timing.get("postprocess_ms"),
                total_ms=round(elapsed, 2),
                status=resp.status,
            )
    except urllib.error.HTTPError as e:
        elapsed = (time.monotonic() - start) * 1000
        return NlpResult(
            entities=[],
            model_ms=None,
            postprocess_ms=None,
            total_ms=round(elapsed, 2),
            status=e.code,
            error=str(e),
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return NlpResult(
            entities=[],
            model_ms=None,
            postprocess_ms=None,
            total_ms=round(elapsed, 2),
            status=0,
            error=str(e),
        )


def entity_key(ent: dict) -> tuple:
    return (ent.get("text", ""), ent.get("label_", ""), ent.get("start_char", 0), ent.get("end_char", 0))


def diff_entities(ents_a: list, ents_b: list) -> dict:
    set_a = {entity_key(e) for e in ents_a}
    set_b = {entity_key(e) for e in ents_b}

    only_a = sorted(set_a - set_b, key=lambda x: x[2])
    only_b = sorted(set_b - set_a, key=lambda x: x[2])
    common = set_a & set_b

    label_changes = []
    positional_a = {(e.get("start_char"), e.get("end_char")): e for e in ents_a}
    positional_b = {(e.get("start_char"), e.get("end_char")): e for e in ents_b}
    for pos in set(positional_a.keys()) & set(positional_b.keys()):
        ea, eb = positional_a[pos], positional_b[pos]
        if ea.get("label_") != eb.get("label_"):
            label_changes.append({
                "text": ea.get("text", ""),
                "position": list(pos),
                "label_a": ea.get("label_"),
                "label_b": eb.get("label_"),
            })

    return {
        "common_count": len(common),
        "only_in_a": [{"text": e[0], "label": e[1], "start": e[2], "end": e[3]} for e in only_a],
        "only_in_b": [{"text": e[0], "label": e[1], "start": e[2], "end": e[3]} for e in only_b],
        "label_changes": label_changes,
        "total_a": len(ents_a),
        "total_b": len(ents_b),
    }


def avg(lst: list) -> Optional[float]:
    return round(sum(lst) / len(lst), 2) if lst else None


def run_comparison(url_a: str, url_b: str, text: str, runs: int = 1) -> dict:
    results_a = []
    results_b = []

    for _ in range(runs):
        results_a.append(call_nlp(url_a, text))
        results_b.append(call_nlp(url_b, text))

    if runs > 1:
        final_a = NlpResult(
            entities=results_a[-1].entities,
            model_ms=avg([r.model_ms for r in results_a if r.model_ms is not None]),
            postprocess_ms=avg([r.postprocess_ms for r in results_a if r.postprocess_ms is not None]),
            total_ms=round(avg([r.total_ms for r in results_a]), 2),
            status=results_a[-1].status,
        )
        final_b = NlpResult(
            entities=results_b[-1].entities,
            model_ms=avg([r.model_ms for r in results_b if r.model_ms is not None]),
            postprocess_ms=avg([r.postprocess_ms for r in results_b if r.postprocess_ms is not None]),
            total_ms=round(avg([r.total_ms for r in results_b]), 2),
            status=results_b[-1].status,
        )
    else:
        final_a = results_a[0]
        final_b = results_b[0]

    return {
        "a": final_a.to_dict(),
        "b": final_b.to_dict(),
        "entity_diff": diff_entities(final_a.entities, final_b.entities),
        "runs": runs,
    }
