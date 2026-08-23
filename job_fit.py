from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import cohere

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
EMBED_DIM = 1024


def load_chunks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def embed(co: cohere.ClientV2, texts: list[str], input_type: str) -> np.ndarray:
    res = co.embed(
        model=EMBED_MODEL,
        texts=texts,
        input_type=input_type,
        embedding_types=["float"],
        output_dimension=EMBED_DIM,
    )
    print(res.embeddings.float)
    return np.array(res.embeddings.float, dtype=np.float32)


def retrieve(resume: list[str], job: list[str], resume_vecs: np.ndarray, job_vecs: np.ndarray, k: int) -> list[dict]:
    hits = []
    scores = cosine(job_vecs, resume_vecs)
    for i, req in enumerate(job):
        idx = np.argsort(-scores[i])[:k]
        for j in idx:
            hits.append(
                {
                    "requirement": req,
                    "evidence": resume[int(j)],
                    "embed_score": float(scores[i, j]),
                }
            )
    # unique evidence, keep best embed score
    best: dict[str, dict] = {}
    for h in hits:
        key = h["evidence"]
        if key not in best or h["embed_score"] > best[key]["embed_score"]:
            best[key] = h
    return sorted(best.values(), key=lambda x: -x["embed_score"])


def rerank_evidence(co: cohere.ClientV2, query: str, candidates: list[dict], top_n: int) -> list[dict]:
    docs = [c["evidence"] for c in candidates]
    res = co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=min(top_n, len(docs)))
    ranked = []
    for r in res.results:
        item = dict(candidates[r.index])
        item["rerank_score"] = float(r.relevance_score)
        ranked.append(item)
    return ranked


def write_memo(co: cohere.ClientV2, ranked: list[dict], job_text: str) -> str:
    evidence = "\n".join(
        f"- [{h['rerank_score']:.3f}] {h['evidence']}" for h in ranked
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You write short, specific hiring memos. No filler. No 'passionate' or 'excited'. "
                "Ground every claim in the evidence list. If evidence is weak, say so."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a 4-paragraph fit memo for this Cohere intern posting using only the "
                f"reranked evidence.\n\nPOSTING:\n{job_text}\n\nEVIDENCE (rerank score in brackets):\n{evidence}\n\n"
                "Cover: (1) production SWE / APIs / frontend, (2) data + ML-adjacent work, "
                "(3) CI/CD and security, (4) one honest gap. End with a one-line recommendation."
            ),
        },
    ]
    res = co.chat(model=CHAT_MODEL, messages=messages)
    return res.message.content[0].text


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Score a resume against the Cohere intern posting.")
    parser.add_argument("--resume", type=Path, default=DATA / "resume_chunks.txt")
    parser.add_argument("--job", type=Path, default=DATA / "cohere_intern.txt")
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument("--rerank-n", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print ranked hits as JSON as well.")
    args = parser.parse_args()

    if not os.getenv("CO_API_KEY"):
        raise SystemExit("Set CO_API_KEY in .env (copy .env.example). Get a key at https://dashboard.cohere.com/api-keys")

    co = cohere.ClientV2()
    resume = load_chunks(args.resume)
    job = load_chunks(args.job)

    print(f"Embedding {len(resume)} resume chunks and {len(job)} job lines ({EMBED_MODEL})...")
    resume_vecs = embed(co, resume, "search_document")
    job_vecs = embed(co, job, "search_query")

    candidates = retrieve(resume, job, resume_vecs, job_vecs, k=args.retrieve_k)
    query = "software engineer intern frontend backend APIs data pipelines CI/CD machine learning production"
    print(f"Reranking {len(candidates)} candidates ({RERANK_MODEL})...")
    ranked = rerank_evidence(co, query, candidates, top_n=args.rerank_n)

    print("\nTop evidence")
    print("-" * 60)
    for i, h in enumerate(ranked, start=1):
        print(f"{i}. rerank={h['rerank_score']:.3f}  embed={h['embed_score']:.3f}")
        print(f"   {h['evidence']}")

    print(f"\nWriting memo ({CHAT_MODEL})...\n")
    memo = write_memo(co, ranked, args.job.read_text(encoding="utf-8"))
    print(memo)

    out = ROOT / "examples" / "last_run.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "models": {"embed": EMBED_MODEL, "rerank": RERANK_MODEL, "chat": CHAT_MODEL},
        "ranked": ranked,
        "memo": memo,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    if args.json:
        print(json.dumps(ranked, indent=2))


if __name__ == "__main__":
    main()
