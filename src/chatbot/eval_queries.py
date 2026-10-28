"""Évalue les appels d'outils du chatbot sur le jeu de requêtes de test.

Nécessite Ollama et le serveur MCP HTTP lancés (voir README, mode 3).
Depuis la racine du dépôt :

    uv run --with-requirements requirements.txt python src/chatbot/eval_queries.py
    ... python src/chatbot/eval_queries.py --models qwen2.5:3b,llama3.2:3b
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client as http_client

from agent import run_agent
from config import MCP_URL, MODEL
from mcp_tools import call_tool, to_ollama_tools
from scoring import score_query, summarize

ROOT = Path(__file__).resolve().parents[2]


def tool_defaults(tools) -> dict:
    """{outil: {argument: valeur par défaut}} lu dans les schémas MCP."""
    out = {}
    for t in tools:
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
        out[t.name] = {k: p["default"] for k, p in schema.get("properties", {}).items()
                       if "default" in p}
    return out


async def check_image_ids(session: ClientSession, queries: list[dict]) -> list[str]:
    """Identifiants d'images du jeu de test absents de la collection."""
    ids = {e["args"]["image_id"] for q in queries for e in q["expected"]
           if "image_id" in (e.get("args") or {})}
    missing = []
    for image_id in sorted(ids):
        out = await call_tool(session, "classify", {"image_id": image_id, "top_k": 1})
        if '"error"' in out or out.startswith("Erreur"):
            missing.append(image_id)
    return missing


async def eval_model(session, ollama_tools, defaults, queries, model) -> dict:
    results = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            reply, trace = await run_agent(
                session, ollama_tools, [{"role": "user", "content": q["query"]}],
                model=model, options={"temperature": 0, "seed": 0},
            )
            error = None
        except Exception as e:  # OllamaError, timeout, serveur MCP coupé...
            reply, trace, error = "", [], str(e)
        seconds = time.perf_counter() - t0

        calls = [{"name": c["name"], "args": c["args"]} for c in trace]
        score = score_query(q, calls, defaults)
        results.append({**q, "calls": calls, "reply": reply, "error": error,
                        "seconds": round(seconds, 2), **score})

        mark = "OK " if score["calls_ok"] else "RATÉ"
        shown = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                          for c in calls) or "aucun outil"
        print(f"  [{mark}] {q['id']:>2} {seconds:5.1f}s  {shown}")

    summary = summarize(results)
    summary["mean_seconds"] = round(sum(r["seconds"] for r in results) / len(results), 2)
    summary["errors"] = sum(r["error"] is not None for r in results)
    return {"model": model, "summary": summary, "results": results}


def print_table(reports: list[dict]) -> None:
    print("\n| Modèle | Bon outil | Bons appels (outil + arguments) | "
          "Rappel des appels | Chaînes réussies | Appels en trop | Temps moyen |")
    print("|---|---|---|---|---|---|---|")
    for r in reports:
        s = r["summary"]
        print(f"| {r['model']} | {s['tool_selection_rate']:.0%} | {s['correct_call_rate']:.0%} | "
              f"{s['expected_call_recall']:.0%} | {s['chain_success_rate']:.0%} "
              f"({s['n_chains']}) | {s['extra_calls']} | {s['mean_seconds']} s |")


async def main(args) -> None:
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    if args.only:
        keep = {int(i) for i in args.only.split(",")}
        queries = [q for q in queries if q["id"] in keep]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    async with http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            ollama_tools = to_ollama_tools(tools)
            defaults = tool_defaults(tools)

            missing = await check_image_ids(session, queries)
            if missing:
                print(f"Attention : images absentes de la collection : {', '.join(missing)}")

            reports = []
            for model in args.models.split(","):
                print(f"\n== {model} ({len(queries)} requêtes)")
                report = await eval_model(session, ollama_tools, defaults, queries, model)
                report["missing_image_ids"] = missing
                reports.append(report)
                path = out_dir / f"{re.sub(r'[^\w.-]', '_', model)}.json"
                path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                print(f"Résultats écrits dans {path}")

    print_table(reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=MODEL,
                        help="Modèles Ollama séparés par des virgules")
    parser.add_argument("--queries", default=str(ROOT / "tests" / "chatbot_queries.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "chatbot"))
    parser.add_argument("--only", help="Identifiants de requêtes à lancer, ex. 1,4,15")
    asyncio.run(main(parser.parse_args()))
