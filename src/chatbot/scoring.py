"""Comparaison des appels d'outils du LLM avec les appels attendus.

Une requête de test attend une liste d'appels {"tool", "args"}. Seuls les
arguments écrits dans "args" sont vérifiés : un appel attendu sans "args"
(par exemple le 2e appel d'une chaîne, dont l'argument dépend du 1er
résultat) est validé par le nom de l'outil seul.
"""

from pathlib import PurePath


def norm(value):
    """Normalise une valeur : race ou image écrite autrement, nombre en texte."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return float(s)
        except ValueError:
            pass
        return PurePath(s).stem.lower().replace(" ", "_").replace("-", "_")
    return value


def arg_ok(expected, actual) -> bool:
    e, a = norm(expected), norm(actual)
    if isinstance(e, float) and isinstance(a, float):
        return abs(e - a) < 1e-6
    return e == a


def call_matches(expected: dict, actual: dict, defaults: dict | None = None) -> bool:
    """Vrai si l'appel réel a le bon outil et les arguments attendus.

    Un argument attendu absent de l'appel réel est accepté s'il vaut la
    valeur par défaut de l'outil (defaults = {outil: {argument: défaut}}).
    """
    if expected["tool"] != actual["name"]:
        return False
    tool_defaults = (defaults or {}).get(expected["tool"], {})
    args = actual.get("args") or {}
    for key, value in (expected.get("args") or {}).items():
        if key in args:
            if not arg_ok(value, args[key]):
                return False
        elif key not in tool_defaults or not arg_ok(value, tool_defaults[key]):
            return False
    return True


def score_query(query: dict, calls: list[dict], defaults: dict | None = None) -> dict:
    """Score d'une requête.

    - tools_ok : chaque outil attendu a été appelé (sans regarder les arguments) ;
    - calls_ok : chaque appel attendu a un appel réel correspondant (outil et
      arguments), dans l'ordre si la requête est "ordered" (défaut : oui) ;
    - matched : nombre d'appels attendus retrouvés ;
    - extra : appels réels qui ne correspondent à aucun appel attendu.
    Une requête sans appel attendu est réussie si le LLM n'appelle aucun outil.
    """
    expected = query["expected"]
    ordered = query.get("ordered", True)

    if not expected:
        ok = not calls
        return {"tools_ok": ok, "calls_ok": ok, "matched": 0,
                "n_expected": 0, "extra": len(calls)}

    names = [c["name"] for c in calls]
    tools_ok = all(names.count(t) >= [e["tool"] for e in expected].count(t)
                   for t in {e["tool"] for e in expected})

    used: set[int] = set()
    matched = 0
    start = 0
    for exp in expected:
        candidates = range(start, len(calls)) if ordered else range(len(calls))
        hit = next((i for i in candidates
                    if i not in used and call_matches(exp, calls[i], defaults)), None)
        if hit is None:
            continue
        used.add(hit)
        matched += 1
        if ordered:
            start = hit + 1

    return {"tools_ok": tools_ok, "calls_ok": matched == len(expected),
            "matched": matched, "n_expected": len(expected),
            "extra": len(calls) - len(used)}


def summarize(scores: list[dict]) -> dict:
    """Taux agrégés sur toutes les requêtes."""
    n = len(scores)
    chains = [s for s in scores if s["n_expected"] >= 2]
    n_expected = sum(s["n_expected"] for s in scores)
    return {
        "n_queries": n,
        "tool_selection_rate": sum(s["tools_ok"] for s in scores) / n if n else 0.0,
        "correct_call_rate": sum(s["calls_ok"] for s in scores) / n if n else 0.0,
        "expected_call_recall": (sum(s["matched"] for s in scores) / n_expected
                                 if n_expected else 0.0),
        "chain_success_rate": (sum(s["calls_ok"] for s in chains) / len(chains)
                               if chains else 0.0),
        "n_chains": len(chains),
        "extra_calls": sum(s["extra"] for s in scores),
    }
