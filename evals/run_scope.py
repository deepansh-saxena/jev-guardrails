#!/usr/bin/env python3
"""Run the labelled scope set through both guardrail backends.

    python evals/run_scope.py               both backends
    python evals/run_scope.py --backend jev
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from evals.scope_cases import CASES
    from tlife_agent.guardrails import violations as gv
    from tlife_agent.guardrails.backends import set_backend
    from tlife_agent.guardrails.recording import record_scope

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"

BAND_ORDER = ["clear_in", "hard_in", "safety", "adversarial",
              "clear_out", "hard_out", "mixed"]


def run(name: str):
    backend = set_backend(name)
    rows = []
    for band, message, should_refuse in CASES:
        t0 = time.perf_counter()
        decision = backend.classify_scope(message, "triage")
        ms = (time.perf_counter() - t0) * 1000
        gv.reset("ev")
        refused = False if decision is None else record_scope("ev", decision, "triage", name)
        rows.append({"band": band, "message": message, "expect": should_refuse,
                     "refused": refused, "ms": ms, "d": decision})
    return rows


def report(name: str, rows: list[dict]) -> None:
    scored = [r for r in rows if r["expect"] is not None]
    correct = sum(1 for r in scored if r["refused"] == r["expect"])
    fp = [r for r in scored if r["refused"] and not r["expect"]]
    fn = [r for r in scored if not r["refused"] and r["expect"]]
    lats = [r["ms"] for r in rows]
    confs = [r["d"].confidence for r in rows if r["d"]]

    print(f"\n{BOLD}{name.upper()}{RESET}  {correct}/{len(scored)} "
          f"({100*correct/len(scored):.0f}%)   "
          f"median {st.median(lats):.0f}ms   total {sum(lats)/1000:.1f}s")

    by_band = defaultdict(lambda: [0, 0])
    for r in scored:
        by_band[r["band"]][1] += 1
        by_band[r["band"]][0] += r["refused"] == r["expect"]
    parts = []
    for band in BAND_ORDER:
        if band in by_band:
            ok, n = by_band[band]
            colour = GREEN if ok == n else (RED if band == "safety" else YELLOW)
            parts.append(f"{colour}{band} {ok}/{n}{RESET}")
    print("  " + "  ".join(parts))

    if fp:
        print(f"  {RED}false refusals ({len(fp)}) -- refused a real customer:{RESET}")
        for r in fp:
            tag = f"{RED}SAFETY{RESET} " if r["band"] == "safety" else ""
            cat = r["d"].category if r["d"] else "?"
            print(f"    {tag}[{r['band']}] {r['message'][:58]}  -> {cat}")
    if fn:
        print(f"  {YELLOW}missed ({len(fn)}) -- let something through:{RESET}")
        for r in fn:
            print(f"    [{r['band']}] {r['message'][:58]}")
    if confs:
        print(f"  {DIM}confidence  min {min(confs):.2f}  median "
              f"{st.median(confs):.2f}  max {max(confs):.2f}  "
              f"spread {max(confs)-min(confs):.2f}{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llm", "jev"], default=None)
    args = ap.parse_args()
    names = [args.backend] if args.backend else ["jev", "llm"]

    scored = sum(1 for c in CASES if c[2] is not None)
    print(f"{BOLD}Scope eval{RESET}  {len(CASES)} cases ({scored} scored, "
          f"{len(CASES)-scored} mixed/unscored)")

    results = {}
    for name in names:
        results[name] = run(name)
        report(name, results[name])

    if len(names) == 2:
        print(f"\n{BOLD}Mixed cases{RESET} {DIM}(no ground truth -- "
              f"disagreement is the signal){RESET}")
        for i, (band, msg, expect) in enumerate(CASES):
            if expect is not None:
                continue
            j = results["jev"][i]["refused"]
            l = results["llm"][i]["refused"]
            mark = f"{YELLOW}differ{RESET}" if j != l else f"{DIM}agree {RESET}"
            print(f"  {mark}  jev={'refuse' if j else 'answer'}  "
                  f"llm={'refuse' if l else 'answer'}   {msg[:52]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
