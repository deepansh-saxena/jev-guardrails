"""Interactive REPL for the care graph.

    python run.py                       # chat
    python run.py --session dana --seed 42
    python run.py --message "why is my bill higher this month?"

In-chat commands:
    /trace      guardrail draws for this session (what was sampled, and why)
    /coverage   how often each guardrail clause has been drawn
    /audit      the audit log (verification, rolls, writes)
    /session    current auth level, handoffs, credits issued
    /whoami     demo accounts and their PINs
    /reset      wipe session state and start over
    /quit
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from . import mock_db as db
from .config import BRAND, DEFAULT_MODEL, missing_key, required_key_env, split_model
from .graph import build_care_graph
from .guardrails.runtime import reset_runtime
from .guardrails import violations as gv
from .guardrails.sampler import reset_sampler, sampler_for
from .prompts import ROUTINE_LABELS
from .tools._session import set_fallback_session

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
MAGENTA, CYAN, YELLOW = "\033[35m", "\033[36m", "\033[33m"
RED, GREY, GREEN = "\033[31m", "\033[90m", "\033[32m"

SEVERITY_COLOR = {"critical": RED, "high": YELLOW, "medium": CYAN, "low": GREY}
SEVERITY_MARK = {"critical": "!!", "high": "!", "medium": "*", "low": "-"}

# Message ids already printed, so replayed history (handoffs carry the desk's
# messages up to the parent) is not rendered twice.
_RENDERED: set[str] = set()

# Per-turn counters, reset at the start of each query.
_TURN_STATS: dict[str, Any] = {"model_calls": 0, "tool_calls": 0,
                               "input_tokens": 0, "output_tokens": 0}


def _drain_stdin() -> list[str]:
    """Lines already sitting in the terminal buffer from a multi-line paste.

    `input()` returns one line per call, so the rest of a pasted block stays
    buffered and is consumed by later prompts -- making every subsequent turn
    look one behind. Pulling them out here lets the REPL queue them and echo
    each as it runs."""
    lines: list[str] = []
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    except Exception:  # noqa: BLE001 - not all stdin types support select
        pass
    return lines


def _banner(session: str, seed: int) -> None:
    print(f"{MAGENTA}{BOLD}{BRAND} care assistant{RESET}")
    from .guardrails.backends import BACKEND_NAME
    from .prompts import static_prompt

    prompt_chars = len(static_prompt("billing"))
    print(f"{DIM}model={DEFAULT_MODEL}  session={session}  guardrail-seed={seed}{RESET}")
    print(f"{DIM}guardrails={BACKEND_NAME}  "
          f"rules-in-prompt={'yes' if BACKEND_NAME == 'llm' else 'no'}  "
          f"desk-prompt={prompt_chars} chars{RESET}")
    print(f"{DIM}Type /help for commands, /quit to exit.{RESET}\n")


def _print_demo_accounts() -> None:
    print(f"\n{BOLD}Demo accounts{RESET} (PIN is what `verify_identity` expects)")
    for cust in db.CUSTOMERS.values():
        pin = cust["account_pin_hash"].split(":")[1]
        flags = []
        if cust["fraud_watch"]:
            flags.append("fraud-watch")
        if cust["customer_id"] in db.HOME_INTERNET:
            flags.append("home-internet")
        bill = db.BILLS.get(cust["customer_id"], [{}])[0]
        print(
            f"  {cust['name']:<22} {cust['phone']}  PIN {pin}  "
            f"{cust['segment']:<16} bill {bill.get('status','-'):<9} "
            f"{' '.join(flags)}"
        )
    print()


def _print_trace(session: str) -> None:
    sess = db.get_session(session)
    trace = sess["guardrail_trace"]
    if not trace:
        print(f"{DIM}No guardrail draws yet.{RESET}")
        return
    print(f"\n{BOLD}Guardrail draws (most recent last){RESET}")
    for entry in trace[-8:]:
        print(
            f"  {CYAN}{entry['routine']:<14}{RESET} turn {entry['turn']:<3} "
            f"strict={entry['strictness']:<5} cap=${entry['credit_cap']:<3} "
            f"n={len(entry['pinned']) + len(entry['sampled'])}"
        )
        print(f"    {DIM}pinned : {', '.join(entry['pinned'])}{RESET}")
        print(f"    {DIM}sampled: {', '.join(entry['sampled']) or '-'}{RESET}")
        if entry["directives"]:
            print(f"    {DIM}injected: {len(entry['directives'])} directive(s){RESET}")
    print()


def _print_coverage(session: str) -> None:
    cov = sampler_for(session).coverage()
    if not cov:
        print(f"{DIM}Nothing drawn yet.{RESET}")
        return
    total = len(sampler_for(session).history)
    print(f"\n{BOLD}Clause coverage over {total} draws{RESET}")
    for cid, n in cov.items():
        bar = "#" * min(30, n)
        print(f"  {cid:<38} {n:>3} {DIM}{bar}{RESET}")
    print()


def _print_audit(session: str) -> None:
    rows = [a for a in db.AUDIT_LOG if a["session_id"] == session]
    if not rows:
        print(f"{DIM}No audit entries yet.{RESET}")
        return
    print(f"\n{BOLD}Audit log{RESET}")
    for row in rows[-20:]:
        detail = {k: v for k, v in row.items() if k not in {"session_id", "at", "event"}}
        print(f"  {YELLOW}{row['event']:<28}{RESET} {DIM}{json.dumps(detail)}{RESET}")
    print()


def _print_violations(session: str) -> None:
    rows = gv.all_for(session)
    if not rows:
        print(f"{GREEN}No guardrail violations this session.{RESET}\n")
        return
    counts = gv.summary(session)
    order = ["critical", "high", "medium", "low"]
    tally = "  ".join(
        f"{SEVERITY_COLOR[s]}{counts[s]} {s}{RESET}" for s in order if s in counts
    )
    print(f"\n{BOLD}Guardrail violations{RESET}  ({tally})")
    for v in rows:
        c = SEVERITY_COLOR.get(v.severity, YELLOW)
        print(f"  {c}{v.violation_id}{RESET} {v.severity:<8} {v.clause_id:<38} "
              f"{v.action:<10} {DIM}{v.routine}{RESET}")
        print(f"      {v.title}")
    print()


def _print_session(session: str) -> None:
    sess = db.get_session(session)
    print(f"\n{BOLD}Session {session}{RESET}")
    for key in ("customer_id", "auth_level", "turn", "handoffs",
                "credits_issued_usd", "actions_taken"):
        print(f"  {key:<22} {sess.get(key)}")
    print()


def _render_update(node: str, payload: dict[str, Any], verbose: bool) -> None:
    messages = payload.get("messages") or []
    for msg in messages:
        mid = getattr(msg, "id", None)
        if mid:
            if mid in _RENDERED:
                continue
            _RENDERED.add(mid)
        if isinstance(msg, AIMessage):
            _TURN_STATS["model_calls"] += 1
            usage = getattr(msg, "usage_metadata", None) or {}
            _TURN_STATS["input_tokens"] += usage.get("input_tokens") or 0
            _TURN_STATS["output_tokens"] += usage.get("output_tokens") or 0
            _TURN_STATS["tool_calls"] += len(msg.tool_calls or [])
            if msg.tool_calls and verbose:
                for call in msg.tool_calls:
                    args = json.dumps(call["args"], default=str)
                    if len(args) > 110:
                        args = args[:107] + "..."
                    print(f"  {DIM}[{node}] -> {call['name']}({args}){RESET}")
            text = getattr(msg, "text", "") or ""
            if isinstance(text, str) and text.strip():
                label = ROUTINE_LABELS.get(node, node)
                print(f"{MAGENTA}{BOLD}{label}:{RESET} {text.strip()}\n")
        elif isinstance(msg, ToolMessage) and verbose:
            content = str(msg.content)
            if len(content) > 160:
                content = content[:157] + "..."
            print(f"  {DIM}[{node}] <- {msg.name}: {content}{RESET}")


def preflight(spec: str) -> int:
    """Verify the key works and the model id exists, before a real conversation."""
    provider, name = split_model(spec)
    print(f"{DIM}checking {spec} ...{RESET}")
    try:
        from .routines.base import get_model

        model = get_model(spec)
        reply = model.invoke("Reply with the single word: ok")
        text = getattr(reply, "text", None) or str(reply.content)
        print(f"  {BOLD}model{RESET}   {spec}")
        print(f"  {BOLD}reply{RESET}   {str(text).strip()[:60]}")
        usage = getattr(reply, "usage_metadata", None)
        if usage:
            print(f"  {BOLD}tokens{RESET}  in={usage.get('input_tokens')} "
                  f"out={usage.get('output_tokens')}")
        print(f"\n{BOLD}Ready.{RESET} Run: python run.py\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"{YELLOW}failed:{RESET} {type(exc).__name__}: {msg[:300]}\n")
        if "model" in msg.lower() and ("not" in msg.lower() or "404" in msg):
            label = ("Deployments on this Azure resource:" if provider == "azure"
                     else "Models this key can use:")
            print(f"That model id is not available on this key. {label}")
            for candidate in _list_models(provider):
                print(f"  {candidate}")
            noun = "deployment" if provider == "azure" else "model"
            print(f"\nPick one and re-run:  "
                  f"TLIFE_MODEL={provider}:<{noun}> python run.py --check")
        elif "api key" in msg.lower() or "401" in msg:
            print(f"The key in {required_key_env(spec)} was rejected.")
        return 1


def _list_models(provider: str, limit: int = 15) -> list[str]:
    """Best-effort listing of usable chat models for the provider."""
    try:
        if provider == "azure":
            # An Azure resource's /models endpoint returns the whole catalogue,
            # most of which is NOT deployed. Only /openai/deployments tells you
            # what this resource can actually serve, and the deployment name is
            # what you pass as the model.
            import json
            import urllib.request

            from ..config import AZURE_API_KEY, AZURE_ENDPOINT

            root = AZURE_ENDPOINT.rstrip("/").split("/openai/")[0]
            url = f"{root}/openai/deployments?api-version=2023-03-15-preview"
            req = urllib.request.Request(url, headers={"api-key": AZURE_API_KEY})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            return [
                f"{d['id']}  {DIM}(model {d.get('model')}, {d.get('status')}){RESET}"
                for d in data.get("data", [])
            ] or ["(no deployments on this resource -- create one in the portal)"]

        if provider == "openai":
            from openai import OpenAI

            ids = sorted(m.id for m in OpenAI().models.list().data)
            chat = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4"))
                    and not any(x in i for x in ("audio", "realtime", "transcribe",
                                                 "tts", "image", "embedding",
                                                 "moderation", "search"))]
            return chat[:limit] or ids[:limit]
    except Exception:  # noqa: BLE001
        pass
    return ["(could not list models -- check the provider console)"]


_OTP_SHOWN: dict[str, str] = {}


def _render_otp(session: str) -> None:
    """Deliver an outstanding one-time code to the *user*, not the agent.

    In a real deployment this arrives by SMS. The agent never sees it -- that
    is the whole point of a second factor -- so the CLI plays the part of the
    customer's phone.
    """
    code = db.OTP_STORE.get(session)
    if not code or _OTP_SHOWN.get(session) == code:
        return
    _OTP_SHOWN[session] = code
    print(f"{CYAN}  [your phone]{RESET} {BOLD}{code}{RESET} "
          f"{DIM}— one-time code. Type it to the agent to authorise the "
          f"action.{RESET}\n")


def _render_violations(session: str) -> None:
    """Print every guardrail that bit during this turn, with what it caught."""
    new = gv.drain(session)
    if not new:
        return
    worst = min(new, key=lambda v: ["critical", "high", "medium", "low"].index(v.severity))
    colour = SEVERITY_COLOR.get(worst.severity, YELLOW)
    print(f"{colour}{BOLD}  GUARDRAIL VIOLATED  {RESET}"
          f"{colour}{len(new)} triggered this turn{RESET}")
    for v in new:
        c = SEVERITY_COLOR.get(v.severity, YELLOW)
        mark = SEVERITY_MARK.get(v.severity, "*")
        print(f"  {c}{mark} [{v.severity.upper()}] {v.title}{RESET}")
        print(f"      {DIM}id      {v.violation_id}   clause {v.clause_id}   "
              f"desk {v.routine}{RESET}")
        print(f"      {DIM}caught  {v.source} -> {v.action}{RESET}")
        print(f"      {v.detail}")
        if v.evidence:
            print(f"      {DIM}evidence: {v.evidence}{RESET}")
    print()


def _render_timing(elapsed: float) -> None:
    stats = _TURN_STATS
    tokens = stats["input_tokens"] + stats["output_tokens"]
    parts = [f"{elapsed:.2f}s"]
    if stats["model_calls"]:
        parts.append(f"{stats['model_calls']} model call"
                     f"{'s' if stats['model_calls'] != 1 else ''}")
    if stats["tool_calls"]:
        parts.append(f"{stats['tool_calls']} tool call"
                     f"{'s' if stats['tool_calls'] != 1 else ''}")
    if tokens:
        parts.append(f"{tokens:,} tokens "
                     f"({stats['input_tokens']:,} in / {stats['output_tokens']:,} out)")
    print(f"{GREY}  {' · '.join(parts)}{RESET}\n")


def run_once(graph, session: str, text: str, verbose: bool) -> None:
    config = {"configurable": {"thread_id": session}, "recursion_limit": 60}
    for key in _TURN_STATS:
        _TURN_STATS[key] = 0
    started = time.perf_counter()
    try:
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config=config,
            stream_mode="updates",
        ):
            for node, payload in chunk.items():
                if isinstance(payload, dict):
                    _render_update(node, payload, verbose)
    except Exception as exc:  # noqa: BLE001 - a REPL should not die on one bad turn
        print(f"{YELLOW}[error]{RESET} {type(exc).__name__}: {exc}\n")
    elapsed = time.perf_counter() - started
    _render_otp(session)
    _render_violations(session)
    _render_timing(elapsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{BRAND} customer service agent")
    parser.add_argument("--session", default="demo", help="conversation / thread id")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the guardrail seed to make a run reproducible")
    parser.add_argument("--model", default=None, help="override the model string")
    parser.add_argument("--message", default=None,
                        help="send one message and exit (non-interactive)")
    parser.add_argument("--quiet", action="store_true",
                        help="hide tool calls and results")
    parser.add_argument("--check", action="store_true",
                        help="verify the API key and model id, then exit")
    parser.add_argument("--guardrails", default=None, choices=["llm", "jev"],
                        help="guardrail backend: 'llm' puts the rules in the "
                             "prompt and judges with a chat model; 'jev' asks "
                             "TypeSafe typed questions and leaves the prompt "
                             "clean (default: TLIFE_GUARDRAIL_BACKEND, else llm)")
    args = parser.parse_args(argv)

    if args.seed is not None:
        os.environ["TLIFE_GUARDRAIL_SEED"] = str(args.seed)
        # config was read at import time; rebuild the sampler with the new seed
        from .config import GUARDRAIL_CONFIG
        object.__setattr__(GUARDRAIL_CONFIG, "seed", args.seed)
        reset_sampler(args.session)

    if args.guardrails:
        os.environ["TLIFE_GUARDRAIL_BACKEND"] = args.guardrails
        try:
            from .guardrails.backends import set_backend

            set_backend(args.guardrails)
        except Exception as exc:  # noqa: BLE001
            print(f"{YELLOW}guardrail backend {args.guardrails!r} unavailable:"
                  f"{RESET} {type(exc).__name__}: {str(exc)[:200]}")
            if args.guardrails == "jev":
                print("  pip install langchain-typesafe   and set TYPESAFE_API_KEY")
            return 1

    spec = args.model or DEFAULT_MODEL
    missing = missing_key(spec)
    if missing:
        provider, name = split_model(spec)
        print(f"{YELLOW}{missing} is not set.{RESET} The configured model is "
              f"{BOLD}{spec}{RESET} (provider={provider}).\n"
              f"  export {missing}=...            then re-run\n"
              f"  or point TLIFE_MODEL at another provider\n"
              f"  or inspect the guardrails with no key at all: "
              f"python demo_guardrails.py")
        return 1

    if args.check:
        return preflight(spec)

    set_fallback_session(args.session)
    graph = build_care_graph(args.model)
    seed = sampler_for(args.session).session_seed
    verbose = not args.quiet

    if args.message:
        run_once(graph, args.session, args.message, verbose)
        return 0

    _banner(args.session, seed)
    _print_demo_accounts()

    pending: list[str] = []

    while True:
        if pending:
            # A pasted block arrives as several buffered lines. Echo each one as
            # it is actually processed -- otherwise the terminal shows all of
            # them at the first prompt and every turn after looks off by one.
            text = pending.pop(0)
            print(f"{CYAN}you>{RESET} {text}  {DIM}(queued){RESET}")
        else:
            try:
                text = input(f"{CYAN}you>{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            pending.extend(_drain_stdin())
        if not text:
            continue
        low = text.lower()
        if low in {"/quit", "/exit", "/q"}:
            return 0
        if low == "/help":
            print(__doc__)
            continue
        if low == "/trace":
            _print_trace(args.session); continue
        if low == "/coverage":
            _print_coverage(args.session); continue
        if low == "/audit":
            _print_audit(args.session); continue
        if low == "/session":
            _print_session(args.session); continue
        if low in {"/violations", "/v"}:
            _print_violations(args.session); continue
        if low == "/whoami":
            _print_demo_accounts(); continue
        if low == "/reset":
            db.reset_session(args.session)
            reset_sampler(args.session)
            reset_runtime(args.session)
            gv.reset(args.session)
            _RENDERED.clear()
            graph = build_care_graph(args.model)
            print(f"{DIM}session reset{RESET}\n")
            continue
        run_once(graph, args.session, text, verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
