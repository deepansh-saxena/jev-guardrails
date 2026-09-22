"""Runtime configuration for the T-Life style customer service agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from the project root, without clobbering real env vars."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# Default provider is Azure OpenAI, pointed at the resource's v1 surface.
# The part after the colon is the *deployment name*, not the base model name --
# on Azure those are frequently different, and a mismatch is a 404
# DeploymentNotFound. `python run.py --check` lists the real deployments.
#
#   TLIFE_MODEL="azure:gpt-5.4-mini"      (default)
#   TLIFE_MODEL="openai:gpt-4.1"          public OpenAI, needs OPENAI_API_KEY
#   TLIFE_MODEL="anthropic:claude-opus-5" needs langchain-anthropic
DEFAULT_MODEL = os.environ.get("TLIFE_MODEL", "azure:gpt-5.4-mini")

# Azure resource. AZURE_OPENAI_ENDPOINT should be the `/openai/v1` form, which
# is OpenAI-wire-compatible -- so the plain ChatOpenAI client works against it
# and no api-version juggling is needed.
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")

# A cheaper model for the narrow, high-volume routines. Triage stays on the
# main model because a bad route is the most expensive mistake in the system.
DEFAULT_ROUTINE_MODEL = os.environ.get("TLIFE_ROUTINE_MODEL", DEFAULT_MODEL)

BRAND = os.environ.get("TLIFE_BRAND", "T-Life")

# Which env var holds the key, per `init_chat_model` provider prefix.
PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_APPLICATION_CREDENTIALS",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "cohere": "COHERE_API_KEY",
    "ollama": "",  # local, no key
}


def split_model(spec: str | None = None) -> tuple[str, str]:
    """('azure:gpt-5.4-mini') -> ('azure', 'gpt-5.4-mini').

    For the azure provider the second half is the DEPLOYMENT name.
    """
    spec = spec or DEFAULT_MODEL
    if ":" in spec:
        provider, _, name = spec.partition(":")
        return provider, name
    return "openai", spec


def required_key_env(spec: str | None = None) -> str:
    """Name of the env var that must hold the API key for this model, or ''."""
    provider, _ = split_model(spec)
    return PROVIDER_KEY_ENV.get(provider, "")


def missing_key(spec: str | None = None) -> str | None:
    """Return the env var name if a credential for this model is missing."""
    provider, _ = split_model(spec)
    if provider == "azure" and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "AZURE_OPENAI_ENDPOINT"
    env = required_key_env(spec)
    if env and not os.environ.get(env):
        return env
    return None


@dataclass(frozen=True)
class GuardrailConfig:
    """Knobs for the stochastic guardrail sampler.

    Nothing here is a hard policy value -- these are the *distributions* the
    sampler draws from. The point is that two consecutive turns never see an
    identical rulebook, so the model cannot cache, pattern-match or be
    social-engineered against one fixed phrasing.
    """

    # How many non-pinned clauses get drawn into a single prompt. The pool is
    # ~60 deep, so a single turn sees roughly a fifth of it -- enough coverage
    # to be meaningful, sparse enough that the rulebook genuinely moves.
    min_sampled: int = 8
    max_sampled: int = 16

    # Probability a drawn clause is rendered with an emphasis decorator.
    emphasis_rate: float = 0.35

    # Probability of injecting a self-audit directive for this turn.
    self_audit_rate: float = 0.30

    # Probability of injecting a "this turn is being QA sampled" directive.
    qa_sample_rate: float = 0.15

    # Probability of injecting an adversarial-input reminder.
    injection_probe_rate: float = 0.25

    # Strictness dial drawn per turn; scales numeric thresholds in clause text.
    strictness_range: tuple[float, float] = (0.55, 1.0)

    # How many soft (judgment-based) rules are drawn into a single output
    # review. The rubric is ~25 deep; judging all of it on every reply would
    # cost more than the reply did, so QA samples it -- same as a real
    # call-quality programme.
    soft_review_min: int = 4
    soft_review_max: int = 8

    # Runtime (tool-level) stochastic controls.
    step_up_auth_rate: float = 0.22       # random OTP challenge on risky actions
    manual_review_rate: float = 0.12      # random hold-for-review on writes
    credit_ceiling_range: tuple[int, int] = (15, 50)  # jittered auto-approve cap

    # Set TLIFE_GUARDRAIL_SEED to make a whole session reproducible.
    seed: int | None = field(
        default_factory=lambda: (
            int(os.environ["TLIFE_GUARDRAIL_SEED"])
            if os.environ.get("TLIFE_GUARDRAIL_SEED")
            else None
        )
    )


GUARDRAIL_CONFIG = GuardrailConfig()
