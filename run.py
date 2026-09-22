#!/usr/bin/env python3
"""Entry point: python run.py [--session ID] [--seed N] [--guardrails llm|jev]"""

import sys
import warnings

# langgraph and langchain-typesafe emit pending-deprecation and beta notices on
# import, and they register their own "always" filters, so an ordinary
# filterwarnings call is overridden. Suppressing around the import is the only
# thing that reliably works. Run with `python -W default run.py` to see them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tlife_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
