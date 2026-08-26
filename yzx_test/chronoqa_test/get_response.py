"""Compatibility entry point for the role-fit response stage."""

import sys

from evaluate_agent_fit import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "respond"])
    main()
