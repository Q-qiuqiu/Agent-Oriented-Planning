#!/usr/bin/env python3
"""Entry point for the read-only ordered First-3 analysis.

The compact schema-v2 Cross-Block reader lives beside the experiment that
created those snapshots. Keeping this entry point in ``llada_server`` makes the
analysis available from the requested production-code directory without
copying or changing any decoder implementation.
"""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent.parent
    / "llada_server_dual"
    / "analyze_ordered_future_fields.py"
)
runpy.run_path(str(TARGET), run_name="__main__")
