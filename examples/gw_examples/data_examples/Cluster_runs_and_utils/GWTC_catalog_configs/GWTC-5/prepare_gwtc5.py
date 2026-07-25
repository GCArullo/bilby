#!/usr/bin/env python3

"""Extract GWTC-5 XPHM-SpinTaylor settings as nocosmo run templates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GWTC4_PATH = ROOT.parent / "GWTC-4" / "prepare_gwtc4.py"
SPEC = importlib.util.spec_from_file_location("prepare_gwtc4", GWTC4_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {GWTC4_PATH}")
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)

catalog.ROOT = ROOT
catalog.RECORD_IDS = (20348005, 20348006)
catalog.CATALOG_NAME = "GWTC-5"
catalog.DOCUMENTATION_URL = "https://gwosc.org/GWTC-5.0/"
catalog.configure()


if __name__ == "__main__":
    raise SystemExit(catalog.common.main())
