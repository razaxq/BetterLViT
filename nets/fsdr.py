"""FSDR: Frequency-aware Semantic-Detail Refinement.

The formal name of FAM-EPPA V4-B, adopted on 2026-09-14. Re-export the
original class so historical imports, pickles and state_dict keys stay valid.
"""

from .eppa import FAMAdaptiveHaarEPPA as FSDR

__all__ = ["FSDR"]
