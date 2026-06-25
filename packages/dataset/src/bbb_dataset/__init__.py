"""BBB peptide dataset pipeline.

Import submodules directly, e.g. ``from bbb_dataset.struct_io import load_coords_npz``.
"""

from bbb_dataset.aa import BOLTZ_CANONICAL, CANONICAL_AA, CANONICAL_AA_STR, THREE_TO_ONE

__all__ = [
    "BOLTZ_CANONICAL",
    "CANONICAL_AA",
    "CANONICAL_AA_STR",
    "THREE_TO_ONE",
]
