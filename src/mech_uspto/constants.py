"""Chemistry constants used by featurization.

Kept in a single module so node/edge feature dimensions are derivable from
one source of truth.
"""

from rdkit import Chem

ALLOWED_ELEMENTS: list[int] = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

ALLOWED_HYBRIDIZATIONS: list[Chem.rdchem.HybridizationType] = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]

ALLOWED_CHIRAL: list[Chem.rdchem.ChiralType] = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

# Single, double, triple, aromatic (encoded as 4.0).
ALLOWED_BOND_TYPES: list[float] = [1.0, 2.0, 3.0, 4.0]

# Resulting feature widths (kept here so downstream code does not hardcode them).
NODE_FEATURE_DIM: int = (
    len(ALLOWED_ELEMENTS) + 1            # element one-hot + unknown
    + 5                                   # degree, charge, aromatic, num_hs, in_ring
    + len(ALLOWED_HYBRIDIZATIONS) + 1     # hybridization one-hot + unknown
    + len(ALLOWED_CHIRAL) + 1             # chirality one-hot + unknown
)  # = 25

EDGE_FEATURE_DIM: int = len(ALLOWED_BOND_TYPES) + 1 + 1  # bond-type one-hot + unknown + in_ring  # = 6
