"""Compute Δ (delta) bond-order matrices for stepwise / end-to-end modes."""

import torch
from rdkit import Chem


class DeltaMatrixGenerator:
    """Build delta matrices from molecule pairs or mechanism arrow strings."""

    @staticmethod
    def compute_adjacency_matrix(mol: Chem.Mol) -> torch.Tensor:
        """Return the bond-order adjacency matrix for ``mol``.

        The mol is kekulized (Hückel rings → alternating single/double) so
        every bond carries an integer order. RDKit's default representation
        marks aromatic bonds with order 1.5, which (a) makes Δ values
        non-integer when a leaving group with an aromatic ring is removed,
        and (b) forces ``apply_delta`` to invent a +2 (double-bond) edge
        when forming an aromatic bond. Kekulization avoids both pitfalls;
        aromaticity is re-perceived in ``apply_delta`` after sanitize.
        """
        num_atoms = mol.GetNumAtoms()
        adjacency = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

        try:
            kek = Chem.Mol(mol)
            Chem.Kekulize(kek, clearAromaticFlags=True)
            bond_mol = kek
        except Exception:
            bond_mol = mol  # fall back to original if kekulization fails

        for bond in bond_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_order = bond.GetBondTypeAsDouble()
            adjacency[i, j] = bond_order
            adjacency[j, i] = bond_order

        return adjacency

    @staticmethod
    def delta_from_reactants_products(
        reactants_mol: Chem.Mol,
        products_mol: Chem.Mol,
    ) -> torch.Tensor:
        """Compute Δ = A_products − A_reactants.

        Stepwise mode: result is constrained to {-1, 0, 1}.
        End-to-end mode: result may include {-2, -1, 0, 1, 2} for double-bond changes.

        Bonds internal to atoms that are present in only one side (e.g. leaving
        groups in R, by-products in P) are explicitly zeroed: the molecule
        on the other side has padded-zero adjacency there, which would
        otherwise spuriously emit ``Δ = -bond_order`` (e.g. ``-1.5`` for
        aromatic bonds in a Cbz/PPh₃ leaving group). Such bonds are
        irrelevant — those atoms float away as a disconnected fragment after
        the *cut* bond (shared↔non-shared) is broken; their internal
        connectivity carries no learnable signal and only confuses the model.
        """
        adjacency_reactants = DeltaMatrixGenerator.compute_adjacency_matrix(reactants_mol)
        adjacency_products = DeltaMatrixGenerator.compute_adjacency_matrix(products_mol)

        n_react = adjacency_reactants.shape[0]
        n_prod = adjacency_products.shape[0]

        # Pad smaller adjacency to the larger size when atom counts differ.
        if n_react != n_prod:
            max_n = max(n_react, n_prod)
            padded_react = torch.zeros((max_n, max_n), dtype=torch.float)
            padded_prod = torch.zeros((max_n, max_n), dtype=torch.float)
            padded_react[:n_react, :n_react] = adjacency_reactants
            padded_prod[:n_prod, :n_prod] = adjacency_products
            adjacency_reactants = padded_react
            adjacency_products = padded_prod

        delta = adjacency_products - adjacency_reactants

        # Mask out bonds internal to non-shared atom sets. After ``align_atoms``
        # both mols start with their shared (map-matched) atoms in identical
        # order, followed by side-specific atoms. The shared prefix length is
        # the intersection size of atom-map numbers.
        shared_maps = {a.GetAtomMapNum() for a in reactants_mol.GetAtoms()} & {
            a.GetAtomMapNum() for a in products_mol.GetAtoms()
        }
        shared_maps.discard(0)
        n_shared = len(shared_maps)
        if n_shared < delta.shape[0]:
            # Zero the lower-right block (both endpoints outside the shared
            # prefix → bond is purely R-internal or purely P-internal).
            delta[n_shared:, n_shared:] = 0

        # Bond changes are symmetric (A→B == B→A).
        delta = (delta + delta.T) / 2
        return delta

    @staticmethod
    def delta_from_mechanism_arrows(
        arrow_string: str,
        mol: Chem.Mol,
    ) -> torch.Tensor:
        """Parse a mechanism arrow string (e.g. ``"1*2=2*3"``) into a Δ matrix.

        Format: ``"1*2=; 3*4=5*6"`` means *break bond 1-2*, then *shift from
        3-4 to 3-6*. Atom indices reference RDKit atom map numbers.
        """
        num_nodes = mol.GetNumAtoms()
        delta = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        map_to_idx = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }

        if not isinstance(arrow_string, str) or not arrow_string.strip():
            return delta

        for arrow in arrow_string.split(";"):
            arrow = arrow.strip().replace("*", "").replace(" ", "")
            if "=" not in arrow:
                continue

            src_s, dst_s = arrow.split("=")
            src_maps = [int(m) for m in src_s.split(",") if m.isdigit()]
            dst_maps = [int(m) for m in dst_s.split(",") if m.isdigit()]

            # Bond breaking: "1,2=" → remove bond 1-2.
            if len(src_maps) == 2 and len(dst_maps) == 0:
                u, v = src_maps[0], src_maps[1]
                if u in map_to_idx and v in map_to_idx:
                    delta[map_to_idx[u], map_to_idx[v]] = -1
                    delta[map_to_idx[v], map_to_idx[u]] = -1

            # Bond formation: "=1,2" → form bond 1-2.
            elif len(src_maps) == 0 and len(dst_maps) == 2:
                u, v = dst_maps[0], dst_maps[1]
                if u in map_to_idx and v in map_to_idx:
                    delta[map_to_idx[u], map_to_idx[v]] = 1
                    delta[map_to_idx[v], map_to_idx[u]] = 1

            # Shift: "u,v=u,w" → break bond u-v and form bond u-w (shared pivot u).
            # Bug history: pre-fix this branch wrote +1 to (v, w) — a non-event
            # pair — and never recorded the actual break or form. Latent because
            # the training pipeline uses ``delta_from_reactants_products``; bites
            # only once arrow-derived supervision (rollout/stepwise) goes live.
            elif len(src_maps) == 2 and len(dst_maps) == 2:
                if src_maps[0] == dst_maps[0]:  # shared pivot
                    pivot = src_maps[0]
                    leaving = src_maps[1]
                    incoming = dst_maps[1]
                    if (
                        pivot in map_to_idx
                        and leaving in map_to_idx
                        and incoming in map_to_idx
                    ):
                        pi = map_to_idx[pivot]
                        li = map_to_idx[leaving]
                        ii = map_to_idx[incoming]
                        # Break pivot–leaving.
                        delta[pi, li] = -1
                        delta[li, pi] = -1
                        # Form pivot–incoming.
                        delta[pi, ii] = 1
                        delta[ii, pi] = 1

        return delta
