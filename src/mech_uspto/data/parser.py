"""Parser for mech-USPTO-31k CSV files (figshare format)."""

import csv
from io import StringIO
from pathlib import Path

from tqdm.auto import tqdm

from mech_uspto.data.schema import MultiStepReaction, ReactionStep


class MechUSPTOParser:
    """Parser for the mech-USPTO-31k CSV (figshare release).

    CSV columns (in order):
        1. original_reactions  — raw SMILES reaction from USPTO
        2. updated_reaction    — MechFinder atom-mapped reaction (reactants>>products)
        3. mechanistic_class   — string label (e.g. "Cbz_deprotection")
        4. mechanistic_label   — arrow-pushing tuples encoding elementary steps
        5. data_source         — source URL
    """

    @staticmethod
    def parse_csv_row(csv_line: str, row_index: int | None = None) -> MultiStepReaction:
        """Parse a single CSV row into a ``MultiStepReaction``.

        Note: each row is currently mapped to a single step. Decomposition
        into individual elementary steps via ``mechanistic_label`` is a TODO.
        """
        reader = csv.reader([csv_line])
        parts = next(reader)

        if len(parts) < 4:
            raise ValueError(f"CSV row must have at least 4 columns, got {len(parts)}")

        original_reactions = parts[0]
        updated_reaction = parts[1]
        mechanistic_class = parts[2]
        mechanistic_label = parts[3]

        reaction_parts = updated_reaction.split(">>")
        if len(reaction_parts) != 2:
            raise ValueError(
                f"Expected single '>>' in updated_reaction, got {len(reaction_parts) - 1}"
            )

        reactants_mapped, products_mapped = reaction_parts

        step = ReactionStep(
            step_id=0,
            reactants_smi=reactants_mapped,
            products_smi=products_mapped,
            reactants_mapped=reactants_mapped,
            products_mapped=products_mapped,
            mechanism_arrow=mechanistic_label,
        )

        if row_index is not None:
            rxn_id = f"rxn_{row_index:06d}"
        else:
            rxn_id = f"rxn_{hash(reactants_mapped + products_mapped) % 1_000_000:06d}"

        return MultiStepReaction(
            reaction_id=rxn_id,
            steps=[step],
            overall_reactants_smi=reactants_mapped,
            overall_products_smi=products_mapped,
            metadata={
                "mechanistic_class": mechanistic_class,
                "mechanistic_label": mechanistic_label,
                "original_reactions": original_reactions,
            },
        )

    @staticmethod
    def parse_csv_file(csv_path: str) -> list[MultiStepReaction]:
        """Parse every row of a mech-USPTO-31k CSV file.

        Malformed rows are logged and skipped.
        """
        reactions: list[MultiStepReaction] = []
        path = Path(csv_path)

        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return reactions

            for i, row in enumerate(tqdm(reader, desc=f"Parsing {path.name}")):
                buf = StringIO()
                csv.writer(buf).writerow(row)
                csv_line = buf.getvalue().rstrip("\r\n")

                try:
                    reactions.append(MechUSPTOParser.parse_csv_row(csv_line, row_index=i))
                except Exception as e:  # noqa: BLE001 - surface but skip bad rows
                    print(f"⚠️  Failed to parse row {i}: {e}")
                    continue

        return reactions
