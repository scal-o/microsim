from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from calibration import utils


START_OD_FILES = [
    "Start_od_7_8.txt",
    "Start_od_8_9.txt",
    "Start_od_9_10.txt",
]

@click.group("pca")
def pca_cli():
    pass

def _load_start_od_from_data_dir(data_dir: Path, file_name: str) -> pd.DataFrame:
    """
    Reuse utils.load_start_od-style loading by building temporary config/sim_setup dicts.
    Expected output columns are:
      0 -> origin zone
      1 -> destination zone
      2 -> OD value
    """
    config = {"NETWORK": data_dir}
    sim_setup = {"prior_od": file_name}
    od_df = utils.load_start_od(config, sim_setup)

    return od_df.iloc[:, :3].copy()


@pca_cli.command("generate-matrix")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to folder containing Start_od_7_8.txt, Start_od_8_9.txt, Start_od_9_10.txt.",
)
@click.option(
    "--samples-per-matrix",
    default=25,
    show_default=True,
    type=int,
    help="Number of generated samples for each starting OD matrix.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--output-file",
    default="od_pca_matrix.csv",
    show_default=True,
    type=str,
    help="Output filename written inside --data-dir.",
)
def main(
    data_dir: Path,
    samples_per_matrix: int,
    seed: int,
    output_file: str,
) -> None:
    """
    Build a PCA-ready matrix from Start_od files.

    Each row is one flattened generated OD vector, computed as:
        od = (0.75 + 0.15*d) * od_true
    where d ~ N(0, (1/3)^2), element-wise.
    """

    rng = np.random.default_rng(seed)
    generated_rows: list[np.ndarray] = []
    expected_len: int | None = None

    for file_name in START_OD_FILES:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        od_df = _load_start_od_from_data_dir(data_dir, file_name)
        od_true = od_df.iloc[:, 2].to_numpy(dtype=float)  # flattened OD vector as stored in file

        if expected_len is None:
            expected_len = od_true.shape[0]
        elif od_true.shape[0] != expected_len:
            raise ValueError(
                f"Inconsistent OD vector length in {file_name}: {od_true.shape[0]} vs expected {expected_len}"
            )

        for _ in range(samples_per_matrix):
            d = rng.normal(loc=0.0, scale=1.0 / 3.0, size=od_true.shape[0])
            od_sample = (0.75 + 0.15 * d) * od_true
            generated_rows.append(od_sample)

    matrix = np.vstack(generated_rows)
    out_path = data_dir / output_file

    # No header, no index, no extra metadata columns
    pd.DataFrame(matrix).to_csv(out_path, index=False, header=False, sep=";")

    click.echo(
        f"Wrote matrix to {out_path} with shape {matrix.shape} "
        f"(rows = 3 * {samples_per_matrix}, cols = {matrix.shape[1]})."
    )


if __name__ == "__main__":
    main()