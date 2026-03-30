from pathlib import Path
import argparse

import pandas as pd
from pymatgen.io.cif import CifParser


def extract_cif_data(cif_path: Path):
    fields = {
        '_database_code_ICSD': '_database_code_icsd',
        '_chemical_formula_sum': '_chemical_formula_sum',
        '_cell_measurement_temperature': '_cell_measurement_temperature',
        '_diffrn_ambient_temperature': '_diffrn_ambient_temperature',
        '_chemical_name_structure_type': '_chemical_name_structure_type',
        '_exptl_crystal_density_diffrn': '_exptl_crystal_density_diffrn',
        '_chemical_formula_weight': '_chemical_formula_weight',
        '_cell_length_a': '_cell_length_a',
        '_cell_length_b': '_cell_length_b',
        '_cell_length_c': '_cell_length_c',
        '_cell_angle_alpha': '_cell_angle_alpha',
        '_cell_angle_beta': '_cell_angle_beta',
        '_cell_angle_gamma': '_cell_angle_gamma',
        '_cell_volume': '_cell_volume',
        '_cell_formula_units_z': '_cell_formula_units_z',
        '_space_group_name_H-M_alt': '_symmetry_space_group_name_H-M',
        '_space_group_IT_number': '_space_group_IT_number',
        '_diffrn_ambient_pressure': '_diffrn_ambient_pressure',
    }

    data = {value: None for value in fields.values()}
    data['file_id'] = str(cif_path)
    data['valid_cif'] = False
    data['type'] = 'experimental'

    try:
        parser = CifParser(str(cif_path))
        cif_data = parser.as_dict()
        cif_key = list(cif_data.keys())[0]
        for cif_field, csv_field in fields.items():
            if cif_field in cif_data[cif_key]:
                data[csv_field] = cif_data[cif_key][cif_field]
        data['valid_cif'] = True
    except Exception as exc:
        print(f"Error processing {cif_path}: {exc}")

    return data


def process_cif_folder(cif_directory: Path, output_csv_path: Path):
    all_data = []
    cif_files = sorted(cif_directory.glob('*.cif'))

    for index, cif_file in enumerate(cif_files, start=1):
        print(f"Processing {cif_file} ({index}/{len(cif_files)})")
        all_data.append(extract_cif_data(cif_file))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, 'w') as handle:
        handle.write("# Extracted CIF data with correct column names and type field\n")
        pd.DataFrame(all_data).to_csv(handle, index=False)

    print(f"Processed CIF data saved to {output_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ICSD metadata from CIF files.')
    parser.add_argument(
        '--cif-dir',
        type=Path,
        default=Path('/3DSC/superconductors_3D/data/source/ICSD/raw/cifs'),
        help='Directory containing raw ICSD CIF files.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/3DSC/superconductors_3D/data/source/ICSD/raw/ICSD_subset.csv'),
        help='Output CSV path for extracted metadata.',
    )
    args = parser.parse_args()
    process_cif_folder(args.cif_dir, args.output)

