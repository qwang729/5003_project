#!/usr/bin/env python
# coding: utf-8
"""
Edgelist generation script for EmbDI
- Converts CSV datasets to edgelist format required by EmbDI
- Supports directed/undirected edges with weights
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os

def generate_edgelist(input_csv, output_edgelist, prefix_config='3#__tn,3$__tt,5$__idx,1$__cid'):
    """
    Generate edgelist from CSV dataset
    :param input_csv: Path to input CSV file
    :param output_edgelist: Path to output edgelist file
    :param prefix_config: Node prefix configuration (per EmbDI specs)
    """
    # Load CSV dataset
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded dataset: {input_csv} (rows: {len(df)}, columns: {len(df.columns)})")
    except FileNotFoundError:
        print(f"Error: Input file {input_csv} not found")
        sys.exit(1)

    # Initialize edgelist data
    edgelist = []
    # Write prefix config as first line
    edgelist.append(prefix_config)

    # Generate edges for each row (RID) and column (CID)
    for rid, row in df.iterrows():
        rid_node = f'idx__{rid}'  # Record ID node (idx__[row_num])
        for cid, (col_name, value) in enumerate(row.items()):
            # Skip NaN values
            if pd.isna(value):
                continue
            # Column ID node (tt__[col_name] for text, tn__ for numeric)
            if isinstance(value, (int, float)):
                cid_node = f'tn__{col_name}'
                val_node = f'tn__{value}'
            else:
                cid_node = f'tt__{col_name}'
                val_node = f'tt__{str(value).replace(" ", "_")}'  # Replace spaces for node ID

            # Add edges: RID <-> ValNode, CID <-> ValNode (undirected, weight=1)
            edgelist.append(f'{rid_node},{val_node},1,1')
            edgelist.append(f'{cid_node},{val_node},1,1')

    # Write edgelist to file
    with open(output_edgelist, 'w') as f:
        f.write('\n'.join(edgelist))
    print(f"Edgelist generated: {output_edgelist} (total edges: {len(edgelist)-1})")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate EmbDI edgelist from CSV')
    parser.add_argument('-i', '--input_file', required=True, help='Path to input CSV file')
    parser.add_argument('-o', '--output_file', required=True, help='Path to output edgelist file')
    parser.add_argument('-p', '--prefix', default='3#__tn,3$__tt,5$__idx,1$__cid', help='Node prefix configuration')
    args = parser.parse_args()

    # Generate edgelist
    generate_edgelist(args.input_file, args.output_file, args.prefix)