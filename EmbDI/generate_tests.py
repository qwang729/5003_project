#!/usr/bin/env python
# coding: utf-8
"""
EQ (Embeddings Quality) test generation script for EmbDI
- Generates MA (match attributes), MR (match rows), MC (match columns) tests
- Based on the specifications in the EmbDI paper and README
"""

import argparse
import pandas as pd
import numpy as np
import random
import os
import sys

def generate_mr_tests(df, output_dir, n_tests=1000, attrs_to_change=None):
    """
    Generate MR (no-match-rows) tests
    :param df: Input DataFrame
    :param output_dir: Output directory for test files
    :param n_tests: Number of tests to generate
    :param attrs_to_change: List of attributes to modify for tests
    """
    if attrs_to_change is None:
        attrs_to_change = df.columns.tolist()
    os.makedirs(os.path.join(output_dir, 'mr'), exist_ok=True)
    test_file = os.path.join(output_dir, 'mr', 'mr_tests.txt')

    with open(test_file, 'w') as f:
        for _ in range(n_tests):
            # Randomly select a row and attribute
            target_row = df.sample(n=1).iloc[0]
            target_attr = random.choice(attrs_to_change)
            target_val = target_row[target_attr]
            # Randomly select a replacement value from another row
            other_vals = df[df[target_attr] != target_val][target_attr].dropna().unique()
            if len(other_vals) == 0:
                continue
            replacement_val = random.choice(other_vals)
            # Build test case: original row values + replacement value
            test_vals = [str(target_row[col]) if col != target_attr else str(replacement_val) for col in df.columns]
            # Write test case (last value is the incorrect one)
            f.write(','.join(test_vals) + f',{target_attr}\n')
    print(f"MR tests generated: {test_file} ({n_tests} tests)")

def generate_ma_tests(df, output_dir, n_tests=1000, col_combinations=None):
    """
    Generate MA (no-match-column) tests
    :param df: Input DataFrame
    :param output_dir: Output directory for test files
    :param n_tests: Number of tests to generate
    :param col_combinations: List of column pairs for tests
    """
    if col_combinations is None:
        col_combinations = [(df.columns[i], df.columns[j]) for i in range(len(df.columns)) for j in range(i+1, len(df.columns))]
    os.makedirs(os.path.join(output_dir, 'ma'), exist_ok=True)
    test_file = os.path.join(output_dir, 'ma', 'ma_tests.txt')

    with open(test_file, 'w') as f:
        for _ in range(n_tests):
            # Randomly select a column pair
            col1, col2 = random.choice(col_combinations)
            # Select random values from col1 and one value from col2
            val1_list = df[col1].dropna().unique()
            val2 = df[col2].dropna().sample(n=1).iloc[0]
            if len(val1_list) < 4:
                continue
            # Select 4 values from col1 + 1 from col2
            test_vals = random.sample(list(val1_list), 4) + [val2]
                        # Shuffle test values and record the incorrect one
            random.shuffle(test_vals)
            incorrect_val = str(val2)
            # Write test case: values + incorrect value
            f.write(','.join(map(str, test_vals)) + f',{incorrect_val}\n')
    print(f"MA tests generated: {test_file} ({n_tests} tests)")

def generate_mc_tests(df, output_dir, fd_pairs, test_length=5, n_tests=1000):
    """
    Generate MC (no-match-concept) tests (based on functional dependencies)
    :param df: Input DataFrame
    :param output_dir: Output directory for test files
    :param fd_pairs: List of (A1, A2) pairs (A1 -> A2 functional dependency)
    :param test_length: Number of related values per test (test_length-1 related + 1 unrelated)
    :param n_tests: Number of tests to generate
    """
    os.makedirs(os.path.join(output_dir, 'mc'), exist_ok=True)
    test_file = os.path.join(output_dir, 'mc', 'mc_tests.txt')

    with open(test_file, 'w') as f:
        for (a1, a2) in fd_pairs:
            # Filter rows with non-null values in both attributes
            valid_df = df[[a1, a2]].dropna()
            if len(valid_df) < test_length:
                continue
            # Group A2 values by A1
            a1_groups = valid_df.groupby(a1)[a2].agg(lambda x: list(set(x))).to_dict()
            # Keep A1 values with at least `test_length` distinct A2 values
            eligible_a1 = [x for x, y in a1_groups.items() if len(y) >= test_length]
            if len(eligible_a1) == 0:
                continue
            # Generate tests for eligible A1 values
            for _ in range(n_tests // len(fd_pairs)):
                target_a1 = random.choice(eligible_a1)
                related_a2 = random.sample(a1_groups[target_a1], test_length - 1)
                # Get unrelated A2 value (not in target_a1's group)
                all_a2 = set(valid_df[a2].unique())
                unrelated_a2 = random.choice(list(all_a2 - set(a1_groups[target_a1])))
                # Combine and shuffle test values
                test_vals = related_a2 + [unrelated_a2]
                random.shuffle(test_vals)
                # Write test case: values + target A1 + unrelated value
                f.write(','.join(map(str, test_vals)) + f',{target_a1},{unrelated_a2}\n')
    print(f"MC tests generated: {test_file} ({n_tests} tests)")

def main():
    """Main function to generate EQ tests"""
    parser = argparse.ArgumentParser(description='Generate EQ tests for EmbDI')
    parser.add_argument('-i', '--input_file', required=True, help='Path to input CSV dataset (pipeline/datasets/...)')
    parser.add_argument('-o', '--output_dir', default='pipeline/test_dir', help='Output directory for tests')
    parser.add_argument('--n_tests', type=int, default=1000, help='Number of tests per EQ type')
    parser.add_argument('--mr_attrs', nargs='+', help='Attributes to modify for MR tests (default: all columns)')
    parser.add_argument('--ma_cols', nargs='+', help='Column pairs for MA tests (format: col1,col2 col3,col4)')
    parser.add_argument('--fd_pairs', nargs='+', help='FD pairs for MC tests (format: a1,a2 a3,a4)')
    args = parser.parse_args()

    # Load dataset
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found")
        sys.exit(1)

    # Process input parameters
    mr_attrs = args.mr_attrs if args.mr_attrs else df.columns.tolist()
    ma_cols = [pair.split(',') for pair in args.ma_cols] if args.ma_cols else None
    fd_pairs = [pair.split(',') for pair in args.fd_pairs] if args.fd_pairs else [('director', 'movie_title')]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate tests
    print("Generating MR (match rows) tests...")
    generate_mr_tests(df, args.output_dir, args.n_tests, mr_attrs)
    
    print("Generating MA (match attributes) tests...")
    generate_ma_tests(df, args.output_dir, args.n_tests, ma_cols)
    
    print("Generating MC (match concepts) tests...")
    generate_mc_tests(df, args.output_dir, fd_pairs, test_length=5, n_tests=args.n_tests)

    print(f"All EQ tests generated in: {args.output_dir}")

if __name__ == "__main__":
    main()