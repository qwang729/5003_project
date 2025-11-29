#!/usr/bin/env python
# coding: utf-8
"""
Main script for EmbDI pipeline
- Supports train, test, match, train-test, train-match tasks
- Parses command line arguments (-d for config directory, -f for single config file)
- Executes graph construction, random walks, embedding training, and integration tasks
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import argparse
import os
import sys
import datetime as dt
import time
import pandas as pd
from EmbDI.utils import (
    load_config, validate_config, load_edgelist, build_graph, execute_walks_generation,
    learn_embeddings
)
from EmbDI.entity_resolution import entity_resolution_pipeline

# Required constants
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
OUTPUT_FORMAT = '# {:.<60} {}'

def setup_directories(config):
    """Create required pipeline directories (if not exist)"""
    required_dirs = [
        'pipeline/dump', 'pipeline/embeddings', 'pipeline/walks',
        'pipeline/generated-matches', 'pipeline/results'
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    # Create output embedding path
    emb_output_dir = 'pipeline/embeddings'
    return os.path.join(emb_output_dir, f"{config['output_file']}.emb")

def run_pipeline(config_path):
    """Run full EmbDI pipeline for a single config file"""
    pipeline_start = dt.datetime.now()
    print(OUTPUT_FORMAT.format('Starting EmbDI pipeline', pipeline_start.strftime(TIME_FORMAT)))
    print(OUTPUT_FORMAT.format('Loading configuration', config_path))

    # Step 1: Load and validate configuration
    config = load_config(config_path)
    config = validate_config(config)
    print(f"# Validated task: {config['task']} (experiment type: {config['experiment_type']})")

    # Step 2: Setup output directories and paths
    emb_output_path = setup_directories(config)
    print(OUTPUT_FORMAT.format('Embedding output path', emb_output_path))

    # Step 3: Load edgelist and prefixes
    print(OUTPUT_FORMAT.format('Loading edgelist', config['input_file']))
    if not os.path.exists(config['input_file']):
        print(f"Error: Edgelist file {config['input_file']} not found")
        return
    prefixes, edgelist = load_edgelist(config['input_file'])
    print(f"# Loaded {len(edgelist)} edges, prefixes: {prefixes}")

    # Step 4: Build heterogeneous graph
    print(OUTPUT_FORMAT.format('Starting graph construction', dt.datetime.now().strftime(TIME_FORMAT)))
    try:
        graph = build_graph(config, edgelist, prefixes)
    except Exception as e:
        print(f"Error building graph: {e}")
        return

    # Step 5: Generate random walks
    print(OUTPUT_FORMAT.format('Starting random walks generation', dt.datetime.now().strftime(TIME_FORMAT)))
    try:
        walks = execute_walks_generation(config, graph)
    except Exception as e:
        print(f"Error generating random walks: {e}")
        return
    print(f"# Random walks saved to: {walks} (write_walks={config['write_walks']})")

    # Step 6: Train embeddings (if task requires training)
    if config['task'] in ['train', 'train-test', 'train-match']:
        print(OUTPUT_FORMAT.format('Starting embedding training', dt.datetime.now().strftime(TIME_FORMAT)))
        try:
            learn_embeddings(
                output_emb_file=emb_output_path,
                walk_data=walks,
                save_walks=config['write_walks'],
                embed_dim=int(config['n_dimensions']),
                window=int(config['window_size']),
                training_algo=config['training_algorithm'],
                learn_mode=config['learning_method'],
                sample_factor=float(config['sampling_factor'])
            )
        except Exception as e:
            print(f"Error training embeddings: {e}")
            return
        if not os.path.exists(emb_output_path):
            print(f"Error: Embedding file not generated at {emb_output_path}")
            return

    # Step 7: Execute post-processing (test/match)
    if config['task'] in ['test', 'train-test']:
        print(OUTPUT_FORMAT.format('Starting ER test pipeline', dt.datetime.now().strftime(TIME_FORMAT)))
        # Validate required parameters for ER test
        required_params = ['match_file', 'dataset_info', 'experiment_type']
        for param in required_params:
            if param not in config or not config[param]:
                print(f"Error: Missing required parameter for test: {param}")
                return
        if config['experiment_type'] != 'ER':
            print(f"Warning: Experiment type {config['experiment_type']} not supported (only ER for now)")
            return
        # Run ER pipeline
        try:
            er_results = entity_resolution_pipeline(
                input_embedding=emb_output_path,
                config=config,
                task_type='test',
                dataset_info_path=config['dataset_info']
            )
            # Save ER results
            result_file = f"pipeline/results/{config['output_file']}_er_results.txt"
            with open(result_file, 'w') as f:
                f.write(f"EmbDI ER Results (task: {config['task']})\n")
                f.write(f"Timestamp: {dt.datetime.now().strftime(TIME_FORMAT)}\n")
                f.write(f"Dataset: {config['dataset_file']}\n")
                f.write("\nMetrics:\n")
                for key, val in er_results.items():
                    f.write(f"{key}: {val:.4f}\n")
            print(OUTPUT_FORMAT.format('ER results saved to', result_file))
        except Exception as e:
            print(f"Error running ER test: {e}")
            return

    elif config['task'] in ['match', 'train-match']:
        print(OUTPUT_FORMAT.format('Starting ER match generation', dt.datetime.now().strftime(TIME_FORMAT)))
        try:
            matches = entity_resolution_pipeline(
                input_embedding=emb_output_path,
                config=config,
                task_type='match',
                dataset_info_path=config['dataset_info']
            )
            # Save matches
            match_file = f"pipeline/generated-matches/{config['output_file']}_matches.txt"
            with open(match_file, 'w') as f:
                f.write('\n'.join(','.join(pair) for pair in matches))
            print(OUTPUT_FORMAT.format('Generated matches saved to', match_file))
        except Exception as e:
            print(f"Error generating ER matches: {e}")
            return

    # Step 8: Finalize pipeline
    pipeline_end = dt.datetime.now()
    total_time = (pipeline_end - pipeline_start).total_seconds()
    print(OUTPUT_FORMAT.format('EmbDI pipeline completed', pipeline_end.strftime(TIME_FORMAT)))
    print(OUTPUT_FORMAT.format('Total execution time', f'{total_time:.2f} seconds'))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EmbDI Pipeline Execution')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--config-dir', help='Path to directory with config files (runs all valid files)')
    group.add_argument('-f', '--config-file', help='Path to single config file')
    return parser.parse_args()

def get_valid_config_files(config_dir):
    """Get valid config files from directory (exclude dirs, start with non-'default')"""
    valid_files = []
    for fname in os.listdir(config_dir):
        fpath = os.path.join(config_dir, fname)
        if os.path.isfile(fpath) and not fname.startswith('default'):
            valid_files.append(fpath)
    # Sort alphabetically
    valid_files.sort()
    return valid_files

def main():
    args = parse_args()
    if args.config_file:
        # Run single config file
        if not os.path.exists(args.config_file):
            print(f"Error: Config file {args.config_file} not found")
            sys.exit(1)
        run_pipeline(args.config_file)
    elif args.config_dir:
        # Run all valid config files in directory
        if not os.path.isdir(args.config_dir):
            print(f"Error: Config directory {args.config_dir} not found")
            sys.exit(1)
        valid_files = get_valid_config_files(args.config_dir)
        if len(valid_files) == 0:
            print(f"Warning: No valid config files found in {args.config_dir}")
            sys.exit(0)
        print(f"# Found {len(valid_files)} valid config files. Running sequentially...")
        for fpath in valid_files:
            print(f"\n{'='*60}")
            print(f"Running pipeline for: {fpath}")
            print(f"{'='*60}")
            run_pipeline(fpath)
    print("\nAll pipeline executions completed.")

if __name__ == "__main__":
    main()