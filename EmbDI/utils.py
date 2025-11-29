#!/usr/bin/env python
# coding: utf-8
"""
Utility classes and functions for EmbDI
- Node: Represents a node in the heterogeneous graph
- Graph: Builds the tripartite graph from edgelist
- RandomWalk: Generates random walks on the graph
- Config handling: Load/validate configuration files
- Graph construction: Build graph from edgelist
- Random walks generation: Create walks for embedding training
"""

import numpy as np
import networkx as nx
import pandas as pd
import warnings
import datetime as dt
import time
import random
import math
from tqdm import tqdm
import gensim.models as models
import multiprocessing as mp
from gensim.models import Word2Vec, FastText
from sklearn.neighbors import NearestNeighbors
import sys
import os
from io import StringIO
import argparse
import pickle
import csv

# Required constants
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
OUTPUT_FORMAT = '# {:.<60} {}'

# Verify NetworkX availability
try:
    import networkx as nx
    NX_NOT_FOUND = False
except ImportError:
    NX_NOT_FOUND = True
    warnings.warn("NetworkX not found, some graph functions may fail")

class Node:
    """Implemented in strict accordance with the official Node class"""
    
    def __init__(self, name, type, node_class, numeric):
        # Initialize neighbor-related attributes
        self.random_neigh = None
        self.neighbors = dict()
        self.neighbor_names = list()
        self.number_neighbors = 0  # Explicit initialization for clarity
        
        # Similar tokens and metadata
        self.n_similar = 1
        self.name = str(name)
        self.type = type
        self.similar_tokens = [name]
        self.similar_distance = [1.0]
        
        # Start nodes and class properties
        self.startfrom = list()
        self.node_class = dict()
        self._extract_class(node_class)
        self.numeric = numeric

    def _extract_class(self, node_type):
        # Convert node type to 3-bit binary string
        three_bit_bin = '{:03b}'.format(node_type)
        # Map binary bits to class flags
        class_flags = ['isfirst', 'isroot', 'isappear']
        for flag_idx, flag_name in enumerate(class_flags):
            # Parse each bit to boolean
            self.node_class[flag_name] = bool(int(three_bit_bin[flag_idx]))

    def get_random_start(self):
        """Get random starting node (fallback to self.name if none)"""
        if len(self.startfrom) > 0:
            # Generate random index within startfrom range
            rand_pos = int(random.random() * len(self.startfrom))
            return self.startfrom[rand_pos]
        return self.name  # Simplify else clause to direct return

    def get_weighted_random_neighbor(self):
        """Retrieve weighted random neighbor via prepped randomizer"""
        return self.random_neigh()

    def get_random_neighbor(self):
        """Get unweighted random neighbor"""
        if self.number_neighbors == 0:
            return None  # Add safe guard (consistent with potential usage)
        rand_idx = int(random.random() * self.number_neighbors)
        return self.neighbor_names[rand_idx]

    def add_neighbor(self, neighbor, weight):
        """Add neighbor with specified weight (prevent duplicate conflicts)"""
        neighbor_id = neighbor.name
        # Check if neighbor already exists
        if neighbor_id not in self.neighbors:
            self.neighbors[neighbor_id] = weight
            # Add to startfrom if neighbor is first-type node
            if neighbor.node_class['isfirst']:
                self.startfrom.append(neighbor_id)
        else:
            # Raise error if weight mismatch for duplicate edge
            existing_weight = self.neighbors[neighbor_id]
            if existing_weight != weight:
                error_msg = f'Duplicate edge between {self.name} and {neighbor} - conflicting weights ({existing_weight} vs {weight})'
                raise ValueError(error_msg)

    def get_random_replacement(self):
        """Get weighted random similar token replacement"""
        # Use choices with weights, unpack single result
        return random.choices(self.similar_tokens, weights=self.similar_distance, k=1)[0]

    def normalize_neighbors(self, uniform):
        """Normalize neighbor data (convert to arrays, prepare randomizers)"""
        # Extract neighbor names and calculate count
        self.neighbor_names = np.array(list(self.neighbors.keys()))
        self.number_neighbors = len(self.neighbor_names)
        
        # Prepare weighted randomizer if not using uniform weights
        if not uniform and self.number_neighbors > 0:
            neighbor_weight_vals = np.array(list(self.neighbors.values()))
            self.random_neigh = self._prepare_aliased_randomizer(self.neighbor_names, neighbor_weight_vals)
        
        # Convert startfrom to numpy array for efficient access
        self.startfrom = np.array(self.startfrom)
        # Clear raw neighbors dict to free memory
        self.neighbors = None

    def _prepare_aliased_randomizer(self, items, weights):
        """Simplified alias method for weighted random sampling"""
        def weighted_randomizer():
            total = sum(weights)
            if total <= 0:
                return items[0] if items else None  # Safe guard for edge case
            rand_threshold = random.random() * total
            # Iterate to find matching weight bin
            for item_idx, item_weight in enumerate(weights):
                rand_threshold -= item_weight
                if rand_threshold <= 0:
                    return items[item_idx]
            # Fallback to last item if calculation drifts
            return items[-1]
        return weighted_randomizer

    def rebuild(self):
        """Placeholder method (full implementation missing in official code)"""
        pass

    def add_similar(self, other_token, distance):
        """Add similar token with corresponding distance"""
        self.similar_tokens.append(other_token)
        self.similar_distance.append(distance)
        self.n_similar += 1  # Update similar count (logical complement)

class Graph:
    """Implemented graph structure with refactored logic flow"""
    
    def __init__(self, edge_data, prefix_info, similarity_data=None, expand_fields=[]):
        self.node_registry = {}
        self.edge_tracker = set()
        self.node_type_defs = {}
        self.is_numeric_type = {}
        
        self.visible_node_list = []
        self.field_expansion = expand_fields
        self.root_node_ids = []
        self.initial_candidate_types = []
        
        self._parse_prefix_config(prefix_info)
        self._validate_expansion_config()
        self.uniform_weight_mode = True
        if self.field_expansion == 'all':
            print('# Flatten = all, all string fields will be expanded.')
            self.field_expansion = list(self.node_type_defs.keys())
        elif len(self.field_expansion) > 0:
            print(f'# Expanding columns: [{", ".join(self.field_expansion)}].')
        else:
            print('# All values will be tokenized without field expansion.')
        for edge_entry in tqdm(edge_data, desc='# Loading and processing edgelist'):
            self._handle_single_edge(edge_entry)
        self._complete_graph_initialization()
            
        if similarity_data and len(similarity_data) > 0:
            self.merge_similarity_data(similarity_data)

    def _handle_single_edge(self, edge_entry):
        src_node_id, tgt_node_id = edge_entry[0], edge_entry[1]
        if pd.isna(src_node_id) or pd.isna(tgt_node_id):
            raise ValueError(f'Invalid edge: Source "{src_node_id}" or target "{tgt_node_id}" contains NaN')
        forward_wt, backward_wt = self._extract_edge_weights(edge_entry)
        if forward_wt != backward_wt or backward_wt is None:
            self.uniform_weight_mode = False
        src_node_set = self._process_node_identifier(src_node_id)
        tgt_node_set = self._process_node_identifier(tgt_node_id)
        for src_node in src_node_set:
            for tgt_node in tgt_node_set:
                if src_node != tgt_node:
                    self.establish_connection(src_node, tgt_node, forward_wt, backward_wt)

    def _extract_edge_weights(self, edge_entry):
        entry_len = len(edge_entry)
        if entry_len == 2:
            return 1, 1
        elif entry_len == 4:
            return edge_entry[2], edge_entry[3]
        elif entry_len == 3:
            return edge_entry[2], None
        else:
            raise ValueError(f'Record {edge_entry} has invalid number of values (expected 2, 3, or 4)')

    def _process_node_identifier(self, node_id):
        conn_list = []
        
        try:
            numeric_val = float(node_id)
            if math.isnan(numeric_val):
                return set()
            node_name = str(node_id)
        except (ValueError, OverflowError):
            node_name = str(node_id)
        node_type = self._determine_node_type(node_name)
        
        if node_type in self.field_expansion:
            components = node_name.split('_')
            for comp_idx, comp in enumerate(components):
                if (comp_idx == 0 and comp in self.node_type_defs) or comp == '':
                    continue
                expanded_id = f'{node_type}__{comp}'
                if expanded_id not in self.node_registry:
                    new_node = Node(expanded_id, node_type,
                                   node_class=self.node_type_defs[node_type],
                                   numeric=False)
                    self.node_registry[expanded_id] = new_node
            conn_list.extend([self.node_registry[f'{node_type}__{comp}']
                           for comp_idx, comp in enumerate(components)
                           if comp_idx > 0 and comp != ''])
        
        if node_name not in self.node_registry:
            new_node = Node(node_name, node_type,
                           node_class=self.node_type_defs[node_type],
                           numeric=self.is_numeric_type[node_type])
            self.node_registry[node_name] = new_node
        conn_list.append(self.node_registry[node_name])
        
        return set(conn_list)

    def _parse_prefix_config(self, prefix_list):
        valid_prefix_exists = False
        for prefix_entry in prefix_list:
            props, prefix_name = prefix_entry.split('__')
            class_code = int(props[0])
            data_type = props[1]
            
            if class_code not in range(8):
                raise ValueError(f'Unsupported node class {class_code} (must be 0-7)')
            self.node_type_defs[prefix_name] = class_code
            if class_code >= 4:
                self.initial_candidate_types.append(prefix_name)
                
            if class_code % 2 == 1:
                self.visible_node_list.append(prefix_entry)
                valid_prefix_exists = True
                
            if data_type not in ['#', '$']:
                raise ValueError(f'Unknown data type prefix {data_type} (expected # or $)')
            self.is_numeric_type[prefix_name] = (data_type == '#')
                
        if not valid_prefix_exists:
            raise ValueError('No nodes with isappear=True found - random walks will be empty')

    def _determine_node_type(self, node_name):
        for node_type in self.node_type_defs:
            if node_name.startswith(f'{node_type}__'):
                return node_type
        raise ValueError(f'Node {node_name} has no recognized prefix')

    def _validate_expansion_config(self):
        if self.field_expansion != 'all':
            for field in self.field_expansion:
                if field not in self.node_type_defs:
                    raise ValueError(f'Unrecognized expansion field {field}')

    def _complete_graph_initialization(self):
        nodes_to_remove = []
        if len(self.node_registry) == 0:
            raise ValueError('No nodes detected in edgelist!')
            
        for node_id in tqdm(self.node_registry, desc='# Preparing neighbor randomizers'):
            current_node = self.node_registry[node_id]
            if current_node.node_class['isroot']:
                self.root_node_ids.append(node_id)
            if len(current_node.neighbors) == 0:
                raise ValueError(f'Node {node_id} has no neighbors - invalid graph structure')
            current_node.normalize_neighbors(uniform=self.uniform_weight_mode)
                
        for node_id in nodes_to_remove:
            self.node_registry.pop(node_id)

    def establish_connection(self, source_node, target_node, forward_wt, backward_wt=None):
        if backward_wt is None:
            backward_wt = forward_wt
            
        source_node.add_neighbor(target_node, forward_wt)
        if backward_wt is not None:
            target_node.add_neighbor(source_node, backward_wt)

    def merge_similarity_data(self, similarity_data):
        for sim_entry in similarity_data:
            node_a, node_b = sim_entry[0], sim_entry[1]
            if node_a not in self.node_registry or node_b not in self.node_registry:
                continue
            sim_score = sim_entry[2]
            try:
                self.node_registry[node_a].add_similar(node_b, sim_score)
                self.node_registry[node_b].add_similar(node_a, sim_score)
            except KeyError:
                pass
        for node_id in self.node_registry:
            self.node_registry[node_id].rebuild()

    def calculate_sentence_count(self, sentence_len, multiplier=1000):
        gen_count = len(self.node_registry) * multiplier // sentence_len
        print(f'# {gen_count} sentences will be generated.')
        return gen_count

    def get_all_nodes(self):
        return list(self.node_registry.keys())

    def find_common_nodes(self, common_node_names):
        common_nodes = set()
        for node_id in self.node_registry:
            if '__' in node_id:
                node_type, name_segment = node_id.split('__', 1)
                if name_segment in common_node_names:
                    common_nodes.add(node_id)
        return common_nodes

class RandomWalk:
    """Implemented strictly following the official RandomWalk class specification"""
    
    def __init__(self, graph_node_registry, start_node_id, walk_length, no_backtrack, use_uniform_weights,
                 replace_strings=True, replace_numbers=True, follow_repl=False):
        start_node = graph_node_registry[start_node_id]
        initial_node_name = start_node.get_random_start()
        self.walk_sequence = [initial_node_name, start_node_id] if initial_node_name != start_node_id else [start_node_id]
        current_node_id = start_node_id
        current_node = graph_node_registry[current_node_id]
        step_count = len(self.walk_sequence)
        while step_count < walk_length:
            next_node_id = current_node.get_random_neighbor() if use_uniform_weights else current_node.get_weighted_random_neighbor()
            if next_node_id is None:
                break
            if replace_numbers:
                next_node_id = self._replace_numeric(next_node_id, graph_node_registry)
            if replace_strings:
                original_node_id, replaced_node_id = self._replace_string(graph_node_registry[next_node_id])
            else:
                original_node_id = next_node_id
                replaced_node_id = next_node_id
            if no_backtrack and next_node_id == self.walk_sequence[-1]:
                continue
            if follow_repl:
                next_node_id = replaced_node_id
            current_node = graph_node_registry[next_node_id]
            if not current_node.node_class['isappear']:
                continue
            self.walk_sequence.append(replaced_node_id if replaced_node_id != original_node_id else next_node_id)
            step_count += 1

    def get_walk(self):
        return self.walk_sequence

    def _replace_numeric(self, node_id, node_registry):
        if node_id in node_registry and node_registry[node_id].numeric:
            try:
                base_num = int(node_id)
            except ValueError:
                return node_id
            retry_count = 0
            new_num = np.around(np.random.normal(loc=base_num, scale=1))
            try:
                new_num = int(new_num)
            except OverflowError:
                return str(node_id)
            while str(new_num) not in node_registry and retry_count < 3:
                new_num = np.around(np.random.normal(loc=base_num, scale=1))
                retry_count += 1
            return str(new_num)
        return node_id

    def _replace_string(self, target_node):
        if len(target_node.similar_tokens) > 1:
            return target_node.name, target_node.get_random_replacement()
        return target_node.name, target_node.name

def learn_embeddings(output_emb_file, walk_data, save_walks, embed_dim, window, training_algo='word2vec',
                     learn_mode='skipgram', worker_count=mp.cpu_count(), sample_factor=0.001):
    """Implemented strictly following the official learn_embeddings function"""
    print(f"Embedding training params: algorithm={training_algo}, method={learn_mode}, dim={embed_dim}")
    
    try:
        if training_algo == 'word2vec':
            sg_flag = 1 if learn_mode == 'skipgram' else 0 if learn_mode == 'CBOW' else None
            if sg_flag is None:
                raise ValueError(f'Unsupported learning method: {learn_mode}')
            
            if save_walks:
                print("Loading walks from file...")
                # Handle gensim version compatibility for parameters
                try:
                    model = Word2Vec(corpus_file=walk_data, vector_size=embed_dim, window=window,
                                    min_count=1, sg=sg_flag, workers=worker_count, sample=sample_factor, epochs=10)
                except TypeError:
                    try:
                        model = Word2Vec(corpus_file=walk_data, size=embed_dim, window=window,
                                        min_count=1, sg=sg_flag, workers=worker_count, sample=sample_factor, iter=10)
                    except TypeError:
                        model = Word2Vec(corpus_file=walk_data, size=embed_dim, window=window,
                                        min_count=1, sg=sg_flag, workers=worker_count, sample=sample_factor)
                model.wv.save_word2vec_format(output_emb_file, binary=False)
            else:
                print("Loading walks from memory...")
                walk_count = len(walk_data)
                first_walk_len = len(walk_data[0]) if walk_count > 0 else 0
                print(f"Number of walks: {walk_count}, Length of first walk: {first_walk_len}")
                
                try:
                    model = Word2Vec(sentences=walk_data, vector_size=embed_dim, window=window,
                                    min_count=1, sg=sg_flag, workers=worker_count, sample=sample_factor, epochs=10)
                except TypeError:
                    try:
                        model = Word2Vec(sentences=walk_data, size=embed_dim, window=window,
                                        min_count=1, sg=sg_flag, workers=worker_count, sample=sample_factor, iter=10)
                    except TypeError:
                        model = Word2Vec(sentences=walk_data, size=embed_dim, window=window,
                                        min_count=1, sg=sg_flag, workers=worker_count, sample=sample_factor)
                model.wv.save_word2vec_format(output_emb_file, binary=False)
                
        elif training_algo == 'fasttext':
            print("Using FastText algorithm")
            if save_walks:
                try:
                    model = FastText(corpus_file=walk_data, window=window, min_count=1,
                                    workers=worker_count, vector_size=embed_dim, epochs=10)
                except TypeError:
                    try:
                        model = FastText(corpus_file=walk_data, window=window, min_count=1,
                                        workers=worker_count, size=embed_dim, iter=10)
                    except TypeError:
                        model = FastText(corpus_file=walk_data, window=window, min_count=1,
                                        workers=worker_count, size=embed_dim)
                model.wv.save_word2vec_format(output_emb_file, binary=False)
            else:
                try:
                    model = FastText(sentences=walk_data, vector_size=embed_dim, workers=worker_count,
                                    min_count=1, window=window, epochs=10)
                except TypeError:
                    try:
                        model = FastText(sentences=walk_data, size=embed_dim, workers=worker_count,
                                        min_count=1, window=window, iter=10)
                    except TypeError:
                        model = FastText(sentences=walk_data, size=embed_dim, workers=worker_count,
                                        min_count=1, window=window)
                model.wv.save_word2vec_format(output_emb_file, binary=False)
        
        print(f"Embedding training completed. Saved to: {output_emb_file}")
        
        # Verify output file
        if os.path.exists(output_emb_file):
            emb_file_size = os.path.getsize(output_emb_file)
            print(f"Embedding file size: {emb_file_size} bytes")
            
            with open(output_emb_file, 'r') as f:
                header_line = f.readline().strip()
                print(f"Embedding file header: {header_line}")
        else:
            print(f"Error: Embedding file {output_emb_file} not generated")
            
    except Exception as err:
        print(f"Embedding training failed: {err}")
        raise

# Configuration handling functions
def load_config(config_path):
    """Load configuration from file (key: value format)"""
    config_dict = {}
    with open(config_path, 'r') as config_file:
        for line_num, line_content in enumerate(config_file):
            cleaned_line = line_content.strip()
            if not cleaned_line or cleaned_line.startswith('#'):
                continue
            if ':' in cleaned_line:
                key, val = cleaned_line.split(':', 1)
                config_dict[key.strip()] = val.strip()
    return config_dict

def set_default_config(config):
    """Set default values for missing configuration parameters"""
    default_config = {
        'ntop': 10,
        'ncand': 1,
        'max_rank': 3,
        'follow_sub': False,
        'smoothing_method': 'no',
        'backtrack': True,
        'training_algorithm': 'word2vec',
        'write_walks': True,
        'flatten': 'all',
        'indexing': 'basic',
        'epsilon': 0.1,
        'num_trees': 250,
        'compression': False,
        'n_sentences': 'default',
        'walks_strategy': 'basic',
        'learning_method': 'skipgram',
        'sentence_length': 60,
        'window_size': 5,
        'n_dimensions': 300,
        'numeric': 'no',
        'experiment_type': 'ER',
        'intersection': False,
        'walks_file': None,
        'mlflow': False,
        'repl_numbers': False,
        'repl_strings': False,
        'sampling_factor': 0.001
    }
    for param in default_config:
        if param not in config:
            config[param] = default_config[param]
    return config

def validate_config(config):
    """Validate and cast configuration parameters to correct types"""
    config = set_default_config(config)
    
    # Convert boolean parameters
    bool_params = ['backtrack', 'write_walks', 'compression', 'intersection', 'repl_strings', 'repl_numbers']
    for param in bool_params:
        if isinstance(config[param], bool):
            continue
        lower_val = config[param].lower()
        if lower_val == 'true':
            config[param] = True
        elif lower_val == 'false':
            config[param] = False
    
    # Convert numeric parameters (skip 'default' n_sentences)
    if config['n_sentences'] != 'default':
        try:
            config['n_sentences'] = int(config['n_sentences'])
            if config['n_sentences'] <= 0:
                raise ValueError('n_sentences must be a positive integer.')
        except ValueError:
            raise ValueError('n_sentences must be an integer or "default".')
    
    # Convert remaining numeric params（补充ncand、max_rank）
    numeric_params = {
        'n_dimensions': int,
        'window_size': int,
        'sentence_length': int,
        'ntop': int,
        'ncand': int,  
        'max_rank': int,  
        'sampling_factor': float
    }
    for param, cast_func in numeric_params.items():
        try:
            config[param] = cast_func(config[param])
        except (ValueError, TypeError):
            raise ValueError(f'Parameter "{param}" must be a valid {cast_func.__name__}')
    
    return config

def load_edgelist(edgelist_file):
    """Load edgelist from file and return prefixes + edge data"""
    edge_data = []
    node_type_list = []
    with open(edgelist_file, 'r') as file:
        for line_idx, line in enumerate(file):
            stripped_line = line.strip()
            if line_idx == 0:
                node_type_list = stripped_line.split(',')
            else:
                split_line = stripped_line.split(',')
                base_edge = split_line[:2]
                if len(split_line) > 2:
                    for weight_str in split_line[2:]:
                        base_edge.append(float(weight_str))
                edge_data.append(base_edge)
    return node_type_list, edge_data

def build_graph(config, edgelist, prefix_list, dict_mapping=None):
    """Build the heterogeneous graph from edgelist and configuration"""
    if config['walks_strategy'] == 'replacement':
        raise NotImplementedError("Replacement walks strategy is not implemented")
    similarity_data = None
    # Process flatten configuration
    flatten_setting = config.get('flatten', '')
    if flatten_setting:
        lower_flatten = flatten_setting.lower()
        if lower_flatten not in ['all', 'false']:
            flatten_config = [item.strip() for item in flatten_setting.split(',')]
        elif lower_flatten == 'false':
            flatten_config = []
        else:
            flatten_config = 'all'
    else:
        flatten_config = []
    # Track graph construction time
    start_time = dt.datetime.now()
    print(OUTPUT_FORMAT.format('Initiating graph construction', start_time.strftime(TIME_FORMAT)))
    
    # Apply dictionary mapping if provided
    if dict_mapping:
        mapped_edgelist = []
        for edge in edgelist:
            mapped_edge = []
            for elem in edge:
                mapped_edge.append(dict_mapping.get(elem, elem))
            mapped_edgelist.append(mapped_edge)
        edgelist = mapped_edgelist
    # Create graph instance
    graph = Graph(edge_data=edgelist, prefix_info=prefix_list, similarity_data=similarity_data, expand_fields=flatten_config)
    
    end_time = dt.datetime.now()
    time_elapsed = end_time - start_time
    print()
    print(OUTPUT_FORMAT.format('Graph construction finished', end_time.strftime(TIME_FORMAT)))
    print(OUTPUT_FORMAT.format('Graph build time:', f'{time_elapsed.total_seconds():.2f} seconds.'))
    return graph

def find_common_values(df, info_path):
    """Find common values between two datasets (for intersection mode)"""
    with open(info_path, 'r') as info_file:
        first_line = info_file.readline()
        item_count = int(first_line.split(',')[1])
    df_part1 = df[:item_count]
    df_part2 = df[item_count:]
    
    # Extract unique values from both parts
    values_part1 = set(str(val) for val in df_part1.values.ravel().tolist())
    values_part2 = set(str(val) for val in df_part2.values.ravel().tolist())
    
    return values_part1.intersection(values_part2)

def create_random_walks(params, graph, common_nodes=None):
    """Generate random walks from the graph (fixed default n_sentences handling)"""
    
    # Determine number of sentences
    if params['n_sentences'] == 'default':
        total_sentences = graph.calculate_sentence_count(int(params['sentence_length']))
    else:
        total_sentences = int(float(params['n_sentences']))
        
    walk_length = int(params['sentence_length'])
    allow_backtrack = params['backtrack']
    # Set intersection nodes
    if common_nodes is None:
        common_nodes = set(graph.get_all_nodes())
    node_count = len(common_nodes)
    
    # Calculate walks per node
    walks_per_node = total_sentences // node_count
    # Define walks output file
    walks_output_path = f'pipeline/walks/{params["output_file"]}.walks'
    # Initialize file handler if writing walks
    walk_file = open(walks_output_path, 'w') if params['write_walks'] else None
    generated_count = 0
    # Start walks generation
    start_gen_time = dt.datetime.now()
    print(OUTPUT_FORMAT.format('Generating base random walks.', start_gen_time.strftime(TIME_FORMAT)))
    # Generate walks for each node
    if walks_per_node > 0:
        progress_bar = tqdm(desc='# Walk generation progress: ', total=node_count * walks_per_node)
        for node_id in common_nodes:
            node_walks = []
            for _ in range(walks_per_node):
                rw = RandomWalk(graph.node_registry, node_id, walk_length, not allow_backtrack, graph.uniform_weight_mode,
                               replace_strings=params['repl_strings'], replace_numbers=params['repl_numbers'])
                node_walks.append(rw.get_walk())
            if params['write_walks']:
                walk_lines = '\n'.join(' '.join(walk) for walk in node_walks) + '\n'
                walk_file.write(walk_lines)
            else:
                if 'generated_walks' not in locals():
                    generated_walks = []
                generated_walks.extend(node_walks)
            
            generated_count += walks_per_node
            progress_bar.update(walks_per_node)
        progress_bar.close()
    # Generate remaining walks if needed
    remaining_walks = total_sentences - generated_count
    if remaining_walks > 0:
        comp_time = dt.datetime.now()
        print(OUTPUT_FORMAT.format('Generating remaining random walks.', comp_time.strftime(TIME_FORMAT)))
        node_list = list(common_nodes)
        with tqdm(total=remaining_walks, desc='# Final walks progress: ') as pbar:
            for _ in range(remaining_walks):
                selected_node = random.choice(node_list)
                rw = RandomWalk(graph.node_registry, selected_node, walk_length, not allow_backtrack, graph.uniform_weight_mode,
                               replace_strings=params['repl_strings'], replace_numbers=params['repl_numbers'])
                walk = rw.get_walk()
                if params['write_walks']:
                    walk_file.write(' '.join(walk) + '\n')
                else:
                    generated_walks.append(walk)
                
                generated_count += 1
                pbar.update(1)
    # Final status
    finish_time = dt.datetime.now()
    print(OUTPUT_FORMAT.format('Random walks generation completed', finish_time.strftime(TIME_FORMAT)))
    print()
    # Cleanup and return
    if params['write_walks']:
        walk_file.close()
        return walks_output_path
    else:
        return generated_walks

def execute_walks_generation(config, graph):
    """Wrapper for random walks generation (handles intersection mode)"""
    start_total_time = dt.datetime.now()
    
    common_nodes = None
    if config['intersection']:
        print('# Searching for common values between datasets. ')
        if config['flatten']:
            warnings.warn('Intersection mode enabled with flatten = True.')
        dataset_df = pd.read_csv(config['dataset_file'])
        common_values = find_common_values(dataset_df, config['dataset_info'])
        common_nodes = graph.find_common_nodes(common_values)
        
        if not common_nodes:
            warnings.warn('No common tokens found between datasets. Disabling intersection.')
            common_nodes = None
        else:
            print(f'# Found {len(common_nodes)} common values.')
    else:
        print('# Skipping common values detection. ')
    # Generate walks
    walks_result = create_random_walks(config, graph, common_nodes=common_nodes)
    
    end_total_time = dt.datetime.now()
    total_elapsed = end_total_time - start_total_time
    print(f"Random walks generation finished. Total time: {total_elapsed.total_seconds():.2f} seconds")
    return walks_result