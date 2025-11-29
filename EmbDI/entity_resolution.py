#!/usr/bin/env python
# coding: utf-8
"""
Entity Resolution (ER) module for EmbDI
- Core functions for similarity search, ground truth evaluation, and match generation
- Supports multiple indexing strategies: basic, annoy, ngt, faiss
"""

import numpy as np
import pandas as pd
import warnings
import datetime as dt
import time
import random
import math
from tqdm import tqdm
import sys
import os
from io import StringIO
import argparse
import pickle
import csv

# -------------------------- 可选依赖的容错导入 --------------------------
# 处理gensim（核心依赖，若缺失会运行时报错）
try:
    import gensim.models as models
    from gensim.models import Word2Vec, FastText
except ImportError:
    raise ImportError("Missing required dependency: 'gensim'. Install with 'pip install gensim==4.3.1'")

# 处理可选索引库（缺失时仅禁用对应策略，不崩溃）
NGT_UNAVAILABLE = ANNOY_UNAVAILABLE = FAISS_UNAVAILABLE = False
try:
    import ngtpy
except ModuleNotFoundError:
    warnings.warn('ngtpy library not installed. NGT indexing strategy is disabled.')
    NGT_UNAVAILABLE = True

try:
    import faiss
except ModuleNotFoundError:
    warnings.warn('faiss library not installed. FAISS indexing strategy is disabled.')
    FAISS_UNAVAILABLE = True

try:
    from gensim.similarities.index import AnnoyIndexer
except ImportError:
    warnings.warn('AnnoyIndexer not found. Annoy indexing strategy is disabled.')
    ANNOY_UNAVAILABLE = True


# -------------------------- 辅助函数 --------------------------
def _parse_node_num(node_id):
    """Helper: 解析节点ID的数字部分（严格校验idx__格式）"""
    if not node_id.startswith('idx__'):
        return None  # 非idx节点直接返回None
    try:
        prefix_part, num_part = node_id.split('__', 1)  # 仅按第一个__分割
        return int(num_part)
    except (ValueError, IndexError):
        print(f"Warning: 无效的idx节点ID格式 '{node_id}'（预期格式：'idx__数字'）")
        return None


def validate_match_symmetry(target_node, similar_nodes_dict, top_k):
    """Validate match symmetry (A matches B => B matches A)"""
    valid_matches = []
    for candidate in similar_nodes_dict.get(target_node, []):
        if target_node in similar_nodes_dict.get(candidate, [])[:top_k]:
            valid_matches.append(candidate)
    return valid_matches


# -------------------------- 核心函数 --------------------------
def parse_command_line_args():
    """Parse command line arguments for standalone ER execution"""
    parser = argparse.ArgumentParser(description='Entity Resolution Standalone Execution')
    parser.add_argument('-i', '--input_file', required=True, type=str,
                        help='Path to the input embedding file (.emb)')
    parser.add_argument('-m', '--matches_file', required=True, type=str,
                        help='Path to the ground truth matches file')
    parser.add_argument('--n_top', default=5, type=int,
                        help='Number of top similar neighbors to retrieve (default: 5)')
    parser.add_argument('--n_candidates', default=1, type=int,
                        help='Number of final candidates to select from top neighbors (default: 1)')
    parser.add_argument('--info_file', required=True, type=str,
                        help='Path to the dataset info file (contains n_items)')
    return parser.parse_args()


def construct_similarity_index(embedding_file, valid_lines, dataset_size, index_strategy,
                               top_k=10, candidate_count=1, tree_count=None, search_epsilon=None):
    """Build similarity candidate set using specified indexing strategy"""

    top_k = int(top_k)
    candidate_count = int(candidate_count)
    
    start_time = dt.datetime.now()
    similarity_map = {}  # Maps node to its candidate matches
    progress_counter = 1
    node_ids = [line.split(' ', maxsplit=1)[0] for line in valid_lines if line.startswith('idx__')]
    if not node_ids:
        raise ValueError("No valid 'idx__' nodes found in embedding file")

    if index_strategy == 'annoy' and ANNOY_UNAVAILABLE:
        warnings.warn('Annoy strategy selected but unavailable. Falling back to basic strategy.')
        index_strategy = 'basic'
    if index_strategy == 'ngt' and NGT_UNAVAILABLE:
        warnings.warn('NGT strategy selected but unavailable. Falling back to basic strategy.')
        index_strategy = 'basic'
    if index_strategy == 'faiss' and FAISS_UNAVAILABLE:
        warnings.warn('FAISS strategy selected but unavailable. Falling back to basic strategy.')
        index_strategy = 'basic'


    # -------------------------- Basic策略（暴力搜索，带严格校验） --------------------------
    if index_strategy == 'basic':
        # 加载嵌入模型
        try:
            embedding_model = models.KeyedVectors.load_word2vec_format(
                embedding_file, unicode_errors='ignore'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

        # 遍历所有idx节点
        for node in tqdm(node_ids, desc='# ER - Retrieving similar nodes: ', file=sys.stdout):
            # 解析当前节点的数字部分
            node_num = _parse_node_num(node)
            if node_num is None:
                continue  # 无效节点跳过

            # 获取top-k相似节点（处理模型中不存在的节点）
            try:
                top_similar = embedding_model.most_similar(str(node), topn=top_k)
            except KeyError:
                print(f"Warning: 节点 '{node}' 未在嵌入模型中找到，跳过")
                continue
            similar_node_ids = [item[0] for item in top_similar]

            # 筛选跨数据集候选（严格校验每个相似节点）
            cross_dataset_candidates = []
            for sim_node in similar_node_ids:
                # 解析相似节点的数字部分
                sim_node_num = _parse_node_num(sim_node)
                if sim_node_num is None:
                    continue

                # 跨数据集判断：当前节点在数据集1 → 找数据集2的节点，反之亦然
                if (node_num < dataset_size and sim_node_num >= dataset_size) or \
                   (node_num >= dataset_size and sim_node_num < dataset_size):
                    cross_dataset_candidates.append(sim_node)

            # 取前N个候选
            final_candidates = cross_dataset_candidates[:candidate_count]
            similarity_map[node] = final_candidates
            progress_counter += 1
        print('')


    # -------------------------- Annoy索引策略（带校验） --------------------------
    elif index_strategy == 'annoy':
        assert tree_count is not None and tree_count > 0, \
            'Annoy strategy requires positive integer tree_count parameter'
        print('Initiating ANNOY indexing for similarity search.')

        # 加载模型+初始化Annoy索引器
        try:
            embedding_model = models.KeyedVectors.load_word2vec_format(
                embedding_file, unicode_errors='ignore'
            )
            annoy_searcher = AnnoyIndexer(embedding_model, num_trees=tree_count)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Annoy index: {e}")

        # 遍历所有idx节点
        for node in tqdm(node_ids, file=sys.stdout):
            node_num = _parse_node_num(node)
            if node_num is None:
                continue

            # 搜索相似节点
            try:
                top_similar = embedding_model.most_similar(
                    str(node), topn=top_k, indexer=annoy_searcher
                )
            except KeyError:
                print(f"Warning: 节点 '{node}' 未在嵌入模型中找到，跳过")
                continue
            similar_node_ids = [item[0] for item in top_similar]

            # 筛选跨数据集候选
            cross_dataset_candidates = []
            for sim_node in similar_node_ids:
                sim_node_num = _parse_node_num(sim_node)
                if sim_node_num is None:
                    continue
                if (node_num < dataset_size and sim_node_num >= dataset_size) or \
                   (node_num >= dataset_size and sim_node_num < dataset_size):
                    cross_dataset_candidates.append(sim_node)

            similarity_map[node] = cross_dataset_candidates[:candidate_count]
            progress_counter += 1
        print('')


    # -------------------------- NGT索引策略（带校验） --------------------------
    elif index_strategy == 'ngt':
        assert search_epsilon is not None and 0 <= search_epsilon <= 1, \
            'NGT strategy requires epsilon parameter in [0, 1]'
        print('Initiating NGT indexing for similarity search.')
        ngt_index_dir = 'pipeline/dump/ngt_index.nn'
        node_name_list = []  # 存储节点名
        vector_dim = None

        # 读取嵌入文件构建NGT索引
        try:
            with open(embedding_file, 'r') as f:
                total_nodes, vector_dim = map(int, f.readline().split())
                ngtpy.create(ngt_index_dir, vector_dim, distance_type='Cosine')
                ngt_search_index = ngtpy.Index(ngt_index_dir)

                # 插入所有节点向量（仅保留idx节点）
                for line in f:
                    node_id, vector_str = line.rstrip().split(' ', maxsplit=1)
                    if _parse_node_num(node_id) is not None:  # 仅保留idx节点
                        vector = list(map(float, vector_str.split(' ')))
                        ngt_search_index.insert(vector)
                        node_name_list.append(node_id)
            ngt_search_index.build_index()
            ngt_search_index.save()
        except Exception as e:
            raise RuntimeError(f"Failed to build NGT index: {e}")

        # 搜索相似节点
        for node in tqdm(node_ids):
            node_num = _parse_node_num(node)
            if node_num is None:
                continue

            # 获取查询向量
            try:
                node_idx = node_name_list.index(node)
                query_vector = ngt_search_index.get_object(node_idx)
            except (ValueError, IndexError):
                print(f"Warning: 节点 '{node}' 未在NGT索引中找到，跳过")
                continue

            # 搜索top-k+1（跳过自身）
            search_results = ngt_search_index.search(query_vector, size=top_k + 1, epsilon=search_epsilon)
            similar_indices = [item[0] for item in search_results[1:]]  # 跳过自身
            similar_node_ids = [node_name_list[idx] for idx in similar_indices]

            # 筛选跨数据集候选
            cross_dataset_candidates = []
            for sim_node in similar_node_ids:
                sim_node_num = _parse_node_num(sim_node)
                if sim_node_num is None:
                    continue
                if (node_num < dataset_size and sim_node_num >= dataset_size) or \
                   (node_num >= dataset_size and sim_node_num < dataset_size):
                    cross_dataset_candidates.append(sim_node)

            similarity_map[node] = cross_dataset_candidates[:candidate_count]
            progress_counter += 1
        print('')


    # -------------------------- FAISS索引策略（带校验） --------------------------
    elif index_strategy == 'faiss':
        print('Initiating FAISS indexing for similarity search.')
        node_name_list = []  # 存储节点名
        vector_matrix = []   # 存储所有向量
        vector_dim = None

        # 读取嵌入文件构建FAISS索引
        try:
            with open(embedding_file, 'r') as f:
                total_nodes, vector_dim = map(int, f.readline().split())
                faiss_search_index = faiss.IndexFlatL2(vector_dim)

                # 加载所有idx节点的向量
                for line in f:
                    node_id, vector_str = line.rstrip().split(' ', maxsplit=1)
                    node_num = _parse_node_num(node_id)
                    if node_num is not None:
                        vector = np.array(list(map(float, vector_str.split(' '))), ndmin=1).astype('float32')
                        vector_matrix.append(vector)
                        node_name_list.append(node_id)
            vector_matrix = np.array(vector_matrix)
            faiss_search_index.add(vector_matrix)
        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS index: {e}")

        # 批量搜索
        distances, indices = faiss_search_index.search(vector_matrix, k=top_k + 1)  # 含自身

        # 处理每个节点
        for node_idx, node in enumerate(tqdm(node_ids)):
            node_num = _parse_node_num(node)
            if node_num is None:
                continue

            # 获取相似节点索引（跳过自身）
            similar_indices = indices[node_idx][1:]  # 跳过自身
            similar_node_ids = [node_name_list[idx] for idx in similar_indices]

            # 筛选跨数据集候选
            cross_dataset_candidates = []
            for sim_node in similar_node_ids:
                sim_node_num = _parse_node_num(sim_node)
                if sim_node_num is None:
                    continue
                if (node_num < dataset_size and sim_node_num >= dataset_size) or \
                   (node_num >= dataset_size and sim_node_num < dataset_size):
                    cross_dataset_candidates.append(sim_node)

            similarity_map[node] = cross_dataset_candidates[:candidate_count]
            progress_counter += 1
        print('')


    # -------------------------- 不支持的策略 --------------------------
    else:
        raise ValueError(f'Indexing strategy "{index_strategy}" is not supported')


    # 保存并返回结果
    end_time = dt.datetime.now()
    time_cost = end_time - start_time
    print(f'# Similarity structure construction time: {time_cost.total_seconds():.2f} seconds')
    pickle.dump(similarity_map, open('pipeline/dump/most_similar.pickle', 'wb'))
    return similarity_map


def evaluate_against_ground_truth(similarity_map, ground_truth_path, dataset_size, top_k):
    """Evaluate ER results against ground truth (precision, recall, F1)"""
    # 加载真实匹配
    ground_truth_matches = load_ground_truth_matches(ground_truth_path)
    if not ground_truth_matches:
        raise ValueError("Ground truth matches file is empty or invalid")

    # 初始化指标
    correct_matches = 0
    total_candidates = 0
    no_candidate_cases = 0
    total_ground_truth = len(ground_truth_matches)

    # 遍历所有真实匹配节点
    for node, true_matches in ground_truth_matches.items():
        # 解析节点数字部分
        node_num = _parse_node_num(node)
        if node_num is None:
            continue

        # 获取候选匹配
        candidates = similarity_map.get(node, [])
        total_candidates += len(candidates)
        if len(candidates) == 0:
            no_candidate_cases += 1
            continue

        # 验证候选是否在真实匹配中
        for candidate in candidates:
            if candidate in true_matches:
                correct_matches += 1


    # 计算指标
    precision = correct_matches / total_candidates if total_candidates > 0 else 0.0
    recall = correct_matches / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 打印结果
    print(f'# Total candidates evaluated: {total_candidates}')
    print(f'# Nodes with no candidates: {no_candidate_cases}')
    print('\nPrecision\tRecall\tF1')
    print(f'{precision*100:.4f}\t\t{recall*100:.4f}\t\t{f1*100:.4f}')
    print(f'\n# Correct matches: {correct_matches}')
    print(f'# Total ground truth matches: {total_ground_truth}')

    return {
        'P': precision,
        'R': recall,
        'F': f1,
    }


def generate_final_matches(similarity_map):
    """Generate final match pairs (avoid duplicates like (A,B) and (B,A))"""
    final_match_pairs = set()  # 用set去重
    for source_node in similarity_map:
        for target_node in similarity_map[source_node]:
            # 解析数字部分并排序（避免重复）
            source_num = _parse_node_num(source_node)
            target_num = _parse_node_num(target_node)
            if source_num is None or target_num is None:
                continue
            # 排序后组成元组（确保(A,B)和(B,A)是同一个）
            sorted_pair = tuple(sorted([f'idx__{source_num}', f'idx__{target_num}']))
            final_match_pairs.add(sorted_pair)
    return list(final_match_pairs)


def entity_resolution_pipeline(input_embedding: str, config: dict,
                               task_type: str = 'test', dataset_info_path: str = None):
    """Main ER pipeline (train-test / match task)"""
    pipeline_start = dt.datetime.now()
    # 提取配置参数
    top_k = config.get('ntop', 10)
    candidate_count = config.get('ncand', 1)
    index_strategy = config.get('indexing', 'basic')
    ground_truth_path = config.get('match_file')
    dataset_size = None

    # 验证必要参数
    if not dataset_info_path:
        raise ValueError("dataset_info_path is required for ER pipeline")
    if task_type == 'test' and not ground_truth_path:
        raise ValueError("match_file is required for test task")

    # 读取数据集大小（n_items）
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if ',' not in first_line:
                raise ValueError("Invalid dataset info file format (expected 'n_items,数字')")
            dataset_size = int(first_line.split(',')[1])
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset info: {e}")

    # 准备测试嵌入（筛选idx节点）
    temp_embedding_file, valid_node_lines = prepare_test_embedding(input_embedding)
    if not valid_node_lines:
        raise ValueError("No valid 'idx__' nodes found in input embedding file")

    # 构建相似候选集
    similarity_map = construct_similarity_index(
        temp_embedding_file, valid_node_lines, dataset_size, index_strategy,
        top_k=top_k, candidate_count=candidate_count,
        tree_count=config.get('num_trees'), search_epsilon=config.get('epsilon')
    )

    # 执行任务
    result = None
    if task_type == 'test':
        result = evaluate_against_ground_truth(
            similarity_map, ground_truth_path, dataset_size, top_k
        )
    elif task_type == 'match':
        result = generate_final_matches(similarity_map)
    else:
        raise ValueError(f'Task type "{task_type}" is not supported (only "test" or "match")')

    # 输出耗时
    pipeline_end = dt.datetime.now()
    total_time = (pipeline_end - pipeline_start).total_seconds()
    print(f'# Total ER pipeline execution time: {total_time:.2f} seconds')
    return result


def load_ground_truth_matches(ground_truth_path):
    """Load ground truth matches from file（自动适配单/双下划线格式）"""
    match_dict = {}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f.readlines()):
            stripped_line = line.strip()
            if not stripped_line:
                continue  # 跳过空行

            # 分割匹配对
            if ',' not in stripped_line:
                print(f"Warning: 第{line_idx+1}行格式无效（预期'节点1,节点2'），跳过")
                continue
            main_node, matched_node = stripped_line.split(',', 1)
            main_node = main_node.strip()
            matched_node = matched_node.strip()

            # 统一转换为idx__格式（处理单/双下划线）
            if main_node.startswith('idx_'):
                main_node = main_node.replace('_', '__', 1)  # 仅替换第一个_
            if matched_node.startswith('idx_'):
                matched_node = matched_node.replace('_', '__', 1)

            # 保存到字典
            if main_node not in match_dict:
                match_dict[main_node] = []
            match_dict[main_node].append(matched_node)

    if not match_dict:
        warnings.warn("Ground truth matches file contains no valid matches")
    return match_dict


def prepare_test_embedding(embedding_file):
    """Filter valid idx__ nodes and generate temporary embedding file"""
    valid_node_entries = []
    vector_dim = None

    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            # 读取头行（节点数+向量维度）
            header_line = f.readline().strip()
            if ' ' not in header_line:
                raise ValueError("Invalid embedding file header (expected '节点数 向量维度')")
            total_nodes, vector_dim = header_line.split(' ', 1)
            vector_dim = int(vector_dim)

            # 筛选idx__开头的节点
            for line in f:
                node_id, vector_str = line.split(' ', maxsplit=1)
                if _parse_node_num(node_id) is not None:  # 仅保留有效idx节点
                    valid_node_entries.append(line)

    except Exception as e:
        raise RuntimeError(f"Failed to process embedding file: {e}")

    # 生成临时嵌入文件
    temp_emb_path = 'pipeline/dump/indices.emb'
    with open(temp_emb_path, 'w', encoding='utf-8') as f:
        # 写头行
        f.write(f'{len(valid_node_entries)} {vector_dim}\n')
        # 写有效节点数据
        for entry in valid_node_entries:
            f.write(entry)

    return temp_emb_path, valid_node_entries