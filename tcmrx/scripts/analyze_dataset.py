"""数据集分析脚本。

该脚本与 `run_train.py` 共用同一套数据读取逻辑，但不会训练模型，
而是输出疾病/方剂/靶点覆盖度、配对重叠度等统计，帮助定位召回率偏低的根因。
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import load_config  # noqa: E402
from dataio.dataset_builder import TCMRXDataset  # noqa: E402
from dataio.filters import (  # noqa: E402
    filter_sd1_by_pki,
    per_chemical_topk,
)
from dataio.joins import (  # noqa: E402
    chemicals_to_targets,
    chp_to_chemicals,
    cpms_to_chp,
    cpms_to_icd11,
    diseases_to_targets,
    formulas_to_targets,
    icd11_to_targets,
)
from dataio.readers import TSVReader  # noqa: E402
from training.splits import stratified_disease_split  # noqa: E402


logger = logging.getLogger("tcmrx.data_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCM-RX 数据集洞察工具")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="训练/数据处理配置路径",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="config/paths.yaml",
        help="数据文件路径配置",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="保存分析结果的 JSON 文件（为空则仅打印）",
    )
    parser.add_argument(
        "--top-targets",
        type=int,
        nargs="*",
        default=(10, 50, 100),
        help="计算靶点覆盖度的前K列表",
    )
    return parser.parse_args()


def load_processed_dataset(config: Dict, paths_config: Dict) -> Tuple[TCMRXDataset, Dict[str, List[str]], Dict[str, List[str]]]:
    """按照训练脚本的流程读取并处理原始数据。"""

    reader = TSVReader(paths_config)

    formula_tables = reader.read_formula_tables()
    disease_tables = reader.read_disease_tables()
    prediction_tables = reader.read_prediction_tables()

    filtering_config = config.get('filtering', {})

    cpms_to_chp_map = cpms_to_chp(formula_tables['D4_CPM_CHP'])
    chp_to_chemicals_map = chp_to_chemicals(
        formula_tables['D9_CHP_InChIKey'],
        formula_tables.get('D12_InChIKey'),
    )

    sd1_df = prediction_tables['SD1_predicted']
    sd1_df = filter_sd1_by_pki(sd1_df, filtering_config.get('pki_threshold'))
    chemical_to_targets_map = per_chemical_topk(
        chemicals_to_targets(sd1_df),
        filtering_config.get('topk_c'),
    )

    icd11_to_targets_map = icd11_to_targets(
        disease_tables['D19_ICD11_CUI'],
        disease_tables['D20_ICD11_MeSH'],
        disease_tables['D22_CUI_targets'],
        disease_tables['D23_MeSH_targets'],
    )

    formula_targets_raw = formulas_to_targets(
        cpms_to_chp_map,
        chp_to_chemicals_map,
        chemical_to_targets_map,
    )
    disease_targets_raw = diseases_to_targets(icd11_to_targets_map)

    cpms_to_icd11_map = cpms_to_icd11(formula_tables['D5_CPM_ICD11'])
    positive_pairs_raw = [
        (icd11, cpm)
        for cpm, icd11_list in cpms_to_icd11_map.items()
        for icd11 in icd11_list
    ]

    dataset = TCMRXDataset(config)
    dataset.build_from_raw_data(disease_targets_raw, formula_targets_raw, positive_pairs_raw)

    disease_to_formulas: Dict[str, List[str]] = defaultdict(list)
    formula_to_diseases: Dict[str, List[str]] = defaultdict(list)
    for disease_id, formula_id in positive_pairs_raw:
        disease_to_formulas[disease_id].append(formula_id)
        formula_to_diseases[formula_id].append(disease_id)

    return dataset, disease_to_formulas, formula_to_diseases


def summarize_counts(values: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {'count': 0}
    return {
        'count': int(arr.size),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p90': float(np.percentile(arr, 90)),
        'p99': float(np.percentile(arr, 99)),
    }


def compute_entropy(weights: Iterable[float]) -> float:
    arr = np.asarray([w for w in weights if w > 0], dtype=np.float32)
    if arr.size == 0:
        return 0.0
    arr = arr / (arr.sum() + 1e-12)
    return float(-(arr * np.log(arr + 1e-12)).sum())


def summarize_target_sets(processed_sets: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Dict[str, float]]:
    lengths = []
    entropies = []
    for items in processed_sets.values():
        if not items:
            continue
        lengths.append(len(items))
        entropies.append(compute_entropy(weight for _, weight in items))
    return {
        'length': summarize_counts(lengths),
        'weight_entropy': summarize_counts(entropies),
    }


def compute_target_frequency(processed_sets: Dict[str, List[Tuple[str, float]]], top_ks: Sequence[int]) -> Dict[str, float]:
    counter: Counter[str] = Counter()
    for items in processed_sets.values():
        counter.update(target_id for target_id, _ in items)

    total_assignments = sum(counter.values())
    if total_assignments == 0:
        return {}

    sorted_counts = [count for _, count in counter.most_common()]
    coverage = {}
    cumulative = np.cumsum(sorted_counts)
    for k in top_ks:
        if k <= 0:
            continue
        capped_k = min(k, len(sorted_counts))
        if capped_k == 0:
            coverage[f'top_{k}'] = 0.0
        else:
            coverage[f'top_{k}'] = float(cumulative[capped_k - 1] / total_assignments)
    return coverage


def to_target_set(items: List[Tuple[str, float]]) -> set:
    return {target_id for target_id, _ in items}


def summarize_pair_overlap(dataset: TCMRXDataset,
                           pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    jaccards = []
    coverage = []
    for disease_id, formula_id in pairs:
        disease_targets = dataset.processed_disease_targets.get(disease_id, [])
        formula_targets = dataset.processed_formula_targets.get(formula_id, [])
        d_set = to_target_set(disease_targets)
        f_set = to_target_set(formula_targets)
        if not d_set or not f_set:
            continue
        intersection = len(d_set & f_set)
        union = len(d_set | f_set)
        if union > 0:
            jaccards.append(intersection / union)
        if len(d_set) > 0:
            coverage.append(intersection / len(d_set))

    return {
        'pair_count': len(jaccards),
        'jaccard_mean': float(np.mean(jaccards)) if jaccards else 0.0,
        'jaccard_median': float(np.median(jaccards)) if jaccards else 0.0,
        'coverage_mean': float(np.mean(coverage)) if coverage else 0.0,
        'coverage_p10': float(np.percentile(coverage, 10)) if coverage else 0.0,
        'coverage_p90': float(np.percentile(coverage, 90)) if coverage else 0.0,
    }


def summarize_link_multiplicity(link_map: Dict[str, List[str]]) -> Dict[str, float]:
    multiplicities = [len(values) for values in link_map.values() if values]
    return summarize_counts(multiplicities)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(args.config)
    paths_config = load_config(args.paths)

    dataset, disease_to_formulas, formula_to_diseases = load_processed_dataset(config, paths_config)

    split = stratified_disease_split(
        [(d, f) for d, formulas in disease_to_formulas.items() for f in formulas],
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        seed=config['training']['seed'],
    )

    analysis = {
        'entity_stats': {
            'num_diseases': dataset.id_mapper.get_count('disease'),
            'num_formulas': dataset.id_mapper.get_count('formula'),
            'num_targets': dataset.id_mapper.get_count('target'),
        },
        'disease_target_sets': summarize_target_sets(dataset.processed_disease_targets),
        'formula_target_sets': summarize_target_sets(dataset.processed_formula_targets),
        'target_frequency': {
            'disease_side': compute_target_frequency(dataset.processed_disease_targets, args.top_targets),
            'formula_side': compute_target_frequency(dataset.processed_formula_targets, args.top_targets),
        },
        'positive_link_multiplicity': {
            'diseases': summarize_link_multiplicity(disease_to_formulas),
            'formulas': summarize_link_multiplicity(formula_to_diseases),
        },
        'pair_overlap': {
            'train': summarize_pair_overlap(dataset, split['train']),
            'val': summarize_pair_overlap(dataset, split['val']),
            'test': summarize_pair_overlap(dataset, split['test']),
        },
    }

    ambiguous_diseases = [
        disease
        for disease, formulas in disease_to_formulas.items()
        if len(formulas) >= 5
    ]
    analysis['notes'] = {
        'high_multiplicity_diseases': ambiguous_diseases[:20],
        'diseases_with_no_targets': [
            disease for disease, items in dataset.processed_disease_targets.items() if not items
        ],
        'formulas_with_no_targets': [
            formula for formula, items in dataset.processed_formula_targets.items() if not items
        ],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info("分析结果已保存到 %s", output_path)
    else:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

