import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcmrx.dataio.dataset_builder import TCMRXDataset


def build_dataset(filtering_config):
    config = {'filtering': filtering_config}
    dataset = TCMRXDataset(config)

    disease_targets = {
        'd1': [('t1', 1.0), ('t2', 1.0), ('t3', 1.0)],
        'd2': [('t1', 1.0), ('t4', 1.0)],
    }
    formula_targets = {
        'f1': [('t1', 1.0), ('t2', 0.5), ('t5', 0.25)],
        'f2': [('t1', 1.0), ('t6', 1.0), ('t7', 1.0)],
    }
    positive_pairs = [('d1', 'f1'), ('d2', 'f2')]

    dataset.build_from_raw_data(disease_targets, formula_targets, positive_pairs)
    return dataset


def test_frequency_weighting_downweights_popular_targets():
    dataset = build_dataset({
        'inverse_freq_weight': False,
        'disease_target_trimming': {
            'enabled': True,
            'max_items': 2,
            'min_items': 1,
            'mass_threshold': 0.6,
        },
        'formula_target_trimming': {
            'enabled': True,
            'max_items': 3,
        },
        'frequency_reweighting': {
            'disease': {
                'enabled': True,
                'method': 'idf',
                'power': 1.0,
            }
        },
    })

    disease_targets = dataset.processed_disease_targets['d1']
    assert len(disease_targets) == 2
    target_ids = {tid for tid, _ in disease_targets}
    assert 't1' not in target_ids  # 高频靶点被裁剪掉


def test_trimming_respects_minimum_items():
    dataset = build_dataset({
        'inverse_freq_weight': False,
        'disease_target_trimming': {
            'enabled': True,
            'max_items': 1,
            'min_items': 2,
        },
        'formula_target_trimming': {
            'enabled': True,
            'max_items': 1,
            'min_items': 2,
        },
    })

    assert len(dataset.processed_disease_targets['d1']) == 2
    assert len(dataset.processed_formula_targets['f1']) == 2


def test_topk_fallback_still_available():
    dataset = build_dataset({
        'inverse_freq_weight': False,
        'topk_d': 1,
        'topk_f': 2,
    })

    assert len(dataset.processed_disease_targets['d1']) == 1
    assert len(dataset.processed_formula_targets['f1']) == 2
