import torch

from models.twin_tower import DualTowerModel, TowerProjection


def _dummy_batch(num_targets: int) -> dict:
    batch_size = 2
    disease_target_indices = torch.tensor([
        [0, 1, 2],
        [2, 3, 4],
    ], dtype=torch.long)
    disease_target_weights = torch.tensor([
        [1.0, 0.5, 0.2],
        [1.0, 0.7, 0.4],
    ], dtype=torch.float)
    disease_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float)

    formula_target_indices = torch.tensor([
        [3, 4, 5],
        [1, 2, 0],
    ], dtype=torch.long)
    formula_target_weights = torch.tensor([
        [1.0, 0.8, 0.1],
        [0.9, 0.6, 0.4],
    ], dtype=torch.float)
    formula_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float)

    return {
        'disease_indices': torch.arange(batch_size, dtype=torch.long),
        'formula_indices': torch.arange(batch_size, dtype=torch.long),
        'disease_target_indices': disease_target_indices % num_targets,
        'disease_target_weights': disease_target_weights,
        'disease_mask': disease_mask,
        'formula_target_indices': formula_target_indices % num_targets,
        'formula_target_weights': formula_target_weights,
        'formula_mask': formula_mask,
    }


def test_tower_projection_identity_path():
    config = {
        'embedding_dim': 8,
        'temperature': 0.1,
        'dropout_rate': 0.0,
        'aggregator_type': 'weighted_sum',
        'tower_head': {'enabled': False},
    }
    model = DualTowerModel(config)
    model.set_entity_counts(3, 3, 6)

    batch = _dummy_batch(num_targets=6)
    outputs = model(batch)

    assert outputs['disease_embeddings'].shape == (2, 8)
    assert outputs['formula_embeddings'].shape == (2, 8)

    disease_norm = torch.norm(outputs['disease_embeddings'], dim=1)
    formula_norm = torch.norm(outputs['formula_embeddings'], dim=1)
    assert torch.allclose(disease_norm, torch.ones_like(disease_norm), atol=1e-4)
    assert torch.allclose(formula_norm, torch.ones_like(formula_norm), atol=1e-4)


def test_tower_projection_shared_head():
    config = {
        'embedding_dim': 8,
        'temperature': 0.1,
        'dropout_rate': 0.0,
        'aggregator_type': 'weighted_sum',
        'tower_head': {
            'enabled': True,
            'share': True,
            'hidden_dims': [16],
            'dropout': 0.0,
            'layer_norm': True,
            'residual': True,
        },
    }
    model = DualTowerModel(config)
    model.set_entity_counts(4, 4, 8)

    assert isinstance(model.disease_head, TowerProjection)
    assert model.disease_head is model.formula_head

    batch = _dummy_batch(num_targets=8)
    outputs = model(batch)

    similarity = model.compute_similarity_matrix(
        outputs['disease_embeddings'], outputs['formula_embeddings']
    )
    assert similarity.shape == (2, 2)


def test_tower_projection_unshared_head():
    config = {
        'embedding_dim': 8,
        'temperature': 0.1,
        'dropout_rate': 0.0,
        'aggregator_type': 'weighted_sum',
        'tower_head': {
            'enabled': True,
            'share': False,
            'hidden_dims': [],
            'activation': 'relu',
            'dropout': 0.0,
            'layer_norm': False,
            'residual': False,
        },
    }
    model = DualTowerModel(config)
    model.set_entity_counts(4, 4, 8)

    assert isinstance(model.disease_head, TowerProjection)
    assert isinstance(model.formula_head, TowerProjection)
    assert model.disease_head is not model.formula_head

    batch = _dummy_batch(num_targets=8)
    outputs = model(batch)
    assert outputs['scaled_similarities'].shape == (2, 2)
