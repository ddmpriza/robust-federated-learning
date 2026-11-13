
import pytest
import torch
from src.gradient_aggregation import huber_aggregation_paper


def test_huber_reduces_outlier_influence():

    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [100.0, 0.0]
    ])

    thresholds = torch.tensor([10.0, 10.0, 10.0])

    result = huber_aggregation_paper(
        gradients,
        thresholds
    )

    mean = gradients.mean(dim=0)

    assert torch.linalg.vector_norm(result - torch.tensor([3.0, 0.0])) < torch.linalg.vector_norm(
        mean - torch.tensor([3.0, 0.0])
    )


def test_identical_gradients():

    gradients = torch.tensor([
        [3.0, 5.0],
        [3.0, 5.0],
        [3.0, 5.0]
    ])

    thresholds = torch.tensor([1.0, 1.0, 1.0])

    result = huber_aggregation_paper(
        gradients,
        thresholds
    )

    expected = torch.tensor([3.0, 5.0])

    assert torch.allclose(result, expected, atol=1e-6)


def test_single_client():

    gradients = torch.tensor([[3.0, 5.0]])
    thresholds = torch.tensor([1.0])

    result = huber_aggregation_paper(
        gradients,
        thresholds
    )

    assert torch.allclose(result, gradients[0], atol=1e-6)

def test_different_thresholds():

    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [100.0, 0.0]
    ])

    thresholds = torch.tensor([10.0, 10.0, 1.0])

    result = huber_aggregation_paper(
        gradients,
        thresholds
    )

    assert torch.isfinite(result).all()
    assert result[0] < 10.0

def test_scalar_threshold():

    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [100.0, 0.0]
    ])

    result = huber_aggregation_paper(
        gradients,
        thresholds=10.0
    )

    assert torch.isfinite(result).all()
    assert result.shape == (2,)

def test_invalid_threshold():

    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0]
    ])

    thresholds = torch.tensor([1.0, -1.0])

    with pytest.raises(ValueError):
        huber_aggregation_paper(gradients, thresholds)
