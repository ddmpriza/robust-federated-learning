
import pytest
import torch
from src.gradient_aggregation import huber_aggregation_paper
from src.gradient_aggregation import huber_aggregation


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


def test_compare_paper_and_weighted_huber():
    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [100.0, 0.0]
    ])

    client_sizes = torch.tensor([100.0, 100.0, 800.0])
    thresholds = torch.tensor([10.0, 10.0, 10.0])

    paper_result = huber_aggregation_paper(
        gradients,
        thresholds
    )

    weighted_result = huber_aggregation(
        gradients,
        client_sizes,
        thresholds
    )

    print("Paper Huber:", paper_result)
    print("Weighted Huber:", weighted_result)

    assert torch.isfinite(paper_result).all()
    assert torch.isfinite(weighted_result).all()

    assert not torch.allclose(
        paper_result,
        weighted_result,
        atol=1e-4
    )


def test_compare_huber_weights_with_same_initial_center():

    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [100.0, 0.0]
    ])

    client_sizes = torch.tensor([100.0, 100.0, 800.0])
    thresholds = torch.tensor([10.0, 10.0, 10.0])

    # Same initial center for both methods
    center = gradients.mean(dim=0)

    distances = torch.linalg.vector_norm(
        gradients - center,
        dim=1
    )

    huber_weights = torch.minimum(
        torch.ones_like(distances),
        thresholds / distances.clamp_min(1e-8)
    )

    # Paper Huber: no client-size weighting
    paper_weights = huber_weights

    paper_result = (
        gradients * paper_weights.unsqueeze(1)
    ).sum(dim=0) / paper_weights.sum()

    # Weighted Huber: client-size weighting
    weighted_weights = client_sizes * huber_weights

    weighted_result = (
        gradients * weighted_weights.unsqueeze(1)
    ).sum(dim=0) / weighted_weights.sum()

    print("Initial center:", center)
    print("Paper Huber (one iteration):", paper_result)
    print("Weighted Huber (one iteration):", weighted_result)

    assert torch.isfinite(paper_result).all()
    assert torch.isfinite(weighted_result).all()

    assert not torch.allclose(
        paper_result,
        weighted_result,
        atol=1e-4
    )

def test_paper_and_weighted_huber_agree_for_equal_sizes():
    gradients = torch.tensor([[2.0, 0.0], [4.0, 0.0], [100.0, 0.0]])
    client_sizes = torch.tensor([100.0, 100.0, 100.0])
    thresholds = torch.tensor([10.0, 10.0, 10.0])

    paper_result = huber_aggregation_paper(gradients, thresholds)
    weighted_result = huber_aggregation(gradients, client_sizes, thresholds)

    assert torch.allclose(paper_result, weighted_result, atol=1e-4)