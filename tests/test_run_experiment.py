import pytest
import torch

from run_experiment import apply_attack, robust_threshold, aggregate
from src.gradient_aggregation import gradient_mean


def make_gradients():
    return torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])


# ----------------------------------------------------------------------------
# apply_attack
# ----------------------------------------------------------------------------
def test_no_attack_returns_unchanged_copy():
    gradients = make_gradients()

    result = apply_attack(gradients, [0, 1], "none", scale=5.0)

    assert torch.equal(result, gradients)
    assert result is not gradients


def test_sign_flip_changes_only_byzantine_clients():
    gradients = make_gradients()

    result = apply_attack(gradients, [1], "sign_flip", scale=5.0)

    assert torch.equal(result[1], -5.0 * gradients[1])
    for honest_id in (0, 2, 3):
        assert torch.equal(result[honest_id], gradients[honest_id])


def test_attack_does_not_modify_the_input():
    gradients = make_gradients()
    original = gradients.clone()

    apply_attack(gradients, [0, 1], "sign_flip", scale=5.0)

    assert torch.equal(gradients, original)


def test_no_byzantine_clients_means_no_change():
    gradients = make_gradients()

    result = apply_attack(gradients, [], "sign_flip", scale=5.0)

    assert torch.equal(result, gradients)


def test_unknown_attack_raises():
    with pytest.raises(ValueError):
        apply_attack(make_gradients(), [0], "unknown_attack", scale=1.0)


# ----------------------------------------------------------------------------
# robust_threshold
# ----------------------------------------------------------------------------
def test_threshold_scales_linearly_with_c():
    gradients = make_gradients()

    t1 = robust_threshold(gradients, 1.0)
    t2 = robust_threshold(gradients, 2.0)

    assert t2 == pytest.approx(2.0 * t1)


def test_threshold_is_positive_for_identical_gradients():
    gradients = torch.ones(5, 3)

    assert robust_threshold(gradients, 1.0) > 0


def test_threshold_is_not_dominated_by_one_extreme_client():
    honest = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.05, 0.05],
    ])
    with_outlier = torch.cat([honest, torch.tensor([[1e6, 1e6]])])

    assert robust_threshold(with_outlier, 1.0) < 1.0


# ----------------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------------
def test_mean_matches_gradient_mean():
    gradients = make_gradients()
    sizes = torch.tensor([1, 2, 3, 4])

    result, threshold = aggregate("mean", gradients, sizes, {"huber_c": 1.0})

    assert threshold is None
    assert torch.allclose(result, gradient_mean(gradients, sizes))


def test_huber_with_huge_c_equals_weighted_mean():
    gradients = make_gradients()
    sizes = torch.tensor([1, 2, 3, 4])

    result, threshold = aggregate("huber", gradients, sizes, {"huber_c": 1e6})

    assert threshold > 0
    assert torch.allclose(result, gradient_mean(gradients, sizes), atol=1e-5)


def make_gradients_with_outlier():
    honest = torch.tensor([
        [1.0, 1.0],
        [1.1, 0.9],
        [0.9, 1.1],
        [1.0, 1.0],
    ])
    return torch.cat([honest, torch.tensor([[-50.0, -50.0]])])


@pytest.mark.parametrize("method", ["huber", "geometric_median"])
def test_robust_aggregators_resist_an_outlier_better_than_mean(method):
    gradients = make_gradients_with_outlier()
    sizes = torch.ones(5)
    config = {"huber_c": 1.0}
    target = torch.tensor([1.0, 1.0])

    robust, _ = aggregate(method, gradients, sizes, config)
    mean, _ = aggregate("mean", gradients, sizes, config)

    assert torch.linalg.vector_norm(robust - target) < torch.linalg.vector_norm(mean - target)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        aggregate("unknown_method", make_gradients(), torch.ones(4), {"huber_c": 1.0})