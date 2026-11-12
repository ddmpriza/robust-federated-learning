
import torch

from src.gradient_aggregation import geometric_median


def test_outlier_gradient():
    gradients = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [100.0, 0.0]
    ])

    mean = gradients.mean(dim=0)
    median = geometric_median(gradients)

    expected = torch.tensor([4.0, 0.0])

    print("Arithmetic mean:", mean)
    print("Geometric median:", median)

    assert torch.allclose(median, expected, atol=1e-3)

    print("Outlier gradient test passed!")


def test_identical_gradients():
    gradients = torch.tensor([
        [3.0, 5.0],
        [3.0, 5.0],
        [3.0, 5.0]
    ])

    median = geometric_median(gradients)

    expected = torch.tensor([3.0, 5.0])

    assert torch.allclose(median, expected, atol=1e-6)

    print("Identical gradients test passed!")

def test_two_identical_gradients():
    gradients = torch.tensor([
        [3.0, 5.0],
        [3.0, 5.0],
        [9.0, 5.0]
    ])

    median = geometric_median(gradients)

    expected = torch.tensor([3.0, 5.0])

    assert torch.allclose(median, expected, atol=1e-6)

    print("Two identical gradients test passed!")

def test_initial_center_matches_gradient():
    gradients = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [12.0, 0.0],
        [3.0, 0.0]
    ])

    median = geometric_median(gradients)

    expected = torch.tensor([0.0, 0.0])
    print("Expected:", expected)
    print("Actual:", median)

    assert torch.allclose(median, expected, atol=1e-6)

    print("Initial center matches gradient test passed!")

def test_single_gradient():
    gradients = torch.tensor([[3., 5.]])
    median = geometric_median(gradients)

    expected = torch.tensor([3., 5.])

    assert torch.allclose(median, expected, atol=1e-6)
    print("Single gradient test passed!")

def test_high_dimensional_gradients():
    gradients = torch.randn(5, 101770)

    median = geometric_median(gradients)

    assert median.shape == (101770,)
    assert torch.isfinite(median).all()

    print("High-dimensional gradients test passed!")

if __name__ == "__main__":
    test_outlier_gradient()
    test_identical_gradients()
    test_two_identical_gradients()
    test_initial_center_matches_gradient()
    test_single_gradient()
    test_high_dimensional_gradients()