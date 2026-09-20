import torch

from tensor_logic.predictive_world import (
    PredictiveWorldModel,
    jepa_cosine_loss,
    latent_transition_error,
)


def test_predictive_world_shapes_and_stop_gradient_target():
    model = PredictiveWorldModel(
        state_dim=6,
        action_dim=2,
        latent_dim=4,
        hidden_dim=8,
    )
    state = torch.randn(3, 6, requires_grad=True)
    action = torch.randn(3, 2)
    target_state = torch.randn(3, 6, requires_grad=True)

    predicted, target = model(state, action, target_state)

    assert predicted.shape == (3, 4)
    assert target.shape == (3, 4)
    assert target.requires_grad is False

    loss = jepa_cosine_loss(predicted, target)
    loss.backward()

    assert state.grad is not None
    assert target_state.grad is None


def test_cosine_loss_is_zero_for_identical_nonzero_latents():
    latent = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0]]
    )
    loss = jepa_cosine_loss(latent, latent)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_latent_transition_error_is_per_example():
    predicted = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0]]
    )
    target = torch.zeros_like(predicted)

    error = latent_transition_error(predicted, target)

    assert torch.allclose(error, torch.tensor([0.0, 5.0]))


def test_invalid_dimensions_fail_closed():
    model = PredictiveWorldModel(
        state_dim=4,
        action_dim=2,
        latent_dim=3,
    )
    try:
        model.predict_latent(
            torch.zeros(1, 4),
            torch.zeros(1, 3),
        )
    except ValueError as exc:
        assert "action must have shape" in str(exc)
    else:
        raise AssertionError("invalid action shape should fail")
