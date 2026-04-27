import torch


def boolean_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return ((A @ B) > 0).to(A.dtype)


def gf2_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.remainder(A.to(torch.int64) @ B.to(torch.int64), 2).to(A.dtype)


def reliability_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Independent path reliability: 1 - product(1 - p_path)."""
    paths = A.unsqueeze(-1) * B.unsqueeze(0)
    return 1.0 - torch.prod(1.0 - paths, dim=1)
