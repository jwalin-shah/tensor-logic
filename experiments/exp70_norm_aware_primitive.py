"""
exp70: Norm-aware optimizers on the TL primitive-set search.

Bernstein & Newhouse, "Old Optimizer, New Norm" (arXiv:2409.20325) recast
Adam, Shampoo, Prodigy etc. as steepest descent under a particular operator
norm assigned to each tensor based on its architectural role. The practical
upshot is that different tensor roles deserve different update geometries.

In exp69 we have a single learnable tensor theta in R^N — an indicator-vector
role. The vector entries have wildly different "natural scales": at the
optimum, theta is large positive on prime indices (sigmoid -> ~1), large
negative elsewhere (sigmoid -> ~0), with a thin transition band. This is
exactly the regime where coordinate-wise normalization (Adam / signSGD)
should beat unnormalized SGD: the gradient magnitude per coordinate carries
no useful information about how far that coordinate has to move; the *sign*
does.

We rerun PROBE A from exp69 (cold start, log-linear lambda anneal from 1e-3
to 1e2 over 3000 steps) under four updaters:

  SGD      — vanilla, no normalization
  signSGD  — sign(grad)  (Lion-style; aligned with the entry-wise/Linf norm
              perspective from the paper for a vector role)
  Adam     — second-moment normalization (the implicit-norm baseline of the
              paper)
  NormSGD  — grad / ||grad||_2  (whole-tensor 2-norm steepest descent)

Each is given the same nominal step size (with one Adam baseline at its
canonical 5e-2 for reference). We report:

  - final f(s)  and  v(s)  (target: f -> ~ f(primes), v -> 0)
  - rounded set size, primitivity, f after greedy primitive reduction
  - L1 distance from the primes indicator (smaller = closer to the
    conjectured extremum)

The point is not "X beats Y by Δ on a benchmark." The point is to show that
on this small TL math-search problem, the *role-aware* updates (signSGD,
Adam) recover the primes essentially exactly, while plain SGD and whole-
tensor normalized SGD do not. That mirrors the paper's prescription: pick
the norm that matches the tensor role.
"""

import math
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp68_erdos_primitive_sum as e68
import exp69_tl_primitive_search as e69


N = 2000
N_STEPS = 3000
LAM_LO = 1e-3
LAM_HI = 1e2
SEED = 0


def run_with_updater(D, w, updater, lr, n_steps=N_STEPS, lam_lo=LAM_LO, lam_hi=LAM_HI, seed=SEED):
    """updater(theta, grad, state) -> updates theta in-place, returns state."""
    torch.manual_seed(seed)
    theta = torch.randn(N, dtype=torch.float64) * 0.1
    state = {}
    log_lo, log_hi = math.log(lam_lo), math.log(lam_hi)
    for t in range(n_steps):
        lam = math.exp(log_lo + (log_hi - log_lo) * (t / max(n_steps - 1, 1)))
        s = torch.sigmoid(theta).requires_grad_(False)
        # explicit gradient: dL/dtheta = -dL/ds * s*(1-s) where dL/ds is computed
        # but cleanest is autograd. Re-enable theta autograd:
        theta.requires_grad_(True)
        s = torch.sigmoid(theta)
        f = torch.einsum("a,a->", s, w)
        v = torch.einsum("a,ab,b->", s, D, s)
        loss = -f + lam * v
        if theta.grad is not None:
            theta.grad.zero_()
        loss.backward()
        with torch.no_grad():
            updater(theta, theta.grad, state, lr, t)
        theta.requires_grad_(False)
    with torch.no_grad():
        s_final = torch.sigmoid(theta)
        f_final = torch.einsum("a,a->", s_final, w).item()
        v_final = torch.einsum("a,ab,b->", s_final, D, s_final).item()
    return s_final.detach(), f_final, v_final


def upd_sgd(theta, grad, state, lr, t):
    theta.add_(grad, alpha=-lr)


def upd_signsgd(theta, grad, state, lr, t):
    theta.add_(grad.sign(), alpha=-lr)


def upd_adam(theta, grad, state, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    if "m" not in state:
        state["m"] = torch.zeros_like(theta)
        state["v"] = torch.zeros_like(theta)
    m = state["m"]
    v = state["v"]
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bc1 = 1 - beta1 ** (t + 1)
    bc2 = 1 - beta2 ** (t + 1)
    m_hat = m / bc1
    v_hat = v / bc2
    theta.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


def upd_normsgd(theta, grad, state, lr, t, eps=1e-8):
    norm = grad.norm() + eps
    theta.add_(grad / norm, alpha=-lr)


def main():
    print(f"=== exp70: norm-aware optimizers on TL primitive-set search ===")
    print(f"N={N}  steps={N_STEPS}  anneal lambda 1e-3 -> 1e2  seed={SEED}")
    D = e68.divisibility_tensor(N).to(torch.float64)
    w = e68.erdos_weights(N)
    primes_N = e68.primes_below(N + 1)
    f_primes = e69.f_of_set(primes_N)
    s_primes = torch.zeros(N, dtype=torch.float64)
    for p in primes_N:
        if 2 <= p <= N + 1:
            s_primes[p - 2] = 1.0
    print(f"baseline f(primes_2..{N+1}) = {f_primes:.6f}  (|primes|={len(primes_N)})")

    configs = [
        ("SGD",         upd_sgd,     5e-1),
        ("signSGD",     upd_signsgd, 5e-2),
        ("Adam",        upd_adam,    5e-2),
        ("NormSGD",     upd_normsgd, 5e0),
    ]

    print(f"\n  {'optim':<10}{'lr':<8}{'f_soft':<10}{'v_soft':<10}{'|S>0.5|':<10}"
          f"{'prim?':<8}{'f_greedy':<12}{'L1 to primes':<14}")
    rows = []
    for name, upd, lr in configs:
        s_final, f_final, v_final = run_with_updater(D, w, upd, lr)
        rounded, prim, f_r, greedy, f_g = e69.round_and_greedy(s_final)
        l1_to_primes = (s_final - s_primes).abs().sum().item()
        rows.append((name, lr, f_final, v_final, len(rounded), prim, f_g, l1_to_primes))
        print(f"  {name:<10}{lr:<8.3g}{f_final:<10.4f}{v_final:<10.4f}{len(rounded):<10}"
              f"{str(prim):<8}{f_g:<12.4f}{l1_to_primes:<14.4f}")

    print(f"\nReference: f(primes) = {f_primes:.4f}, |primes| = {len(primes_N)}")
    print("Reading: optimizers whose 'f_greedy' equals f(primes) and whose 'L1 to primes'")
    print("is small are recovering the conjectured extremum. The expectation from the")
    print("Bernstein-Newhouse framing is that coordinate-wise normalized updates (Adam,")
    print("signSGD) match the indicator-vector role and outperform un-normalized SGD")
    print("and whole-tensor NormSGD on this problem.")


if __name__ == "__main__":
    main()
