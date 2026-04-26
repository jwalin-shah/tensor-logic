"""
exp33 — Tighter logit-space soft-OR operators

HYPOTHESIS: A tighter soft-OR operator (not logsumexp) makes log-odds tensor
logic give F1 ≥ 0.95 on the exp1 transitive-closure task at threshold=0.5.

FALSIFIED IF: All four candidate operators (max-pool, T-scaled logsumexp,
noisy-OR-in-logit, temperature-annealed logsumexp) give F1 < 0.95.

SMALLEST TEST: Same 5-node graph as exp1; init absent edges to -6.0 logit;
4 fixpoint variants; compare.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

N = 5
edge_pairs = [(0, 1), (1, 2), (2, 3), (4, 2)]
true_closure = {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(4,2),(4,3)}
TC_true = torch.zeros(N, N)
for (i, j) in true_closure: TC_true[i, j] = 1.0

LARGE = 6.0
E_logits = torch.full((N, N), -LARGE)
for i, j in edge_pairs: E_logits[i, j] = +LARGE


def fixpoint_logsumexp(max_iters=60, T=1.0):
    """Soft-AND via sum, soft-OR over y via logsumexp. The exp29 baseline."""
    L = E_logits.clone()
    for _ in range(max_iters):
        comp = torch.logsumexp((L.unsqueeze(2) + E_logits.unsqueeze(0)) / T, dim=1) * T
        new_L = torch.logsumexp(torch.stack([E_logits, comp], dim=0) / T, dim=0) * T
        new_L = new_L.clamp(-LARGE * 2, LARGE * 2)
        if torch.allclose(new_L, L, atol=1e-4): break
        L = new_L
    return torch.sigmoid(L)


def fixpoint_max(max_iters=60):
    """Soft-AND via sum, OR via max (hard max in logit space)."""
    L = E_logits.clone()
    for _ in range(max_iters):
        # composition: max over y of (L[x,y] + E[y,z])  ← hard max, not logsumexp
        comp = (L.unsqueeze(2) + E_logits.unsqueeze(0)).max(dim=1).values
        new_L = torch.maximum(E_logits, comp)
        new_L = new_L.clamp(-LARGE * 2, LARGE * 2)
        if torch.allclose(new_L, L, atol=1e-4): break
        L = new_L
    return torch.sigmoid(L)


def fixpoint_noisyor(max_iters=60):
    """Proper noisy-OR in logit space.
    p_xz = 1 - prod_y(1 - p_xy * p_yz)  for direct edges combined with composition.
    Implemented via prob space then re-converted.
    """
    P = torch.sigmoid(E_logits)
    for _ in range(max_iters):
        E_p = torch.sigmoid(E_logits)
        # composed: p_xz_via_y = p_xy * p_yz; combine via 1 - prod_y(1 - p_xy*p_yz)
        prod = torch.ones(N, N)
        for y in range(N):
            prod = prod * (1 - P[:, y:y+1] * E_p[y:y+1, :])
        composed = 1 - prod
        # combine direct + composed via noisy-OR
        new_P = 1 - (1 - E_p) * (1 - composed)
        new_P = new_P.clamp(1e-6, 1 - 1e-6)
        if torch.allclose(new_P, P, atol=1e-5): break
        P = new_P
    return P


def fixpoint_logsumexp_lowT(max_iters=60):
    """Logsumexp with very low T → behaves like max."""
    return fixpoint_logsumexp(max_iters=max_iters, T=0.1)


def eval_pred(P, threshold=0.5):
    pred = (P >= threshold).float()
    tp = (pred * TC_true).sum().item()
    fp = (pred * (1 - TC_true)).sum().item()
    fn = ((1 - pred) * TC_true).sum().item()
    cov = tp / max(TC_true.sum().item(), 1)
    hal = fp / max((1 - TC_true).sum().item(), 1)
    prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return cov, hal, f1


print("exp33 — log-odds soft-OR operators (init=-6, threshold=0.5)\n")
print(f"{'method':<35} {'cov':>5}  {'hal':>5}  {'F1':>5}")
print("-" * 60)

for name, fn in [
    ("logsumexp T=1.0  (exp29 baseline)", fixpoint_logsumexp),
    ("logsumexp T=0.1  (low-T)",          fixpoint_logsumexp_lowT),
    ("max-pool (hard max)",               fixpoint_max),
    ("noisy-OR (probability space)",      fixpoint_noisyor),
]:
    P = fn()
    cov, hal, f1 = eval_pred(P)
    print(f"  {name:<33} {cov:>5.2f}  {hal:>5.2f}  {f1:>5.3f}")

print("\nHYPOTHESIS CHECK (any operator with F1 ≥ 0.95?):")
results = []
for name, fn in [("logsumexp T=1.0", fixpoint_logsumexp),
                 ("logsumexp T=0.1", fixpoint_logsumexp_lowT),
                 ("max-pool", fixpoint_max),
                 ("noisy-OR", fixpoint_noisyor)]:
    P = fn(); _, _, f1 = eval_pred(P)
    results.append((name, f1))
winners = [n for n, f1 in results if f1 >= 0.95]
print(f"  Operators clearing F1=0.95: {winners if winners else 'NONE'}")
verdict = "CONFIRMED — at least one tight operator works." if winners else "FALSIFIED — no operator clears the bar."
print(f"  Verdict: {verdict}")
