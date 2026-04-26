"""
exp30 — Re-test the exp1 transitive-closure task with -LARGE init for absent edges.

HYPOTHESIS: The 'sigmoid floor' problem from exp1 was caused by initializing
absent-edge logits to 0 (so sigmoid(0)=0.5 leaks everywhere). Initializing
absent edges to a strong negative logit (-LARGE) eliminates the floor and
threshold=0.5 gives perfect F1 at any temperature.

FALSIFIED IF: F1 < 0.95 at any T in [0.1, 2.0] with -LARGE init and threshold=0.5.

SMALLEST TEST: exp1's 5-node graph, threshold=0.5, T sweep, two init regimes.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

N = 5
edge_pairs = [(0, 1), (1, 2), (2, 3), (4, 2)]
true_closure = {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(4,2),(4,3)}
TC_true = torch.zeros(N, N)
for (i, j) in true_closure: TC_true[i, j] = 1.0


def fixpoint(E_logits, T, max_iters=60):
    P = torch.sigmoid(E_logits / T)
    for _ in range(max_iters):
        E_p = torch.sigmoid(E_logits / T)
        composed = torch.einsum("xy,yz->xz", P, E_p)
        new_P_prob = (E_p + composed - E_p * composed).clamp(1e-6, 1 - 1e-6)
        new_L = torch.log(new_P_prob) - torch.log(1 - new_P_prob)
        new_P = torch.sigmoid(new_L / T)
        if torch.allclose(new_P, P, atol=1e-5): break
        P = new_P
    return P


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


print("exp30 — init regime sweep, threshold=0.5\n")
print(f"{'init for absent':<25} {'T':>5}  {'cov':>5}  {'hal':>5}  {'F1':>5}")
print("-" * 60)

for init_absent in [0.0, -1.0, -3.0, -6.0, -10.0]:
    for T in [0.1, 0.3, 1.0, 2.0]:
        E = torch.full((N, N), init_absent)
        for i, j in edge_pairs: E[i, j] = 6.0
        P = fixpoint(E, T)
        cov, hal, f1 = eval_pred(P)
        print(f"  absent_logit={init_absent:>+5.1f}      {T:>5.1f}  {cov:>5.2f}  {hal:>5.2f}  {f1:>5.3f}")
    print()

# Hypothesis check: at -6.0 init, all four T values must give F1 > 0.95
print("HYPOTHESIS CHECK (init=-6.0, all T must give F1 > 0.95):")
all_pass = True
for T in [0.1, 0.3, 1.0, 2.0]:
    E = torch.full((N, N), -6.0)
    for i, j in edge_pairs: E[i, j] = 6.0
    P = fixpoint(E, T)
    _, _, f1 = eval_pred(P)
    flag = "PASS" if f1 > 0.95 else "FAIL"
    if f1 <= 0.95: all_pass = False
    print(f"  T={T:.1f}: F1={f1:.3f}  [{flag}]")

print(f"\n  Verdict: {'CONFIRMED — exp1 floor was an init bug.' if all_pass else 'FALSIFIED'}")
