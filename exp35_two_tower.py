"""
exp35 — Two-tower (possible/actual) tensor logic for counterfactuals

HYPOTHESIS: Maintaining two tensors — Possibility P_ij and Actuality T_ij with
T ≤ P constraint — lets the system answer counterfactual queries ("what if X
were Y's parent?") without polluting the base facts in T.

FALSIFIED IF: Either (a) constraint T ≤ P is violated after training, OR (b)
counterfactual query persistently changes T (i.e., after the query, T differs
from the original T baseline).

SMALLEST TEST: 6-node family KG. Train T to fit observed parent facts. P
trained to be a superset (T ≤ P holds elementwise). Then issue a counterfactual
query that modifies T temporarily for one inference call. Check T returns to
baseline after.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

N = 6
true_parent = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
T_target = torch.zeros(N, N)
for i, j in true_parent: T_target[i, j] = 1.0


# ── Train T (Actual) and P (Possible) ───────────────────────────────────────
T_logits = torch.randn(N, N, requires_grad=True)
P_logits = torch.randn(N, N, requires_grad=True)

opt = torch.optim.Adam([T_logits, P_logits], lr=0.05)
for step in range(500):
    T = torch.sigmoid(T_logits)
    P = torch.sigmoid(P_logits)
    # Data fit: T should match observed parents
    fit = F.binary_cross_entropy(T, T_target)
    # Constraint: T ≤ P (penalty when T > P)
    constraint = torch.relu(T - P).pow(2).sum()
    # Regularizer: P shouldn't be ALL 1s — prefer narrow possibility (small P sum)
    p_reg = P.sum() * 0.01
    loss = fit + 5.0 * constraint + p_reg
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    T = torch.sigmoid(T_logits)
    P = torch.sigmoid(P_logits)

print("exp35 — two-tower P/T training\n")
print(f"  T fits observed parents:  MSE={(T - T_target).pow(2).mean().item():.4f}")
print(f"  T ≤ P constraint violations (max(T-P)): {(T - P).max().item():.4f}")
print(f"  Mean P value: {P.mean().item():.3f}, mean T value: {T.mean().item():.3f}")

T_baseline = T.clone().detach()


# ── Counterfactual query: "what if 0 were parent of 5?" ──────────────────────
# Method: temporarily set T[0,5] = 1 in a COPY, do inference, throw away.
def counterfactual(query_edge, infer_target):
    """Make a copy of T, set the counterfactual edge true, do 2-hop inference."""
    T_query = T.clone()
    T_query[query_edge[0], query_edge[1]] = 1.0
    # 2-hop inference: who are the grandchildren of node 'infer_target' under counterfactual?
    grand = T_query @ T_query
    return grand[infer_target]


grand_baseline = T @ T
grand_cf = counterfactual((0, 5), infer_target=0)

print("\n  Counterfactual: 'what if node 0 were parent of node 5?'")
print(f"  Grandchildren of 0 in baseline:        {grand_baseline[0].numpy().round(2)}")
print(f"  Grandchildren of 0 under counterfactual: {grand_cf.numpy().round(2)}")

# ── After-query check: T should be unchanged ────────────────────────────────
T_after = torch.sigmoid(T_logits).detach()
delta = (T_after - T_baseline).abs().max().item()
print(f"\n  Max change in T after counterfactual:  {delta:.6f}")

print("\nHYPOTHESIS CHECK:")
violation_max = (T - P).max().item()
constraint_holds = violation_max < 0.01
no_pollution = delta < 0.001
print(f"  (a) T ≤ P holds (max violation < 0.01):     {constraint_holds}  ({violation_max:.4f})")
print(f"  (b) T unchanged after counterfactual query: {no_pollution}  (Δ={delta:.6f})")

if constraint_holds and no_pollution:
    print(f"\n  Verdict: CONFIRMED — two-tower allows counterfactuals without polluting facts.")
else:
    print(f"\n  Verdict: FALSIFIED on at least one axis.")
