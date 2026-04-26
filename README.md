# Tensor Logic → Cognition

Working through Pedro Domingos' [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269) (2025) and following the threads where they lead — into architectures, world models, developmental learning, and continual learning.

A learning project, not a product. Each demo is intended to make one idea click viscerally.

## Headline result

On transitive closure (graph reachability), a **3-scalar tensor-logic recurrence** generalizes zero-shot from random 16-node DAGs to real Python import graphs of arbitrary size. Trained once on synthetic data, then evaluated on 8 OSS packages without any fine-tuning:

| package | nodes | TL F1 (3 params) | MLP F1 (37k params, cropped to n=16) |
|---|---|---|---|
| requests | 18 | **1.000** | 0.370 |
| httpx | 23 | **1.000** | 0.414 |
| flask | 24 | **1.000** | 0.322 |
| markdown | 33 | **1.000** | 0.389 |
| tqdm | 31 | **1.000** | 0.519 |
| click | 17 | 0.992 | 0.189 |
| jinja2 | 25 | 0.982 | 0.221 |
| rich | 100 | 0.825 | 0.228 |
| **mean** | — | **0.975** | **0.331** |

The MLP can't be evaluated at native graph sizes (it's trained at fixed n=16), so its score above is the cropped top-16 subgraph — already a much easier task. TL handles any n for free; the MLP has no zero-shot answer for variable-size graphs.

### Pushing to bigger packages (n up to 1,532)

`experiments/exp54_big_imports.py` runs the same TL on substantially larger codebases, with K scaled to graph size:

| package | nodes | density | K | TL F1 |
|---|---|---|---|---|
| fastapi | 48 | 0.104 | 10 | **1.000** |
| networkx | 580 | 0.359 | 14 | 0.967 |
| sqlalchemy | 256 | 0.421 | 12 | 0.879 |
| sympy | 1,532 | 0.254 | 15 | 0.842 |
| django | 899 | 0.101 | 14 | **0.657** |
| **mean** | — | — | — | **0.869** |

TL degrades gracefully on bigger graphs rather than failing catastrophically. Django at n=899 is the weakest case (F1=0.657) — likely a combination of cyclic import structure (Django has many) and the trained α/β/γ being tuned for a sparser regime. This is a real limitation, not a hidden caveat.

### Where this fails entirely

The "TL beats MLPs by 4+ orders of magnitude" story is **task-specific**. A 3-scalar TL recurrence works for transitive closure because closure has a clean closed-form tensor expression. It does **not** work for tasks whose target function isn't expressible in TL's operator basis:

- **XOR / parity** (`experiments/exp48_crossterm_xor.py`, `experiments/exp50_cos_parity.py`): even with a 4-parameter cross-term TL variant, parity is unlearnable — the cross-term cannot create the alternating sign structure XOR requires. Confirmed by trying a cosine activation as a sanity check; it falsifies the "just need a different operator" hypothesis.
- **Code-closure tasks beyond import graphs** (`experiments/exp49_crossterm_imports.py`): when the underlying relation isn't a simple reachability closure (e.g. typed dataflow, control flow with branches), the 3- or 4-scalar TL variants do not capture it.

The honest framing: **TL is enormously parameter-efficient when a closed-form tensor-logic operator exists for the task; it cannot magic one into existence when one doesn't.** See `notes/OPENHUMAN_TL_MEMO.md` for the broader argument about which problem classes this lands on.

### MLP capacity at scale

For completeness on the comparison side: a **71M-parameter MLP fails completely at n=128** on the same closure task (`experiments/exp52_mlp_capacity.py`). The MLP isn't undertrained — it's the wrong inductive bias. This is the headline number for the parameter-ratio claim.

See `experiments/exp53_real_imports.py` (small/mid packages table), `experiments/exp54_big_imports.py` (large packages), `experiments/exp44_import_closure.py` (original single-graph version), `exp48–50` (parity-class failures), `experiments/exp51_bignscale.py` (scaling sweep), `experiments/exp52_mlp_capacity.py`, and `notes/EXPERIMENTS.md` for the full log.

## What's here

| File | What it shows |
|---|---|
| `demos/transitive_closure.py` | Tensor logic in 80 lines. One einsum, three semantics: deductive (step), analogical (sigmoid), embedding-space retrieval. |
| `demos/train_kg.py` | Knowledge graph completion via gradient descent through einsum-rules. Learns object embeddings such that applying `Parent ∘ Parent` produces the correct grandparent matrix. |
| `demos/joint_lm_kg.py` | Joint LM + KG training in one autograd graph. Tiny einsum-form transformer + tensor-logic rule head, sharing token/object embeddings. T-annealing from 1.0 to 0.05. |
| `demos/throwing.py` | Embodied learner: agent learns "force → distance" from 100 random throws, then plans inverse to hit unseen targets. Probes the network and finds an emergent "force magnitude" neuron. |
| `demos/catastrophic_forgetting.py` | Continual learning demo. Naive sequential training forgets Task A completely; EWC (Kirkpatrick 2017) preserves it by anchoring weights important to past tasks. |
| `notes/SESSION_TRANSCRIPT.md` | Full transcript of the conversation that produced this repo. ~25 questions walking from "what is tensor logic" to "how do you build a continual learner that knows over time." |

Beyond the headline demos, the repo also contains `experiments/exp1`–`exp54` and `phase_training/train_phase*.py` — a longer experimental arc probing the limits of tensor-logic operators (parity, code-closure, scaling, capacity). See `notes/EXPERIMENTS.md` for the running log.

## Memos & writing

| File | What it is |
|---|---|
| `notes/EXPERIMENTS.md` | Running log of every experiment, what it tested, what it falsified or confirmed. |
| `notes/RESEARCH_NOTES.md` | Working notes on the underlying research direction. |
| `notes/IDEAS.md` | Open questions and things worth trying next. |
| `notes/OPENHUMAN_TL_MEMO.md` | Memo: tensor logic as a substrate for openhuman, with knowledge-base reframe and related-work convergence. |
| `notes/OPENHUMAN_TL_PROTOTYPE_PLAN.md` | 7-phase, ~14-day prototype plan for a TL+KB go/no-go spike. |

## Run

```bash
uv run --with torch python demos/transitive_closure.py
uv run --with torch python demos/train_kg.py
uv run --with torch python demos/joint_lm_kg.py
uv run --with torch python demos/throwing.py
uv run --with torch python demos/catastrophic_forgetting.py
```

Each runs on CPU in seconds to a few minutes.

## Repo layout

```
demos/             5 headline runnable demos — one idea each
experiments/       exp1..exp54 — the broader experimental arc
phase_training/    train_phase*.py + world/model files (embodied-agent thread)
notes/             long-form memos, research notes, session transcript
```

## The conceptual ladder these demos climb

1. **`demos/transitive_closure.py`**: a rule and an einsum are the same thing. Activation function picks the semantic.
2. **`demos/train_kg.py`**: rules are differentiable. Gradient descent through einsum-rules learns embeddings that make logical inference work.
3. **`demos/joint_lm_kg.py`**: language and symbolic structure can share parameters. One model, two loss terms, three predictions.
4. **`demos/throwing.py`**: concepts (force, distance) emerge as compression coordinates of action-outcome pairs. No textbook required.
5. **`demos/catastrophic_forgetting.py`**: weights move when you train; without selective plasticity, learning new things destroys old. Memory + replay + selective plasticity is how brains avoid this.

## What's tensor logic, in one paragraph

A logical rule and an Einstein-summation over tensors are the same operation. `Path(x,z) :- Edge(x,y), Path(y,z)` becomes `Path = step(Σ_y Path · Edge)` — join on shared indices, multiply, sum-reduce, threshold. Swap boolean semiring for (+,×) on reals → matrix multiply. Swap step for softmax → attention. Swap for sigmoid with temperature `T` → analogical reasoning at `T>0`, strict deduction at `T=0`. One operator (the tensor equation = generalized einsum) expresses transformers, GNNs, RNNs, Datalog programs, probabilistic graphical models, kernel machines. Get GPU + autodiff for free, plus sound logical semantics.

## Where this points

The transcript walks the path from "tensor logic" to a coherent research vision:

- **Strong architectural priors** (Spelke-style core knowledge as tensor-logic rules)
- **Object-centric perception** + equivariance
- **Active embodied learning** with curiosity
- **Episodic + semantic memory** with replay consolidation
- **Curriculum** from primitives to complex
- **SSMs** for efficient long recurrence
- **Tensor logic** as the substrate language tying it all together
- **Tool use** for knowledge externalization
- **Multimodal grounding** eventually

The pieces exist. The integration is the open problem. These demos are the smallest, cleanest versions of each piece.

## What's next

Things worth adding here (in roughly increasing ambition):

- **Schema discovery from rendered video**: agent watches simple physics, proposes tensor-logic rules ("things fall," "no overlap"), verifies on new clips. Genuinely novel research at toy scale.
- **Curiosity-driven explorer**: agent in a gridworld, intrinsic reward = own prediction error. Watch what it explores first.
- **Object permanence via violation-of-expectation**: train a forward model on physics with impossible events. Measure surprise. Compare a model with built-in solidity rules vs one without.
- **Tensor-logic-augmented SSM**: Mamba where the recurrence step is a tensor-logic rule application. Linear-cost reasoning over a typed state.

## References

- Domingos, P. (2025). *Tensor Logic: The Language of AI*. arXiv:2510.12269
- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS.
- McClelland, McNaughton, O'Reilly (1995). *Why there are complementary learning systems in the hippocampus and neocortex*.
- Spelke, E. (2003). *Core knowledge*. American Psychologist.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*.
- Tenenbaum et al. (2017). *Building machines that learn and think like people*.
- "Downloading Kung Fu" — [statedpreferences.com/essays/downloading-kung-fu](https://statedpreferences.com/essays/downloading-kung-fu/) (the corrected-practice argument from the pedagogy side).

## Citing the paper

This repo is built on Pedro Domingos' *Tensor Logic*. If you write about anything here, cite the paper:

```bibtex
@article{domingos2025tensorlogic,
  title   = {Tensor Logic: The Language of AI},
  author  = {Domingos, Pedro},
  journal = {arXiv preprint arXiv:2510.12269},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.12269}
}
```

## License

MIT. Use, modify, share.
