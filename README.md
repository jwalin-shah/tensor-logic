# Tensor Logic → Cognition

Working through Pedro Domingos' [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269) (2025) and following the threads where they lead — into architectures, world models, developmental learning, and continual learning.

A learning project, not a product. Each demo is intended to make one idea click viscerally.

## What's here

| File | What it shows |
|---|---|
| `transitive_closure.py` | Tensor logic in 80 lines. One einsum, three semantics: deductive (step), analogical (sigmoid), embedding-space retrieval. |
| `train_kg.py` | Knowledge graph completion via gradient descent through einsum-rules. Learns object embeddings such that applying `Parent ∘ Parent` produces the correct grandparent matrix. |
| `joint_lm_kg.py` | Joint LM + KG training in one autograd graph. Tiny einsum-form transformer + tensor-logic rule head, sharing token/object embeddings. T-annealing from 1.0 to 0.05. |
| `throwing.py` | Embodied learner: agent learns "force → distance" from 100 random throws, then plans inverse to hit unseen targets. Probes the network and finds an emergent "force magnitude" neuron. |
| `catastrophic_forgetting.py` | Continual learning demo. Naive sequential training forgets Task A completely; EWC (Kirkpatrick 2017) preserves it by anchoring weights important to past tasks. |
| `curiosity_gridworld.py` | Curiosity-driven explorer in a gridworld. Intrinsic reward is one-step prediction error of its own world model; compared against a random baseline for exploration coverage. |
| `SESSION_TRANSCRIPT.md` | Full transcript of the conversation that produced this repo. ~25 questions walking from "what is tensor logic" to "how do you build a continual learner that knows over time." |

## Run

```bash
uv run --with torch python transitive_closure.py
uv run --with torch python train_kg.py
uv run --with torch python joint_lm_kg.py
uv run --with torch python throwing.py
uv run --with torch python catastrophic_forgetting.py
uv run --with torch python curiosity_gridworld.py
```

Each runs on CPU in seconds to a few minutes.

## The conceptual ladder these demos climb

1. **`transitive_closure.py`**: a rule and an einsum are the same thing. Activation function picks the semantic.
2. **`train_kg.py`**: rules are differentiable. Gradient descent through einsum-rules learns embeddings that make logical inference work.
3. **`joint_lm_kg.py`**: language and symbolic structure can share parameters. One model, two loss terms, three predictions.
4. **`throwing.py`**: concepts (force, distance) emerge as compression coordinates of action-outcome pairs. No textbook required.
5. **`catastrophic_forgetting.py`**: weights move when you train; without selective plasticity, learning new things destroys old. Memory + replay + selective plasticity is how brains avoid this.
6. **`curiosity_gridworld.py`**: intrinsic motivation can be implemented as prediction-error seeking. Exploration emerges from model surprise, not external reward.

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

## License

MIT. Use, modify, share.
