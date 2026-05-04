"""
exp82: TL-augmented Transformer (Proof of Concept)
==================================================
Embeds a differentiable Tensor Logic (TL) closure layer into a Transformer.
Solves multi-hop relational reasoning by replacing expensive attention-based
search with a deterministic, differentiable logic circuit in the forward pass.

Key Idea:
1. Residual Stream -> Relation Projector -> NxN Tensor
2. NxN Tensor -> K steps of Boolean-style matmul (Differentiable TL) -> Derived Tensor
3. Derived Tensor -> Output Projector -> Residual Stream

Synthetic Task: Kinship reachability (e.g., "Is Alice the great-aunt of Dave?").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Differentiable TL Layer ----------

class TLLayer(nn.Module):
    def __init__(self, d_model, n_entities, n_relations, k_steps=3):
        super().__init__()
        self.d_model = d_model
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.k_steps = k_steps
        
        # Project hidden state to initial relation tensors (NxNxR)
        self.to_rels = nn.Linear(d_model, n_entities * n_entities * n_relations)
        
        # Logic parameters: which relations can be composed?
        # We learn a 'composition weight' matrix (R, R, R)
        # where C[i,j,k] is the weight of (Rel_i @ Rel_j) contributing to Rel_k
        self.comp_weights = nn.Parameter(torch.randn(n_relations, n_relations, n_relations) * 0.02)
        
        # Project back to residual stream
        self.to_out = nn.Linear(n_entities * n_entities * n_relations, d_model)

    def forward(self, x):
        # x: (B, L, D) - we take the mean or last token to represent the 'world state'
        # For simplicity in this PoC, we assume x is the latent representation of the KB
        z = x.mean(dim=1)  # (B, D)
        
        # 1. Project to NxNxR
        rels = self.to_rels(z).view(-1, self.n_relations, self.n_entities, self.n_entities)
        rels = torch.sigmoid(rels) # Bound facts between 0 and 1
        
        # 2. Differentiable Closure
        # For K steps: R_k = sigmoid(sum_{i,j} W_{ijk} * (R_i @ R_j))
        for _ in range(self.k_steps):
            # rels: (B, R, N, N)
            # We want all-pairs matmul: (B, R, R, N, N)
            # This is expensive. Instead, we use a weighted sum of compositions.
            
            # (B, R, N, N) -> (B, R, 1, N, N)
            r_left = rels.unsqueeze(2)
            # (B, R, N, N) -> (B, 1, R, N, N)
            r_right = rels.unsqueeze(1)
            
            # Composition: (B, R, R, N, N)
            composed = torch.matmul(r_left, r_right)
            
            # Apply learned logic weights: (R, R, R)
            # out_rels[k] = sum_{i,j} comp_weights[i,j,k] * composed[i,j]
            new_rels = torch.einsum("ijk,bijxy->bkxy", self.comp_weights, composed)
            
            # Residual-like update with sigmoid to maintain fact-like properties
            rels = torch.sigmoid(rels + new_rels)
            
        # 3. Project back
        out = self.to_out(rels.view(-1, self.n_relations * self.n_entities * self.n_entities))
        return x + out.unsqueeze(1)

# ---------- Minimal Transformer ----------

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, n_entities=8, n_relations=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        layers = []
        for i in range(n_layers):
            layers.append(nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, batch_first=True))
            # Inject TL-Layer in the middle
            if i == n_layers // 2:
                layers.append(TLLayer(d_model, n_entities, n_relations))
        
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        b, t = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        
        # Correctly pass through the mixed sequence of layers
        for block in self.blocks:
            x = block(x)
            
        return self.head(x)

# ---------- Synthetic Kinship Task ----------

def generate_kinship_data(n_samples=1000, n_entities=8):
    # entities: 0..7
    # relations: 0:parent, 1:sibling
    # target: 2:grandparent (parent @ parent), 3:uncle (sibling @ parent)
    
    data = []
    for _ in range(n_samples):
        # Generate random base facts
        base_facts = []
        for _ in range(12):
            r = torch.randint(0, 2, (1,)).item()
            s = torch.randint(0, n_entities, (1,)).item()
            d = torch.randint(0, n_entities, (1,)).item()
            if s != d:
                base_facts.append((r, s, d))
        
        # Pick a query
        q_type = torch.randint(0, 2, (1,)).item() # 0:grandparent, 1:uncle
        qs = torch.randint(0, n_entities, (1,)).item()
        qd = torch.randint(0, n_entities, (1,)).item()
        
        # Compute ground truth via matrix logic
        m = {0: torch.zeros(n_entities, n_entities), 1: torch.zeros(n_entities, n_entities)}
        for r, s, d in base_facts:
            m[r][s, d] = 1.0
            
        if q_type == 0: # grandparent
            res = (m[0] @ m[0]) > 0
        else: # uncle
            res = (m[1] @ m[0]) > 0
            
        label = int(res[qs, qd].item())
        
        # Format as sequence of tokens
        # [R S D] ... [QUERY_R QS QD] -> [LABEL]
        seq = []
        for r, s, d in base_facts:
            seq.extend([r + 10, s + 20, d + 30]) # Offset to distinguish
        seq.extend([q_type + 40, qs + 20, qd + 30])
        data.append((torch.tensor(seq), label))
    return data

def train():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    
    print(f"Training on {device}...")
    
    n_entities = 8
    n_relations = 4 # parent, sibling, grandparent, uncle
    model = MiniGPT(vocab_size=100, n_entities=n_entities, n_relations=n_relations).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_data = generate_kinship_data(5000)
    
    for epoch in range(10):
        total_loss = 0
        correct = 0
        for seq, label in train_data[:1000]: # Mini batch
            optimizer.zero_grad()
            logits = model(seq.unsqueeze(0).to(device))
            # Prediction is on the last token's head
            pred_logit = logits[0, -1, :2] # binary label 0 or 1
            loss = F.cross_entropy(pred_logit.unsqueeze(0), torch.tensor([label]).to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if torch.argmax(pred_logit).item() == label:
                correct += 1
                
        print(f"Epoch {epoch} | Loss: {total_loss/1000:.4f} | Acc: {correct/1000:.2f}")

if __name__ == "__main__":
    train()
