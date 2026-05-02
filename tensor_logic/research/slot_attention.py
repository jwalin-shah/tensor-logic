import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    def __init__(self, num_slots, input_dim, slot_dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )

    def forward(self, inputs):
        # inputs: (B, N, D)
        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs) # (B, N, D_slot)
        v = self.to_v(inputs) # (B, N, D_slot)

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(sigma) # (B, K, D_slot)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots) # (B, K, D_slot)

            dots = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = F.softmax(dots, dim=1) + self.eps # normalize over slots
            attn = attn / attn.sum(dim=2, keepdim=True) # normalize over inputs
            
            updates = torch.einsum('bkn,bnd->bkd', attn, v)
            
            # Gru update
            slots = self.gru(
                updates.reshape(-1, slots.shape[-1]),
                slots_prev.reshape(-1, slots.shape[-1])
            ).reshape(b, self.num_slots, -1)
            
            # MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

class VisualEncoder(nn.Module):
    def __init__(self, slot_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # (B, 64, 8, 8)
            nn.ReLU(),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 64, 8, 8))

    def forward(self, x):
        features = self.net(x) + self.pos_emb # (B, 64, 8, 8)
        return features.flatten(2).transpose(1, 2) # (B, 64, 64)
