import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    # Token vocabularies
    N_X_TOKENS: int = 129       # 0 = PAD, 1..128 = x bins
    N_Y_TOKENS: int = 129       # 0 = PAD, 1..128 = y bins
    N_E_TOKENS: int = 112       # 0 = PAD, 1..111 = energy bins

    # Maximum number of hits in one event
    N_MAX_HITS: int = 192

    # Transformer
    EMBED_DIM: int = 256
    NUM_HEADS: int = 8
    NUM_LAYERS: int = 4
    DROPOUT: float = 0.1

    # Experts
    NUM_EXPERTS: int = 4
    EXPERT_HIDDEN_DIM: int = 512

    # Conditioning ranges used for normalization
    MOMENTUM_MIN: float = 0.5
    MOMENTUM_MAX: float = 20.0

    N_MIN: float = 1.018
    N_MAX: float = 1.026


class ConditionEmbedding(nn.Module):
    """
    Converts continuous momentum and refractive index into
    a learned conditioning vector.
    """

    def __init__(self, config):
        super().__init__()

        self.momentum_min = config.MOMENTUM_MIN
        self.momentum_max = config.MOMENTUM_MAX

        self.n_min = config.N_MIN
        self.n_max = config.N_MAX

        self.mlp = nn.Sequential(
            nn.Linear(2, config.EMBED_DIM),
            nn.GELU(),
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM),
            nn.GELU(),
        )

    def forward(self, momentum, refractive_index):
        # Normalize to approximately [-1, 1]
        p = 2.0 * (
            (momentum - self.momentum_min)
            / (self.momentum_max - self.momentum_min)
        ) - 1.0

        n = 2.0 * (
            (refractive_index - self.n_min)
            / (self.n_max - self.n_min)
        ) - 1.0

        conditions = torch.stack([p, n], dim=-1)

        return self.mlp(conditions)


class Expert(nn.Module):
    """
    Small feed-forward expert.

    Each expert receives the same hidden representation but learns
    a different physical correction.
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class FactorizedMoE(nn.Module):
    """
    Four experts:

        Expert 0 -> refractive-index expert
        Expert 1 -> momentum expert
        Expert 2 -> reserved/frozen expert
        Expert 3 -> reserved/frozen expert

    During foundation training:

        output = x + N_expert(x) + P_expert(x)

    Reserved experts are present in the model but do not participate
    in foundation training.

    They can later be activated and fine-tuned for new conditions.
    """

    def __init__(self, config):
        super().__init__()

        self.experts = nn.ModuleList([
            Expert(
                config.EMBED_DIM,
                config.EXPERT_HIDDEN_DIM
            )
            for _ in range(config.NUM_EXPERTS)
        ])

        # Expert 0:
        # refractive-index adaptation
        self.n_expert = self.experts[0]

        # Expert 1:
        # momentum adaptation
        self.momentum_expert = self.experts[1]

        # Experts 2 and 3 are reserved for later fine-tuning
        self.reserved_expert_2 = self.experts[2]
        self.reserved_expert_3 = self.experts[3]

        # Initialize reserved experts from trained experts later.
        # For now they are frozen.
        for parameter in self.reserved_expert_2.parameters():
            parameter.requires_grad = False

        for parameter in self.reserved_expert_3.parameters():
            parameter.requires_grad = False

        self.active_mode = "foundation"

    def initialize_reserved_experts(self):
        """
        Copy the foundation experts into the two reserved experts.

        This should be called AFTER foundation training and before
        fine-tuning on a new condition.
        """

        self.reserved_expert_2.load_state_dict(
            self.n_expert.state_dict()
        )

        self.reserved_expert_3.load_state_dict(
            self.momentum_expert.state_dict()
        )

        # Keep them frozen until explicitly enabled.
        for parameter in self.reserved_expert_2.parameters():
            parameter.requires_grad = False

        for parameter in self.reserved_expert_3.parameters():
            parameter.requires_grad = False

    def enable_reserved_expert(self, expert_id):
        """
        Enable one reserved expert for later fine-tuning.
        """

        if expert_id == 2:
            expert = self.reserved_expert_2

        elif expert_id == 3:
            expert = self.reserved_expert_3

        else:
            raise ValueError(
                "Only reserved experts 2 and 3 can be enabled."
            )

        for parameter in expert.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        """
        Foundation mode:

            x
            + refractive-index expert
            + momentum expert

        This is a residual factorized MoE rather than a conventional
        single-choice sparse router.
        """

        n_correction = self.n_expert(x)
        p_correction = self.momentum_expert(x)

        output = x + n_correction + p_correction

        return output


class TransformerMoEBlock(nn.Module):
    """
    Transformer block containing self-attention followed by
    the factorized MoE.
    """

    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.EMBED_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=config.EMBED_DIM,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(config.EMBED_DIM)

        self.moe = FactorizedMoE(config)

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x

        x_norm = self.norm1(x)

        key_padding_mask = None

        if attention_mask is not None:
            # attention_mask:
            # True  = valid token
            # False = padding
            key_padding_mask = ~attention_mask

        attended, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = residual + self.dropout(attended)

        # MoE
        residual = x

        x = self.norm2(x)
        x = self.moe(x)

        x = residual + self.dropout(x)

        return x


class CherenkovGenerator(nn.Module):
    """
    Conditional non-autoregressive Cherenkov hit-pattern generator.

    Input:
        momentum
        refractive_index

    Output:
        occupancy
        x position
        y position
        photon energy
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = GeneratorConfig()

        self.config = config

        # ---------------------------------------------------------
        # Condition embedding
        # ---------------------------------------------------------

        self.condition_embedding = ConditionEmbedding(config)

        # ---------------------------------------------------------
        # Learned query slots
        # ---------------------------------------------------------

        self.query_slots = nn.Parameter(
            torch.randn(
                config.N_MAX_HITS,
                config.EMBED_DIM
            ) * 0.02
        )

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        self.layers = nn.ModuleList([
            TransformerMoEBlock(config)
            for _ in range(config.NUM_LAYERS)
        ])

        self.final_norm = nn.LayerNorm(config.EMBED_DIM)

        # ---------------------------------------------------------
        # Output heads
        # ---------------------------------------------------------

        # HIT / EMPTY
        self.occupancy_head = nn.Linear(
            config.EMBED_DIM,
            2
        )

        # x coordinate
        self.x_head = nn.Linear(
            config.EMBED_DIM,
            config.N_X_TOKENS
        )

        # y coordinate
        self.y_head = nn.Linear(
            config.EMBED_DIM,
            config.N_Y_TOKENS
        )

        # photon energy
        self.energy_head = nn.Linear(
            config.EMBED_DIM,
            config.N_E_TOKENS
        )

    def forward(
        self,
        momentum,
        refractive_index,
        attention_mask=None,
    ):

        batch_size = momentum.shape[0]

        # ---------------------------------------------------------
        # Condition
        # ---------------------------------------------------------

        condition = self.condition_embedding(
            momentum,
            refractive_index
        )

        # [B, D] -> [B, 1, D]
        condition = condition.unsqueeze(1)

        # ---------------------------------------------------------
        # Expand learned query slots
        # ---------------------------------------------------------

        queries = self.query_slots.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )

        # Add physical condition to every hit slot
        x = queries + condition

        # ---------------------------------------------------------
        # Transformer + MoE
        # ---------------------------------------------------------

        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attention_mask
            )

        x = self.final_norm(x)

        # ---------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------

        occupancy_logits = self.occupancy_head(x)

        x_logits = self.x_head(x)

        y_logits = self.y_head(x)

        energy_logits = self.energy_head(x)

        return {
            "occupancy_logits": occupancy_logits,
            "x_logits": x_logits,
            "y_logits": y_logits,
            "energy_logits": energy_logits,
        }


if __name__ == "__main__":

    print("=" * 60)
    print("Testing CherenkovGenerator")
    print("=" * 60)

    config = GeneratorConfig()

    model = CherenkovGenerator(config)

    print("\nModel created successfully.")

    # Test conditions
    batch_size = 4

    momentum = torch.tensor([
        0.5,
        2.0,
        5.0,
        20.0,
    ])

    refractive_index = torch.tensor([
        1.018,
        1.020,
        1.024,
        1.026,
    ])

    # Forward pass
    output = model(
        momentum=momentum,
        refractive_index=refractive_index,
    )

    print("\nInput:")
    print("  momentum:", momentum.shape)
    print("  refractive_index:", refractive_index.shape)

    print("\nOutput:")
    for name, tensor in output.items():
        print(
            f"  {name:20s}: {tuple(tensor.shape)}"
        )

    # Parameter counts
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_parameters = sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad
    )

    print("\nParameters:")
    print(f"  Total:      {total_parameters:,}")
    print(f"  Trainable:  {trainable_parameters:,}")
    print(f"  Frozen:     {frozen_parameters:,}")

    print("\nReserved experts:")

    for layer_index, layer in enumerate(model.layers):

        moe = layer.moe

        expert_2_frozen = not any(
            p.requires_grad
            for p in moe.reserved_expert_2.parameters()
        )

        expert_3_frozen = not any(
            p.requires_grad
            for p in moe.reserved_expert_3.parameters()
        )

        print(
            f"  Layer {layer_index}: "
            f"Expert 2 frozen = {expert_2_frozen}"
        )

        print(
            f"  Layer {layer_index}: "
            f"Expert 3 frozen = {expert_3_frozen}"
        )

    print("\n✓ Model test completed.")