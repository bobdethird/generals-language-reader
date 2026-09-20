"""A continuous spatial residual adapter; diagnostic heads never generate words."""
import math
import re
import torch
from torch import nn


def coordinate_features(grid=8):
    row, col = torch.meshgrid(torch.linspace(-1, 1, grid), torch.linspace(-1, 1, grid), indexing="ij")
    features = [row, col]
    for frequency in (1, 2, 4, 8):
        for coordinate in (row, col):
            features.extend([torch.sin(math.pi * frequency * coordinate),
                             torch.cos(math.pi * frequency * coordinate)])
    return torch.stack(features, -1).reshape(grid * grid, -1)


class SpatialAdapter(nn.Module):
    """Preserve the old soft prefix and learn a spatially contextual residual.

    Inputs are the three global/history tokens followed by an 8x8 grid of 3x3
    board patches. The decoder sees continuous embeddings, never predicted
    labels. Auxiliary heads provide automatic spatial supervision during training.
    """
    def __init__(self, base, activation_dim, language_dim, width=256):
        super().__init__()
        self.base = base
        self.input = nn.Sequential(nn.LayerNorm(activation_dim), nn.Linear(activation_dim, width))
        self.register_buffer("coordinates", coordinate_features())
        self.position = nn.Linear(18, width, bias=False)
        self.global_types = nn.Parameter(torch.randn(3, width) * .01)
        self.local = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1))
        layer = nn.TransformerEncoderLayer(width, 8, 3 * width, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, language_dim)
        # Start with precisely the prior reader's continuous prefix.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.locations = nn.Linear(width, 3 * 9)
        self.balance = nn.Linear(width, 3)

    def contextualize(self, activations):
        if activations.ndim != 3 or activations.shape[1] != 67:
            raise ValueError("Spatial adapter requires 3 global tokens and 64 row-major patches")
        h = self.input(activations)
        return self.contextualize_embeddings(h)

    def base_activations(self, activations):
        return activations

    def contextualize_embeddings(self, h):
        patches = h[:, 3:] + self.position(self.coordinates)[None]
        grid = patches.transpose(1, 2).reshape(len(h), -1, 8, 8)
        patches = patches + self.local(grid).flatten(2).transpose(1, 2)
        h = torch.cat([h[:, :3] + self.global_types[None], patches], 1)
        return self.norm(self.context(h))

    def forward_with_aux(self, activations):
        h = self.contextualize(activations)
        mapped = self.base(self.base_activations(activations)) + .1 * self.output(h)
        # Cell ordering: patch-row, patch-col, local-row, local-col.
        locations = self.locations(h[:, 3:]).reshape(len(h), 64, 3, 9)
        aux = {"locations": locations.permute(0, 2, 1, 3).reshape(len(h), 3, 576),
               "balance": self.balance(h[:, :3].mean(1)), "context": h}
        return mapped, aux

    def forward(self, activations):
        return self.forward_with_aux(activations)[0]


FACT_VALUES = re.compile(
    r"\b(?:upper|middle|lower) (?:left|center|right)\b|"
    r"\b(?:Own|Enemy)(?= army is larger)|\bsimilar(?= in size)|"
    r"\bNo(?= enemy troops are visible)")


def factual_token_weights(captions, offsets, attention_mask, value_weight):
    """Weight factual words, including tokens that overlap a multiword value."""
    weights = attention_mask.to(dtype=torch.float32).clone()
    for row, caption in enumerate(captions):
        spans = [match.span() for match in FACT_VALUES.finditer(caption)]
        for col, (start, end) in enumerate(offsets[row].tolist()):
            if end > start and any(start < right and end > left for left, right in spans):
                weights[row, col] *= value_weight
    return weights
