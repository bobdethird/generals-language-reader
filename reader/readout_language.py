"""Decision-conditioned language from the actor's own frozen internal readout.

Unlike a free activation probe, this variant deliberately uses a copied original
action head. It tests faithful verbalization, not independent recovery of intent.
No game-player weights are trained and no language output drives the player.
"""
import torch
from torch import nn
from torch.nn import functional as F
from reader.spatial import SpatialAdapter
from reader.language import LanguageReader, ActivationAdapter
from reader.decision_language import decision_token_weights


class ReadoutSpatialAdapter(SpatialAdapter):
    def __init__(self, base, activation_dim, language_dim, width=256):
        super().__init__(base, activation_dim, language_dim, width)
        self.activation_dim = activation_dim
        self.mask_embedding = nn.Linear(36, width, bias=False)
        nn.init.zeros_(self.mask_embedding.weight)
        self.register_buffer("head_weight", torch.zeros(81, activation_dim))
        self.register_buffer("head_bias", torch.zeros(81))
        self.row_words = nn.Embedding(25, language_dim)
        self.col_words = nn.Embedding(25, language_dim)
        self.direction_words = nn.Embedding(5, language_dim)
        self.amount_words = nn.Embedding(3, language_dim)
        self.selected_context = nn.Linear(width, language_dim)
        self.preference = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, language_dim))
        for embedding in (self.row_words, self.col_words, self.direction_words, self.amount_words):
            nn.init.normal_(embedding.weight, std=.02)

    def base_activations(self, activations):
        return activations[..., :self.activation_dim]

    def contextualize(self, activations):
        if activations.shape[1:] != (67, self.activation_dim + 36):
            raise ValueError("Readout reader requires hidden activations and legal-move flags")
        hidden = self.input(self.base_activations(activations))
        return self.contextualize_embeddings(hidden + self.mask_embedding(activations[..., self.activation_dim:]))

    @torch.no_grad()
    def read_logits(self, activations):
        # Match the original B200 JAX computation: FP32 accumulation inside the
        # BF16 dot, followed by a separately rounded BF16 bias addition. The mask
        # is applied in float32. Parity is still measured, never assumed.
        patches = self.base_activations(activations)[:, 3:].bfloat16()
        scores = ((patches @ self.head_weight.bfloat16().T) + self.head_bias.bfloat16()).float()
        flags = activations[:, 3:, self.activation_dim:].reshape(-1, 64, 9, 4).bool()
        flags = flags.permute(0, 1, 3, 2)
        flags = torch.cat([flags, flags, torch.ones_like(flags[:, :, :1])], 2).flatten(2)
        scores = scores.masked_fill(~flags, -1e9)
        return scores.reshape(-1, 8, 8, 9, 3, 3).permute(0, 3, 1, 4, 2, 5).reshape(-1, 5184)

    def forward_with_aux(self, activations):
        mapped, aux = super().forward_with_aux(activations)
        logits = self.read_logits(activations)
        action = logits.argmax(-1).clamp_max(4608)
        passing = action == 4608
        channel, cell = action // 576, action % 576
        row, col = cell // 24, cell % 24
        direction, amount = channel % 4, channel // 4
        patch = row // 3 * 8 + col // 3
        context = aux["context"][torch.arange(len(mapped), device=mapped.device), patch + 3]
        context = torch.where(passing[:, None], aux["context"][:, 0], context)
        top = logits.topk(2, -1).values
        probs = logits.softmax(-1)
        features = torch.stack([(top[:, 0]-top[:, 1]).clamp(max=10)/10,
                                probs.max(-1).values,
                                -(probs * logits.log_softmax(-1)).sum(-1)/9], -1)
        decision_tokens = torch.stack([
            self.row_words(row.masked_fill(passing, 24)),
            self.col_words(col.masked_fill(passing, 24)),
            self.direction_words(direction.masked_fill(passing, 4)),
            self.amount_words(amount.masked_fill(passing, 2)),
            .1*self.selected_context(context), .1*self.preference(features),
        ], 1)
        aux["readout_policy"] = logits
        return torch.cat([mapped, decision_tokens], 1), aux


class ReadoutLanguageReader(LanguageReader):
    def __init__(self, model_path, activation_dim, device, prompt):
        super().__init__(model_path, activation_dim, device, prompt,
                         adapter_type="spatial", precision="bfloat16", attention="sdpa")
        self.legal_mask = True
        self.policy_aux = False
        if self.device.type == "cuda":
            # Reduced-precision intermediate reductions changed near-tied actions
            # in the original diagnostic. This flag affects this reader process.
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.adapter = ReadoutSpatialAdapter(ActivationAdapter(activation_dim, self.model.config.hidden_size),
                                             activation_dim, self.model.config.hidden_size).to(self.device)

    def decision_loss(self, hidden, captions, teacher_logits, *, spatial, balance):
        # Teacher logits are targets/verification only. Readout parameters are buffers.
        return self.training_loss(hidden, captions, spatial_targets=spatial, balance_targets=balance,
                                  aux_weight=.1, token_weight_fn=decision_token_weights)

    @torch.no_grad()
    def audit_readout(self, hidden, target_logits, batch_size=128):
        exact = total = 0
        max_error = 0.
        for start in range(0, len(hidden), batch_size):
            x = torch.as_tensor(hidden[start:start+batch_size], dtype=torch.float32, device=self.device)
            target = torch.as_tensor(target_logits[start:start+batch_size], device=self.device)
            actual = self.adapter.read_logits(x)
            exact += int((actual.argmax(-1).clamp_max(4608) == target.argmax(-1).clamp_max(4608)).sum())
            total += len(x)
            max_error = max(max_error, float((actual-target).abs().max()))
        return {"positions": total, "greedy_agreement": exact/total, "max_logit_error": max_error,
                "note": "Measures the copied frozen action head separately from language accuracy."}
