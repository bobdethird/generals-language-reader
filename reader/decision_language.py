"""Decision-focused language supervision with an optional continuous policy head."""
import re
import torch
from torch import nn
from torch.nn import functional as F
from reader.language import LanguageReader, ActivationAdapter
from reader.spatial import SpatialAdapter


class DecisionSpatialAdapter(SpatialAdapter):
    """Learn action structure as an auxiliary task; no teacher outputs are inputs."""
    def __init__(self, base, activation_dim, language_dim, width=256):
        super().__init__(base, activation_dim, language_dim, width)
        self.policy = nn.Linear(width, 81)
        self.policy_to_language = nn.Linear(81, language_dim)
        nn.init.zeros_(self.policy_to_language.weight)
        nn.init.zeros_(self.policy_to_language.bias)

    def forward_with_aux(self, activations):
        mapped, auxiliary = super().forward_with_aux(activations)
        patch_policy = self.policy(auxiliary["context"][:, 3:])
        contribution = .1 * self.policy_to_language(self.policy_features(patch_policy, activations))
        mapped = torch.cat([mapped[:, :3], mapped[:, 3:] + contribution], 1)
        logits = patch_policy.reshape(len(mapped), 8, 8, 9, 3, 3)
        auxiliary["policy"] = logits.permute(0, 3, 1, 4, 2, 5).reshape(len(mapped), -1)
        return mapped, auxiliary

    def policy_features(self, patch_policy, activations):
        return F.layer_norm(patch_policy, (81,))


class MaskedDecisionSpatialAdapter(DecisionSpatialAdapter):
    """Read the same legal-action constraint used by the frozen game player."""
    def __init__(self, base, activation_dim, language_dim, width=256):
        super().__init__(base, activation_dim, language_dim, width)
        self.activation_dim = activation_dim
        self.mask_embedding = nn.Linear(36, width, bias=False)

    def base_activations(self, activations):
        return activations[..., :self.activation_dim]

    def contextualize(self, activations):
        if activations.shape[1:] != (67, self.activation_dim + 36):
            raise ValueError("Expected activations and 36 per-patch legal direction flags")
        hidden = self.input(self.base_activations(activations))
        hidden = hidden + self.mask_embedding(activations[..., self.activation_dim:])
        return self.contextualize_embeddings(hidden)

    def policy_features(self, patch_policy, activations):
        mask = activations[:, 3:, self.activation_dim:].reshape(len(activations), 64, 9, 4).bool()
        direction_mask = mask.permute(0, 1, 3, 2)
        full_mask = torch.cat([direction_mask, direction_mask, torch.ones_like(direction_mask[:, :, :1])], 2)
        masked = patch_policy.masked_fill(~full_mask.flatten(2), -1e9)
        # Across-patch normalization retains the complete raw action ordering,
        # including the original individual pass encodings.
        return (masked.flatten(1).log_softmax(-1).reshape_as(masked).clamp_min(-20) + 10) / 10


VALUES = re.compile(r"\b\d+\b|\b(?:north|south|east|west|half|all but one|wait|friendly|enemy|neutral|unknown|closer|farther|same)\b", re.I)


def decision_token_weights(captions, offsets, attention_mask, value_weight):
    weights = attention_mask.float().clone()
    for row, caption in enumerate(captions):
        spans = [m.span() for m in VALUES.finditer(caption)]
        for col, (left, right) in enumerate(offsets[row].tolist()):
            if right > left and any(left < end and right > start for start, end in spans):
                weights[row, col] *= value_weight
    return weights


def policy_distillation(student, teacher):
    """Preserve sampled probabilities AND emphasize exact deployment decisions."""
    teacher = teacher.detach().float()
    student = student.float().masked_fill(teacher < -1e8, -1e9)
    p = teacher.softmax(-1)
    full_kl = F.kl_div(student.log_softmax(-1), p, reduction="batchmean")
    move_count = teacher.shape[-1] * 8 // 9
    moving = (teacher[:, :move_count] > -1e8).any(-1)
    move_kl = student.sum() * 0
    if moving.any():
        move_kl = F.kl_div(student[moving, :move_count].log_softmax(-1),
                           teacher[moving, :move_count].softmax(-1), reduction="batchmean")
    preferred = teacher.argmax(-1)
    teacher_top = teacher.topk(2, dim=-1).values
    margin = (teacher_top[:, 0] - teacher_top[:, 1]).clamp(max=1)
    preferred_student = student.gather(1, preferred[:, None]).squeeze(1)
    alternatives = student.scatter(1, preferred[:, None], -1e9).max(-1).values
    # Unlike hard-label cross entropy, this term is zero at the true policy;
    # it does not reward inventing extra confidence beyond the teacher's margin.
    rank_loss = F.relu(alternatives - preferred_student + margin).mean()
    return full_kl + .5 * move_kl + .5 * rank_loss, {
        "policy_full_kl": float(full_kl.detach()), "policy_move_kl": float(move_kl.detach()),
        "policy_rank_loss": float(rank_loss.detach())}


class DecisionLanguageReader(LanguageReader):
    def __init__(self, model_path, activation_dim, device, prompt, *, policy_aux=False, legal_mask=False):
        super().__init__(model_path, activation_dim, device, prompt,
                         adapter_type="spatial", precision="bfloat16", attention="sdpa")
        self.policy_aux = policy_aux or legal_mask
        self.legal_mask = legal_mask
        if self.policy_aux:
            base = ActivationAdapter(activation_dim, self.model.config.hidden_size)
            cls = MaskedDecisionSpatialAdapter if legal_mask else DecisionSpatialAdapter
            self.adapter = cls(base, activation_dim, self.model.config.hidden_size).to(self.device)

    def decision_loss(self, hidden, captions, teacher_logits, *, spatial, balance):
        loss, stats = self.training_loss(hidden, captions, spatial_targets=spatial,
                                        balance_targets=balance, aux_weight=.1,
                                        token_weight_fn=decision_token_weights)
        if self.policy_aux:
            _, auxiliary = self.adapter.forward_with_aux(
                torch.as_tensor(hidden, dtype=torch.float32, device=self.device).detach())
            policy_loss, policy_stats = policy_distillation(
                auxiliary["policy"], torch.as_tensor(teacher_logits, device=self.device))
            loss = loss + .2 * policy_loss
            stats.update(policy_stats)
        return loss, stats
