"""A pretrained text decoder reading detached activations as prefix embeddings."""
import torch
from torch import nn


class ActivationAdapter(nn.Module):
    def __init__(self, activation_dim, language_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(activation_dim),
                                    nn.Linear(activation_dim, 256), nn.GELU(),
                                    nn.Linear(256, language_dim))

    def forward(self, activations):
        return self.layers(activations) * 0.1


PROMPT = (
    "Describe the generals.io game facts encoded in the preceding activation tokens. "
    "Use four short sentences about army balance, enemy troop visibility, "
    "whether the enemy general has been located, and territory balance."
)


class LanguageReader:
    def __init__(self, model_path, activation_dim, device="cpu", prompt=PROMPT, *,
                 adapter_type="tokenwise", precision="float32", attention="eager"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[precision]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype,
            attn_implementation=attention).to(self.device).eval()
        self.model.requires_grad_(False)
        self.adapter = ActivationAdapter(activation_dim, self.model.config.hidden_size).to(self.device)
        self.adapter_type = adapter_type
        if adapter_type == "spatial":
            from reader.spatial import SpatialAdapter
            self.adapter = SpatialAdapter(self.adapter, activation_dim, self.model.config.hidden_size).to(self.device)
        elif adapter_type != "tokenwise":
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

    def prefix(self, activations, return_aux=False):
        hidden = torch.as_tensor(activations, dtype=torch.float32, device=self.device)
        aux = None
        if return_aux and hasattr(self.adapter, "forward_with_aux"):
            mapped, aux = self.adapter.forward_with_aux(hidden.detach())
        else:
            mapped = self.adapter(hidden.detach())
        prompt = self.model.get_input_embeddings()(self.prompt_ids).expand(len(hidden), -1, -1)
        prefix = torch.cat([mapped.to(prompt.dtype), prompt], dim=1)
        return (prefix, aux) if return_aux else prefix

    def training_loss(self, activations, captions, *, value_weight=5.0,
                      spatial_targets=None, balance_targets=None, aux_weight=.1,
                      token_weight_fn=None):
        from torch.nn import functional as F
        from reader.spatial import factual_token_weights
        prefix, aux = self.prefix(activations, return_aux=True)
        targets = self.tokenizer([x + self.tokenizer.eos_token for x in captions],
                                 add_special_tokens=False, padding=True, return_offsets_mapping=True,
                                 return_tensors="pt")
        offsets = targets.pop("offset_mapping")
        weight_fn = token_weight_fn or factual_token_weights
        weights = weight_fn(captions, offsets, targets.attention_mask, value_weight).to(self.device)
        targets = targets.to(self.device)
        embeddings = self.model.get_input_embeddings()(targets.input_ids)
        mask = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device),
                          targets.attention_mask], dim=1)
        # Only project the last prefix token and the target sequence to vocabulary
        # logits. The full prefix still participates in every attention layer.
        logits = self.model(inputs_embeds=torch.cat([prefix, embeddings], 1), attention_mask=mask,
                            use_cache=False, logits_to_keep=targets.input_ids.shape[1] + 1).logits[:, :-1]
        ce = F.cross_entropy(logits.float().transpose(1, 2), targets.input_ids, reduction="none")
        weighted = (ce * weights).sum() / weights.sum()
        nll = (ce * targets.attention_mask).sum() / targets.attention_mask.sum()
        auxiliary = torch.zeros((), device=self.device)
        if aux is not None and spatial_targets is not None:
            spatial_targets = torch.as_tensor(spatial_targets, dtype=torch.float32, device=self.device)
            counts = spatial_targets.sum(-1)
            target_distribution = spatial_targets / counts.clamp_min(1)[..., None]
            per_field = -(target_distribution * F.log_softmax(aux["locations"].float(), -1)).sum(-1)
            location_loss = per_field[counts > 0].mean()
            balance_loss = F.cross_entropy(aux["balance"].float(),
                                          torch.as_tensor(balance_targets, device=self.device))
            auxiliary = location_loss + balance_loss
        loss = weighted + aux_weight * auxiliary
        return loss, {"caption_nll": float(nll.detach()), "weighted_nll": float(weighted.detach()),
                      "auxiliary_loss": float(auxiliary.detach())}

    def loss(self, activations, captions):
        prefix = self.prefix(activations)
        targets = self.tokenizer([x + self.tokenizer.eos_token for x in captions],
                                 add_special_tokens=False, padding=True,
                                 return_tensors="pt").to(self.device)
        embeddings = self.model.get_input_embeddings()(targets.input_ids)
        labels = targets.input_ids.masked_fill(targets.attention_mask == 0, -100)
        ignored = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=self.device)
        labels = torch.cat([ignored, labels], dim=1)
        mask = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device),
                          targets.attention_mask], dim=1)
        return self.model(inputs_embeds=torch.cat([prefix, embeddings], dim=1),
                          attention_mask=mask, labels=labels, use_cache=False).loss

    @torch.no_grad()
    def generate(self, activations, max_new_tokens=80):
        prefix = self.prefix(activations)
        tokens = self.model.generate(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
