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
    def __init__(self, model_path, activation_dim, device="cpu", prompt=PROMPT):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager").to(self.device).eval()
        self.model.requires_grad_(False)
        self.adapter = ActivationAdapter(activation_dim, self.model.config.hidden_size).to(self.device)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

    def prefix(self, activations):
        hidden = torch.as_tensor(activations, dtype=torch.float32, device=self.device)
        mapped = self.adapter(hidden.detach())
        prompt = self.model.get_input_embeddings()(self.prompt_ids).expand(len(hidden), -1, -1)
        return torch.cat([mapped, prompt], dim=1)

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
