import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM
from reader.language import LanguageReader, ActivationAdapter


class TinyTokenizer:
    eos_token = "."

    def __call__(self, texts, return_offsets_mapping=False, **kwargs):
        width = max(map(len, texts))
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        offsets = torch.zeros(len(texts), width, 2, dtype=torch.long)
        for row, text in enumerate(texts):
            ids[row, :len(text)] = torch.tensor([ord(c) % 50 + 3 for c in text])
            mask[row, :len(text)] = 1
            offsets[row, :len(text)] = torch.tensor([[i, i + 1] for i in range(len(text))])
        data = {"input_ids": ids, "attention_mask": mask}
        if return_offsets_mapping:
            data["offset_mapping"] = offsets
        return BatchEncoding(data)


def test_cropped_language_loss_matches_full_teacher_forcing_and_keeps_backbone_frozen():
    torch.manual_seed(5)
    reader = LanguageReader.__new__(LanguageReader)
    reader.device = torch.device("cpu")
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8)
    reader.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    reader.tokenizer = TinyTokenizer()
    reader.adapter = ActivationAdapter(8, 32)
    reader.prompt_ids = torch.tensor([[1, 2]])
    hidden = torch.randn(2, 67, 8, requires_grad=True)
    captions = ["upper left", "middle right"]
    regular = reader.loss(hidden, captions)
    cropped, stats = reader.training_loss(hidden, captions, value_weight=1)
    torch.testing.assert_close(cropped, regular, atol=1e-6, rtol=1e-6)
    cropped.backward()
    assert reader.adapter.layers[1].weight.grad.abs().sum() > 0
    assert hidden.grad is None
    assert all(p.grad is None for p in reader.model.parameters())
