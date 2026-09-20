"""Restore a trained decision reader without changing its inference contract."""
from huggingface_hub import snapshot_download
from reader.decision_language import DecisionLanguageReader


def restore_reader(checkpoint, device):
    variant = checkpoint.get("variant")
    if variant not in ("spatial", "policy_aux", "policy_mask", "readout"):
        raise ValueError("Use a decision-reader checkpoint, not a board-grounding reader")
    lock = checkpoint["model"]
    path = snapshot_download(lock["model"], revision=lock["revision"], local_files_only=True)
    if variant == "readout":
        from reader.readout_language import ReadoutLanguageReader
        reader = ReadoutLanguageReader(path, checkpoint["activation_dim"], device, checkpoint["prompt"])
    else:
        reader = DecisionLanguageReader(path, checkpoint["activation_dim"], device, checkpoint["prompt"],
                                       policy_aux=variant != "spatial", legal_mask=variant == "policy_mask")
    reader.adapter.load_state_dict(checkpoint["adapter"])
    reader.adapter.eval().requires_grad_(False)
    return reader
