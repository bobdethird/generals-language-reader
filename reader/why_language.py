"""A frozen text decoder translates measured policy sensitivity into language.

Only its continuous-prefix adapter learns. Inputs contain player activations and
raw numerical intervention effects; no rationale labels or words are inputs.
At deployment the interventions run afresh on the frozen policy.
"""
import numpy as np
import torch
from torch import nn
from reader.language import LanguageReader

WHY_PROMPT = (
    "Explain which tested input change influences the frozen player's preferred action. "
    "The continuous tokens encode ten controlled probes and their measured effects at two strengths. "
    "Report the strongest consistent influence and whether the stronger test changes the preferred action. "
    "If none is consistent, say the tests do not identify a consistent influence. "
    "Use two short factual sentences. Do not invent a strategic intention or future outcome."
)


def pack_why_inputs(hidden, evidence):
    hidden, evidence = np.asarray(hidden), np.asarray(evidence)
    if hidden.shape[1:] != (67, 448) or evidence.shape != (len(hidden), 10, 8):
        raise ValueError("Unexpected hidden/evidence dimensions")
    packed = np.zeros((len(hidden), 77, 448), np.float32)
    packed[:, :67] = hidden
    packed[:, 67:, :8] = evidence
    return packed


class WhyAdapter(nn.Module):
    def __init__(self, language_dim):
        super().__init__()
        self.probe = nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, language_dim))
        self.position = nn.Parameter(torch.randn(10, language_dim)*.02)
        self.context = nn.Sequential(nn.LayerNorm(448), nn.Linear(448, 128), nn.GELU(), nn.Linear(128, language_dim))

    def forward(self, packed):
        if packed.shape[1:] != (77, 448):
            raise ValueError("Why decoder requires hidden state and measured probe effects")
        context = self.context(packed[:, :3])
        probes = self.probe(packed[:, 67:, :8]) + self.position
        return torch.cat([context, probes], 1)*.1


class StructuredWhyAdapter(nn.Module):
    """Learn to compare the ten measured effects before the frozen decoder.

Auxiliary predictions are continuous distributions at inference, never supplied
reference labels. This layer learns evidence selection from the automatic tests.
"""
    def __init__(self,language_dim,width=128):
        super().__init__()
        self.input=nn.Linear(8,width)
        self.position=nn.Parameter(torch.randn(10,width)*.02)
        self.queries=nn.Parameter(torch.randn(3,width)*.02)
        layer=nn.TransformerEncoderLayer(width,4,4*width,dropout=0.,activation="gelu",batch_first=True,norm_first=True)
        self.comparison=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.probe_head=nn.Linear(width,11)
        self.effect_head=nn.Linear(width,3)
        self.switch_head=nn.Linear(width,2)
        self.probe_words=nn.Parameter(torch.randn(11,language_dim)*.02)
        self.effect_words=nn.Parameter(torch.randn(3,language_dim)*.02)
        self.switch_words=nn.Parameter(torch.randn(2,language_dim)*.02)
        self.evidence_language=nn.Linear(width,language_dim)
        self.context=nn.Sequential(nn.LayerNorm(448),nn.Linear(448,128),nn.GELU(),nn.Linear(128,language_dim))

    def forward_with_aux(self,packed):
        x=self.input(packed[:,67:,:8])+self.position
        x=torch.cat([self.queries[None].expand(len(x),-1,-1),x],1)
        x=self.comparison(x)
        auxiliary={"probe":self.probe_head(x[:,0]),"effect":self.effect_head(x[:,1]),"switch":self.switch_head(x[:,2])}
        words=torch.stack([auxiliary["probe"].softmax(-1)@self.probe_words,
                           auxiliary["effect"].softmax(-1)@self.effect_words,
                           auxiliary["switch"].softmax(-1)@self.switch_words],1)
        mapped=torch.cat([self.context(packed[:,:3]),self.evidence_language(x[:,3:]),words],1)*.1
        return mapped,auxiliary

    def forward(self,packed):return self.forward_with_aux(packed)[0]


class PointerWhyAdapter(StructuredWhyAdapter):
    """Share a learned influence scorer across all ten probes, including rare ones."""
    def __init__(self,language_dim,width=128):
        super().__init__(language_dim,width)
        self.probe_head=nn.Linear(width,1)

    def forward_with_aux(self,packed):
        features=packed[:,67:,:8]
        x=self.input(features)+self.position
        x=self.comparison(torch.cat([self.queries[None].expand(len(x),-1,-1),x],1))
        eligible=features[:,:,5]>.5
        scores=self.probe_head(x[:,3:]).squeeze(-1).masked_fill(~eligible,-1e4)
        none=torch.where(eligible.any(-1),-1e4,0.).to(scores.dtype)
        probe=torch.cat([none[:,None],scores],-1)
        probabilities=probe.softmax(-1)
        selected=(probabilities[:,1:,None]*x[:,3:]).sum(1)+probabilities[:,:1]*x[:,0]
        auxiliary={"probe":probe,"effect":self.effect_head(selected),"switch":self.switch_head(selected)}
        words=torch.stack([probabilities@self.probe_words,
                           auxiliary["effect"].softmax(-1)@self.effect_words,
                           auxiliary["switch"].softmax(-1)@self.switch_words],1)
        mapped=torch.cat([self.context(packed[:,:3]),self.evidence_language(x[:,3:]),words],1)*.1
        return mapped,auxiliary


class WhyLanguageReader(LanguageReader):
    def __init__(self, model_path, device="cpu",architecture="tokenwise"):
        super().__init__(model_path, 448, device, WHY_PROMPT, precision="bfloat16", attention="sdpa")
        classes={"tokenwise":WhyAdapter,"structured":StructuredWhyAdapter,"pointer":PointerWhyAdapter}
        if architecture not in classes:raise ValueError("Unknown why architecture")
        cls=classes[architecture]
        self.adapter=cls(self.model.config.hidden_size).to(self.device)
