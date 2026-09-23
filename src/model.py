"""Multimodal Laya: frozen SigLIP + Laya's text encoder + a trained bridge.

    image ──► SigLIP (frozen, cached) ──► [P, d_v] ──► projector ──► [P, d]
                                                                       │
                                                                 cross-attention
                                                                       │
    text  ──► ModernBERT ──► [CLS] q [SEP] [MASK]o0 [MASK]o1 [SEP] ────┤
                                              │                        │
                                        gather at markers ◄────────────┘
                                              │
                                     scorer: Linear(d, 1)

Design decisions, and why:

* **The text side is initialised from the Laya checkpoint**, not from raw
  ModernBERT. The scorer, type_emb and marker behaviour are already trained —
  26.5M parameters of decision machinery that cost a full RLCD run. Reusing
  them means visual grounding is the *only* new thing in the model, so a
  failure points at the bridge rather than at an undertrained head.

* **Both towers frozen by default.** Trainable budget is the projector (~4M)
  plus cross-attention (~25M). Once image features are cached the vision tower
  is not even resident in VRAM.

* **The scorer is untouched.** One vector in, one number out, applied per
  marker. K never enters a weight matrix, so the label set stays open-ended
  exactly as in text Laya.
"""
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Question/option tokens read from the already-encoded image."""

    def __init__(self, d, nhead, dropout=0.1):
        super().__init__()
        self.nq, self.nkv, self.nff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv, kv_pad=None):
        kvn = self.nkv(kv)
        q = q + self.attn(self.nq(q), kvn, kvn, key_padding_mask=kv_pad,
                          need_weights=False)[0]
        return q + self.ff(self.nff(q))


class Projector(nn.Module):
    """SigLIP patch dim -> text hidden dim. Two layers; LLaVA found this beats linear."""

    def __init__(self, d_v, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_v), nn.Linear(d_v, d),
                                 nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class MultimodalDecisionModel(nn.Module):
    def __init__(self, laya_model, d_v=1152, cross_layers=2,
                 freeze_text=True, dropout=0.1):
        """`laya_model` is a built laya.common.DecisionModel with weights loaded."""
        super().__init__()
        d = laya_model.encoder.config.hidden_size
        self.d, self.d_v = d, d_v

        # --- inherited from Laya, unchanged ---
        self.encoder = laya_model.encoder
        self.head = laya_model.head
        self.type_emb = laya_model.type_emb
        self.scorer = laya_model.scorer

        # --- the only new weights ---
        self.projector = Projector(d_v, d)
        self.cross = nn.ModuleList(
            [CrossAttentionBlock(d, max(1, d // 64), dropout) for _ in range(cross_layers)])

        if freeze_text:
            for p in self.encoder.parameters():
                p.requires_grad = False
        try:
            self.encoder.config.reference_compile = False   # hurts at our batch sizes
        except Exception:
            pass

    # ------------------------------------------------------------------
    def trainable_report(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fr = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print("trainable %.1fM | frozen %.1fM | total %.1fM"
              % (tr / 1e6, fr / 1e6, (tr + fr) / 1e6))
        for n, m in (("projector", self.projector), ("cross", self.cross),
                     ("head", self.head), ("scorer", self.scorer)):
            if m is not None:
                c = sum(p.numel() for p in m.parameters())
                g = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print("   %-10s %6.1fM  (%.1fM trainable)" % (n, c / 1e6, g / 1e6))
        return tr

    def unfreeze_text(self, last_n=0):
        """Stage 2: let the text encoder adapt. last_n=0 unfreezes everything."""
        layers = getattr(self.encoder, "layers", None) or \
                 getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if last_n and layers is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for l in layers[-last_n:]:
                for p in l.parameters():
                    p.requires_grad = True
            print("unfroze the last %d encoder layers" % last_n)
        else:
            for p in self.encoder.parameters():
                p.requires_grad = True
            print("unfroze the whole text encoder")

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask,
                qtype, image_feats, image_mask=None):
        """image_feats: [B, P, d_v] cached SigLIP patches (or [1, P, d_v] shared)."""
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]

        v = self.projector(image_feats)                    # [B, P, d]
        if v.size(0) == 1 and h.size(0) > 1:               # one image, many questions
            v = v.expand(h.size(0), -1, -1)                # a view, not a copy
        v_pad = None if image_mask is None else ~image_mask.bool()
        for blk in self.cross:
            h = blk(h, v, v_pad)

        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


# ---------------------------------------------------------------------- build
def load_laya_backbone(repo="convaiinnovations/laya", subfolder=None, token=None):
    """Fetch a Laya checkpoint and return (DecisionModel, tokenizer, cfg)."""
    import json, os
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model

    d = snapshot_download(repo, token=token,
                          allow_patterns=[f"{subfolder}/*"] if subfolder else None)
    if subfolder:
        d = os.path.join(d, subfolder)
    _fix_tokenizer_config(d)
    cfg = json.load(open(os.path.join(d, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(d, "encoder"))
    model.load_state_dict(load_file(os.path.join(d, "model.safetensors")), strict=True)
    print("loaded %s — d=%d, max_len=%d, head_max_len=%d"
          % (repo, model.encoder.config.hidden_size, cfg.get("max_len"),
             cfg.get("head_max_len")))
    return model, tok, cfg


def build(repo="convaiinnovations/laya", d_v=1152, cross_layers=2,
          freeze_text=True, token=None):
    laya_model, tok, cfg = load_laya_backbone(repo, token=token)
    mm = MultimodalDecisionModel(laya_model, d_v=d_v, cross_layers=cross_layers,
                                 freeze_text=freeze_text)
    mm.trainable_report()
    return mm, tok, cfg
