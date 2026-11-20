"""The plain decoder: a phoneme recogniser with no language model, restricted to English sounds.

facebook/wav2vec2-lv-60-espeak-cv-ft is wav2vec 2.0 large, pretrained on 60k hours of LibriLight and fine-tuned on
Common Voice to write espeak phonemes (CTC). We take its single best sound per 20 ms frame, after renormalising its
odds over the American-English sounds only. Nothing guesses words, so nothing fills in what didn't arrive.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForCTC

from .sounds import EN, respell

MODEL = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
HOP_S = 0.02                     # one decoder frame
GAP_FRAMES = 4                   # blank frames in a row that end a written chunk (80 ms)


@dataclass
class Decode:
    ids: np.ndarray                         # best English sound per frame (vocab ids; pad = blank)
    speech: np.ndarray                      # frames whose best sound isn't the blank
    entropy_en: np.ndarray                  # per frame, bits, over English sounds
    heard: str                              # the sounds in order, repeats collapsed (IPA)
    tokens: list = field(default_factory=list)      # the same as a list of tokens
    timeline: list = field(default_factory=list)    # [{'t': s, 's': respelled, 'gap': bool}] for playback

    @property
    def blur(self):
        """Mean per-frame entropy (bits) over the decoder's own speech frames: how unsure it was."""
        return float(self.entropy_en[self.speech].mean()) if self.speech.any() else None


class Decoder:
    def __init__(self, model=MODEL, threads=10):
        path = Path(model)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(model))
        torch.set_num_threads(threads)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).eval()
        vocab = json.loads((path / 'vocab.json').read_text(encoding='utf-8'))
        self.inv = {i: t for t, i in vocab.items()}
        self.pad = vocab.get('<pad>', 0)
        self.en_ids = np.array(sorted({vocab[t] for t in EN} | {self.pad}))

    @staticmethod
    def to16k(x, sr):
        """float32 audio at 48 or 16 kHz -> 16 kHz."""
        return x.astype(np.float32) if sr == 16000 else resample_poly(x, 1, sr // 16000).astype(np.float32)

    def logprobs(self, x16):
        """Per frame: log-odds over every sound, and over the English sounds renormalised."""
        z = (x16 - x16.mean()) / (x16.std() + 1e-7)      # the model's own input scaling: loudness never matters
        with torch.no_grad():
            lp = torch.log_softmax(self.model(torch.from_numpy(z)[None]).logits[0], -1).numpy()
        le = lp[:, self.en_ids] - np.logaddexp.reduce(lp[:, self.en_ids], axis=1, keepdims=True)
        return lp, le

    def decode(self, x16, lp_le=None):
        lp, le = lp_le if lp_le is not None else self.logprobs(x16)
        ids = self.en_ids[le.argmax(-1)]
        speech = ids != self.pad
        ent = -(np.exp(le) * le).sum(-1) / np.log(2)
        tokens = [self.inv[i] for j, i in enumerate(ids) if i != self.pad and (j == 0 or i != ids[j - 1])]
        hop = len(x16) / 16000 / max(1, len(ids))
        timeline, prev, run = [], None, 0
        for f, i in enumerate(ids):
            if i == self.pad:
                run += 1; prev = None
                continue
            if i != prev:
                timeline.append({'t': round(f * hop, 3), 's': respell(self.inv[i]), 'gap': run >= GAP_FRAMES and bool(timeline)})
            run = 0; prev = i
        return Decode(ids=ids, speech=speech, entropy_en=ent, heard=''.join(tokens), tokens=tokens, timeline=timeline)


def written(timeline):
    """A playback timeline as one plain-letter line, chunks split where the decoder heard a pause."""
    return ''.join((' ' if e['gap'] else '') + e['s'] for e in timeline)
