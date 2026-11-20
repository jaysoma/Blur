#!/usr/bin/env python3
"""Step 1: decode every CMU take of one sentence with the plain decoder. One JSON line per take, written as it goes;
a rerun skips takes already written.

  heard_en   the sounds it heard, English only (IPA)          blur_en  its mean per-frame doubt, bits, English only
  blur       the same over all ~390 sounds (comparison only)  foreign  share of its odds on sounds English doesn't use
  hilo_db    energy 4-8 kHz over 0.25-1 kHz on its speech frames (the classic facing cue, for comparison)
  level_db   the take's loudness

usage: python scripts/01_decode.py subjectrecording.zip results/fox_all.jsonl [--utt recording1] [--channel 1] [--limit N]
(about 30 min for the 5,760 fox takes on a 12-thread CPU)
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur import dov                                     # noqa: E402
from blur.decoder import Decoder, MODEL                  # noqa: E402

HOP48 = 960
HI, LO = (4000, 8000), (250, 1000)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('out')
    ap.add_argument('--utt', default='recording1'); ap.add_argument('--channel', type=int, default=1)
    ap.add_argument('--model', default=MODEL); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dec = Decoder(a.model)
    z = zipfile.ZipFile(a.zip)
    todo = dov.takes(z, a.utt, a.channel)
    out = Path(a.out)
    done = {json.loads(l)['clip'] for l in out.open(encoding='utf-8')} if out.exists() else set()
    todo = [t for t in todo if t.name not in done][:a.limit or None]
    fr = np.fft.rfftfreq(HOP48, 1 / 48000)
    mh, ml = (fr >= HI[0]) & (fr < HI[1]), (fr >= LO[0]) & (fr < LO[1])
    win = np.hanning(HOP48)
    t0 = time.perf_counter()
    with out.open('a', encoding='utf-8') as fh:
        for t in tqdm(todo, desc='decoding takes', unit='take', file=sys.stdout):
            x = dov.read(z, t.name)
            x16 = dec.to16k(x, 48000)
            lp, le = dec.logprobs(x16)
            d = dec.decode(x16, (lp, le))
            p = np.exp(lp)
            ids_all = lp.argmax(-1); sp_all = ids_all != dec.pad
            ent_all = -(p * lp).sum(-1) / np.log(2)
            foreign = 1 - p[:, dec.en_ids].sum(-1)
            hi = lo = 0.0
            for j in np.flatnonzero(d.speech):
                seg = x[j * HOP48:(j + 1) * HOP48]
                if len(seg) < HOP48: continue
                P = np.abs(np.fft.rfft(seg * win)) ** 2
                hi += P[mh].sum(); lo += P[ml].sum()
            rec = {'clip': t.name, 'person': t.person, 'room': t.room, 'place': t.place, 'trial': t.trial, 'spot': t.spot,
                   'dist_m': t.dist_m, 'polar_deg': t.polar_deg, 'utt': t.utt, 'dov_deg': t.dov_deg,
                   'blur_en': d.blur, 'blur': float(ent_all[sp_all].mean()) if sp_all.any() else None,
                   'foreign': float(foreign[d.speech].mean()) if d.speech.any() else None,
                   'hilo_db': float(10 * np.log10(hi / lo)) if hi > 0 and lo > 0 else None,
                   'level_db': float(10 * np.log10(np.mean(x ** 2) + 1e-12)),
                   'speech_frames_en': int(d.speech.sum()), 'heard_en': d.heard}
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush()
    print('done: %d takes in %.0f s' % (len(todo), time.perf_counter() - t0))


if __name__ == '__main__':
    main()
