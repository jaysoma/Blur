#!/usr/bin/env python3
"""Step 3: is the consonant loss the room's noise, or the turn? Re-decode one distance from cleaner audio, two ways:

  ch0   the ReSpeaker's own processed channel (its on-board speech processing)
  sum   its four raw mics (channels 1-4) added sample for sample

then compare, run by run, how often a turned take kept fewer consonants than its facing take, against step 1's one raw
mic. Decodes are written as they go (resumable); a rerun with everything written only prints the comparison.

usage: python scripts/03_cleanup_check.py subjectrecording.zip results/fox_all.jsonl results/fox_5m_clean.jsonl [--dist 5]
(about 20 min for 5 m)
"""
import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur import dov                                     # noqa: E402
from blur.decoder import Decoder, MODEL                  # noqa: E402
from blur.sentence import CONS_SLOTS, kept_slots         # noqa: E402
from blur.sounds import tokenize                          # noqa: E402


def counts(rows, dist):
    G = defaultdict(dict)
    for r in rows:
        if r['dist_m'] == dist:
            kept = kept_slots(tokenize(r['heard_en'] or ''))
            G[(r['person'], r['room'], r['place'], r['trial'], r['spot'])][r['dov_deg']] = sum(1 for k in CONS_SLOTS if k in kept)
    return {k: v for k, v in G.items() if len(v) == len(dov.ANGLES)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('raw'); ap.add_argument('out')
    ap.add_argument('--dist', type=int, default=5); ap.add_argument('--model', default=MODEL)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    takes = dov.takes(z, 'recording1', 1, a.dist)
    out = Path(a.out)
    done = {(r['take'], r['source']) for r in map(json.loads, out.open(encoding='utf-8'))} if out.exists() else set()
    todo = [(t, s) for t in takes for s in ('ch0', 'sum') if (t.key, s) not in done]
    if todo:
        dec = Decoder(a.model)
        with out.open('a', encoding='utf-8') as fh:
            for t, src in tqdm(todo, desc='decoding %d m (processed, summed)' % a.dist, unit='decode', file=sys.stdout):
                if src == 'ch0':
                    x, mics = dov.read(z, t.key + '_0.wav'), 1
                else:
                    chans = [dov.read(z, t.key + '_%d.wav' % c) for c in (1, 2, 3, 4) if t.key + '_%d.wav' % c in names]
                    L = min(len(c) for c in chans)
                    x, mics = sum(c[:L] for c in chans), len(chans)
                d = dec.decode(dec.to16k(x, 48000))
                fh.write(json.dumps({'take': t.key, 'source': src, 'person': t.person, 'room': t.room, 'place': t.place,
                                     'trial': t.trial, 'spot': t.spot, 'dist_m': t.dist_m, 'polar_deg': t.polar_deg,
                                     'dov_deg': t.dov_deg, 'blur_en': d.blur, 'heard_en': d.heard, 'mics': mics},
                                    ensure_ascii=False) + '\n'); fh.flush()
    raw = [json.loads(l) for l in open(a.raw, encoding='utf-8')]
    clean = [json.loads(l) for l in open(a.out, encoding='utf-8')]
    S = {'raw mic': counts(raw, a.dist), 'processed': counts([r for r in clean if r['source'] == 'ch0'], a.dist),
         '4 mics summed': counts([r for r in clean if r['source'] == 'sum'], a.dist)}
    print('\n%d m: runs where the turned take kept FEWER consonants than its facing take' % a.dist)
    print('%-14s' % '' + ''.join('%16s' % n for n in S))
    for g in dov.ANGLES[1:]:
        print('%-14s' % ('%d°' % g) + ''.join('%11d of %3d' % (sum(1 for k in G if G[k][g] < G[k][0]), len(G)) for G in S.values()))
    print('facing takes keeping 10+ of the 19 consonants: ' + '   '.join('%s %d' % (n, sum(1 for k in G if G[k][0] >= 10)) for n, G in S.items()))


if __name__ == '__main__':
    main()
