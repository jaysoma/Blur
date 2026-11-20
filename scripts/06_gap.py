#!/usr/bin/env python3
"""Step 6: the phone's partials against what arrived. At every partial result the phone logged, the consonants its
words so far promise (from their spelling) are lined up against the consonants the plain decoder heard in the same
clip up to that moment, re-decoded on that prefix alone so it never hears the future. The gap is the share of the
phone's consonants that never arrived. Then: per head angle, how many people's gap at the phone's last partial was
bigger than their own facing take's; how often the phone gave up; and how a single fixed cut does with no facing take.

usage: python scripts/06_gap.py results/phone_log.json clips/ results/gap.json [--limit N]
(about 10 min for 160 clips: every partial is a fresh decode)
"""
import argparse
import json
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur.decoder import Decoder, MODEL                                        # noqa: E402
from blur.dov import ANGLES                                                    # noqa: E402
from blur.spelling import found, heard_consonants, text_consonants             # noqa: E402

CUT = 0.40


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('log'); ap.add_argument('clips'); ap.add_argument('out')
    ap.add_argument('--model', default=MODEL); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    log = json.loads(Path(a.log).read_text(encoding='utf-8'))
    by = defaultdict(list)
    for e in log['events']:
        by[e['clip']].append(e)
    clips = log['clips'][:a.limit or None]
    dec = Decoder(a.model)
    bar = tqdm(total=sum(len({p['ms'] for p in by[k] if p['event'] == 'partial'}) for k in range(len(clips))),
               desc='partials re-decoded', unit='partial', file=sys.stdout)
    rows = []
    for k, name in enumerate(clips):
        _, p, dist, deg = name.split('-')
        with wave.open(str(Path(a.clips) / (name + '.wav'))) as w:
            x16 = dec.to16k(np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768, w.getframerate())
        cache, steps = {}, []
        for e in [e for e in by[k] if e['event'] == 'partial']:
            t = e['ms'] / 1000
            if t not in cache:
                cache[t] = heard_consonants(dec.decode(x16[:max(1600, int(t * 16000))]).tokens)
                bar.update(1)
            promised = text_consonants(e['text'])
            m = found(promised, cache[t])
            steps.append({'t': t, 'text': e['text'], 'claimed': len(promised), 'found': m, 'gap': len(promised) - m})
        errs = [e.get('why') for e in by[k] if e['event'] == 'error']
        final = next((e['text'] for e in by[k] if e['event'] in ('segment', 'results') and e.get('text')), '')
        rows.append({'clip': name, 'person': int(p[1:]), 'dist_m': int(dist[:-1]), 'deg': int(deg), 'steps': steps,
                     'final': final, 'gave_up': bool(errs) and not final, 'no_partials': not steps})
    bar.close()
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', a.out)

    def share(r):
        s = r['steps'][-1] if r['steps'] else None
        return None if (not s or not s['claimed']) else s['gap'] / s['claimed']
    G = {(r['person'], r['dist_m'], r['deg']): r for r in rows}
    for d in sorted({r['dist_m'] for r in rows}):
        people = sorted({r['person'] for r in rows if r['dist_m'] == d})
        print('\n%d m' % d)
        print('   phone gave up or wrote nothing: ' + '  '.join('%d°:%d' % (g, sum(1 for p in people if (p, d, g) in G and (G[(p, d, g)]['gave_up'] or G[(p, d, g)]['no_partials']))) for g in ANGLES))
        for g in ANGLES[1:]:
            pairs = [(share(G[(p, d, g)]), share(G[(p, d, 0)])) for p in people if (p, d, g) in G and (p, d, 0) in G]
            pairs = [(x, f) for x, f in pairs if x is not None and f is not None]
            print('   %4d°: bigger gap than the same person facing in %d of %d' % (g, sum(1 for x, f in pairs if x > f), len(pairs)))
        fa = [share(G[(p, d, 0)]) for p in people if (p, d, 0) in G]
        bk = [share(G[(p, d, 180)]) for p in people if (p, d, 180) in G]
        print('   one fixed cut at %.0f%% missing: facing under it %d of %d; back turned over it %d of %d' % (
            100 * CUT, sum(1 for x in fa if x is not None and x <= CUT), sum(1 for x in fa if x is not None),
            sum(1 for x in bk if x is not None and x > CUT), sum(1 for x in bk if x is not None)))


if __name__ == '__main__':
    main()
