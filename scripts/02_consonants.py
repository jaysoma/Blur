#!/usr/bin/env python3
"""Step 2: which consonants survive the turn. From step 1's decodes:

  1. per distance and head angle, the share of the sentence's 19 consonants the decoder never heard;
  2. per consonant, in how many takes it was heard, facing and with the back turned, and what it became when turned;
  3. every turned take against the SAME person's facing take, same place, same run-through: did it keep fewer
     consonants, the same, or more? Nothing is pooled across people or places for this one; each pair is its own test.

usage: python scripts/02_consonants.py results/fox_all.jsonl [--out results/consonants.json]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur.dov import ANGLES                                      # noqa: E402
from blur.sentence import CONS_SLOTS, NAME, REF, kept_slots      # noqa: E402
from blur.sounds import respell, tokenize                         # noqa: E402

TURN = {0: 'facing', 45: '45°', 90: 'side-on', 135: '135°', 180: 'back turned', 225: '225°', 270: 'side-on, other side',
        315: '45°, other side'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('fox'); ap.add_argument('--out')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.fox, encoding='utf-8')]
    takes = []
    for r in rows:
        toks = tokenize(r['heard_en'] or '')
        kept = kept_slots(toks)
        takes.append({**{k: r[k] for k in ('person', 'room', 'place', 'trial', 'spot', 'dist_m', 'dov_deg')},
                      'kept': sorted(kept), 'n_cons': sum(1 for k in CONS_SLOTS if k in kept)})
    dists = sorted({t['dist_m'] for t in takes})

    print('1. Share of the 19 consonants never heard, all takes at each head angle')
    for d in dists:
        cells = []
        for g in ANGLES:
            T = [t for t in takes if t['dist_m'] == d and t['dov_deg'] == g]
            cells.append('%s %.0f%%' % (TURN[g], 100 * sum(19 - t['n_cons'] for t in T) / (19 * len(T))))
        print('   %d m: %s' % (d, ' | '.join(cells)))

    print('\n2. Takes that kept each consonant, facing -> back turned (of the takes at that distance and angle)')
    for d in dists:
        F = [t for t in takes if t['dist_m'] == d and t['dov_deg'] == 0]
        B = [t for t in takes if t['dist_m'] == d and t['dov_deg'] == 180]
        print('   %d m (%d takes each)' % (d, len(F)))
        for k in CONS_SLOTS:
            print('      %-28s %4d -> %4d' % (NAME[k], sum(k in t['kept'] for t in F), sum(k in t['kept'] for t in B)))

    print('\n3. Each turned take against the same person\'s facing take (same place, same run): fewer / same / more consonants')
    pair = defaultdict(dict)
    for t in takes:
        pair[(t['person'], t['room'], t['place'], t['trial'], t['spot'], t['dist_m'])][t['dov_deg']] = t['n_cons']
    for d in dists:
        K = [k for k in pair if k[5] == d and len(pair[k]) == len(ANGLES)]
        print('   %d m, %d place-runs' % (d, len(K)))
        for g in ANGLES[1:]:
            f = sum(1 for k in K if pair[k][g] < pair[k][0]); s = sum(1 for k in K if pair[k][g] == pair[k][0])
            print('      %-22s %3d / %3d / %3d' % (TURN[g], f, s, len(K) - f - s))
        people = sorted({k[0] for k in K}, key=lambda p: int(p[1:]))
        print('      back turned, per person (fewer of their runs): ' + '  '.join(
            '%s %d/%d' % (p, sum(1 for k in K if k[0] == p and pair[k][180] < pair[k][0]), sum(1 for k in K if k[0] == p)) for p in people))
    if a.out:
        Path(a.out).write_text(json.dumps(takes), encoding='utf-8')
        print('\nwrote', a.out)


if __name__ == '__main__':
    main()
