#!/usr/bin/env python3
"""Step 7: the data behind the page (HomePage research/blur.html): blur.json and the phone takes' audio at 16 kHz.

  dist[d].kept[angle][letter]   takes (of dist[d].n[angle]) in which each letter's sound was heard     (step 1, all takes)
  dropped[d][angle]             consonants never heard, of 19 x takes                                   (step 1)
  phone_blur[d][angle]          per letter: takes where the phone's finished sentence had its word, and where the
                                plain decoder heard its sound, on the phone takes; and how often the phone gave up
  phone_dropped[d][angle]       consonants the phone didn't write / that didn't arrive, of 19 x takes
  gap[d][angle]                 per person: share of the phone's consonants that never arrived (step 6, last partial)
  takes                         every phone take: audio, the phone's partials and final, the plain sounds' timeline

usage: python scripts/07_page_data.py results/fox_all.jsonl results/gap.json clips/ OUT_DIR
       (writes OUT_DIR/blur.json and OUT_DIR/blur/cmu/*.wav; OUT_DIR = HomePage/research for the live page)
"""
import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur.decoder import Decoder, MODEL                                              # noqa: E402
from blur.dov import ANGLES, parse                                                    # noqa: E402
from blur.sentence import CONS_SLOTS, FOX, LETTERS, NAME, WORDS, kept_slots, word_of_slot   # noqa: E402
from blur.sounds import tokenize                                                       # noqa: E402

DEFAULT_SESSION = ('upstairs', 'nowall', 'trial2')
SPOT = {1: 'A0_1_0', 3: 'B0_3_0', 5: 'C0_5_0'}
WROTE = {'jumps': {'jumps', 'jumped', 'jump'}, 'dog': {'dog', 'dogs'}}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('fox'); ap.add_argument('gap'); ap.add_argument('clips'); ap.add_argument('out')
    ap.add_argument('--model', default=MODEL)
    a = ap.parse_args()
    out = Path(a.out); (out / 'blur' / 'cmu').mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.fox, encoding='utf-8')]
    dists = sorted({r['dist_m'] for r in rows})

    # every take: which sound slots it kept
    kept, by_take = {}, {}
    for r in tqdm(rows, desc='lining up takes', unit='take', file=sys.stdout):
        k = kept_slots(tokenize(r['heard_en'] or ''))
        kept[r['clip']] = k
        by_take[r['clip'].rsplit('_', 1)[0]] = k
    dist, dropped = {}, {}
    for d in dists:
        dist[str(d)] = {'n': {}, 'kept': {}}
        dropped[str(d)] = {}
        for g in ANGLES:
            T = [r['clip'] for r in rows if r['dist_m'] == d and r['dov_deg'] == g]
            dist[str(d)]['n'][str(g)] = len(T)
            dist[str(d)]['kept'][str(g)] = [sum(1 for c in T if sl and all(s in kept[c] for s in sl)) if sl else 0 for sl in LETTERS]
            dropped[str(d)][str(g)] = {'miss': sum(19 - sum(1 for s in CONS_SLOTS if s in kept[c]) for c in T), 'of': 19 * len(T)}
    letters = [{'ch': ch, 'sound': (NAME[sl[0]] if len(sl) == 1 else 'the k and s in fox') if sl else None} for ch, sl in zip(FOX, LETTERS)]

    # the phone takes
    gap = json.loads(Path(a.gap).read_text(encoding='utf-8'))
    man_f = Path(a.clips) / 'manifest.json'
    manifest = json.loads(man_f.read_text(encoding='utf-8')) if man_f.exists() else {}
    wos = word_of_slot()
    dec = Decoder(a.model)
    G, PB, PD, takes = {}, {}, {}, []
    for r in tqdm(gap, desc='phone takes for the page', unit='take', file=sys.stdout):
        d, g, p = r['dist_m'], r['deg'], r['person']
        src = manifest.get(r['clip']) or 'data/s%d/s%d_%s_%s_%s/%s/recording1_%d_1.wav' % ((p, p) + DEFAULT_SESSION + (SPOT[d], g))
        arrived = by_take[parse(src).key]
        last = r['steps'][-1] if r['steps'] else None
        share = (last['gap'] / last['claimed']) if (last and last['claimed']) else None
        gave = r['gave_up'] or r['no_partials']
        G.setdefault(str(d), {}).setdefault(str(g), []).append(
            {'p': p, 'share': None if share is None else round(share, 3), 'gave_up': gave, 'wrote': last['text'] if last else ''})
        said = set(re.findall(r"[a-z']+", (r['final'] or '').lower()))
        have = [bool(said & WROTE.get(w, {w})) for w in WORDS]
        e = PB.setdefault(str(d), {}).setdefault(str(g), {'n': 0, 'phone': [0] * len(FOX), 'arrived': [0] * len(FOX), 'gave_up': 0})
        e['n'] += 1; e['gave_up'] += int(not r['final'])
        for c, sl in enumerate(LETTERS):
            if sl is None: continue
            e['phone'][c] += int(have[wos[sl[0]]])
            e['arrived'][c] += int(all(s in arrived for s in sl))
        q = PD.setdefault(str(d), {}).setdefault(str(g), {'phone_miss': 0, 'arrived_miss': 0, 'of': 0})
        q['phone_miss'] += sum(1 for s in CONS_SLOTS if not have[wos[s]])
        q['arrived_miss'] += sum(1 for s in CONS_SLOTS if s not in arrived); q['of'] += 19
        with wave.open(str(Path(a.clips) / (r['clip'] + '.wav'))) as w:
            x16 = dec.to16k(np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768, w.getframerate())
        with wave.open(str(out / 'blur' / 'cmu' / (r['clip'] + '.wav')), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(np.clip(x16 * 32768, -32768, 32767).astype('<i2').tobytes())
        takes.append({'p': p, 'dist': d, 'deg': g, 'audio': 'blur/cmu/' + r['clip'] + '.wav',
                      'phone': [{'t': round(s['t'], 3), 'text': s['text']} for s in r['steps']],
                      'final': r['final'], 'gave_up': gave, 'plain': dec.decode(x16).timeline})
    for dd in G.values():
        for v in dd.values():
            v.sort(key=lambda e: e['p'])
    doc = {'sentence': FOX, 'letters': letters, 'words': WORDS, 'angles': ANGLES, 'dist': dist, 'dropped': dropped,
           'phone_blur': PB, 'phone_dropped': PD, 'gap': G, 'takes': takes,
           'about': {'people': len({r['person'] for r in rows}), 'takes': len(rows),
                     'mic': 'one raw mic (channel 1) of a ReSpeaker USB array',
                     'decoder': 'wav2vec2-lv-60-espeak-cv-ft, English sounds only, no language model',
                     'corpus': 'CMU Direction-of-Voice (Ahuja et al., UIST 2020), GPL v2',
                     'phone': 'Samsung Galaxy Z Flip4, on-device SpeechRecognizer',
                     'gap_takes': len(takes), 'gap_spot': 'upstairs, device away from the wall, straight in front, run 2'}}
    (out / 'blur.json').write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    print('wrote %s and %d clips' % (out / 'blur.json', len(takes)))


if __name__ == '__main__':
    main()
