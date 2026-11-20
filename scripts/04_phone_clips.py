#!/usr/bin/env python3
"""Step 4: cut the takes for the phone. For each person, one place (upstairs, device away from the wall, run 2,
straight in front; the next session in a fixed order if any take is missing) at 1 and 3 m, all eight head angles,
from the same raw mic as step 1. Each clip gets 1 s of silence before and 2 s after, so the recogniser hears the start
and has time to finish; 48 kHz mono, which the phone's own 48->16 kHz filter takes down.

usage: python scripts/04_phone_clips.py subjectrecording.zip clips/ [--dists 1,3] [--channel 1]
"""
import argparse
import json
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur import dov                                     # noqa: E402

ORDER = [('upstairs', 'nowall', 'trial2'), ('upstairs', 'nowall', 'trial1'), ('upstairs', 'wall', 'trial2'), ('downstairs', 'nowall', 'trial2')]
SPOT = {1: 'A0_1_0', 3: 'B0_3_0', 5: 'C0_5_0'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('out')
    ap.add_argument('--dists', default='1,3'); ap.add_argument('--channel', type=int, default=1)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip); names = set(z.namelist())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    people = sorted({dov.parse(n).person for n in names if n.count('/') == 4}, key=lambda p: int(p[1:]))
    n, manifest = 0, {}
    for p in people:
        for d in map(int, a.dists.split(',')):
            for room, place, trial in ORDER:
                pre = 'data/%s/%s_%s_%s_%s/%s/' % (p, p, room, place, trial, SPOT[d])
                want = [pre + 'recording1_%d_%d.wav' % (g, a.channel) for g in dov.ANGLES]
                if all(w in names for w in want): break
            else:
                print('missing: %s at %d m' % (p, d)); continue
            if (room, place, trial) != ORDER[0]:
                print('%s at %d m uses %s %s %s' % (p, d, room, place, trial))
            for g, w in zip(dov.ANGLES, want):
                x = (dov.read(z, w) * 32768).astype('<i2')
                y = np.concatenate([np.zeros(48000, '<i2'), x, np.zeros(96000, '<i2')])
                clip = 'cmu-p%02d-%dm-%03d' % (int(p[1:]), d, g)
                with wave.open(str(out / (clip + '.wav')), 'wb') as f:
                    f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000); f.writeframes(y.tobytes())
                manifest[clip] = w
                n += 1
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print('%d clips in %s (manifest.json: which CMU take each clip is)' % (n, out))


if __name__ == '__main__':
    main()
