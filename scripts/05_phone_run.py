#!/usr/bin/env python3
"""Step 5: play step 4's clips through the phone's speech recogniser, one fresh session each, and save its log.
Real time: about 9.6 s per clip (the clip, its last result, the gap), so 160 clips take ~26 min.
The phone must have its microphone off; the probe refuses otherwise. ADB must already be connected to it.

usage: python scripts/05_phone_run.py clips/ results/phone_log.json --phone http://PHONE:8792 --serial ADB_SERIAL
       [--adb path/to/adb] [--gap 1500] [--no-push]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blur.phone import Phone                              # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('clips'); ap.add_argument('out')
    ap.add_argument('--phone', required=True); ap.add_argument('--serial', required=True)
    ap.add_argument('--adb', default='adb'); ap.add_argument('--gap', type=int, default=1500)
    ap.add_argument('--no-push', action='store_true')
    a = ap.parse_args()
    clips = sorted(Path(a.clips).glob('*.wav'))
    ph = Phone(a.phone, a.adb, a.serial)
    if not a.no_push:
        ph.push(clips)
    log = ph.run(clips, a.gap)
    Path(a.out).write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote %s: %d events' % (a.out, len(log.get('events', []))))


if __name__ == '__main__':
    main()
