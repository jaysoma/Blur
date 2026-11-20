"""Playing recordings through a phone's on-device speech recogniser (Android SpeechRecognizer).

The phone side is SiriLean's SttProbe (SiriLean/app/src/main/java/com/jay/siri/SttProbe.kt): GET /sttprobe?wav=a,b,c
plays each WAV from the app's probe/ folder into the recogniser's audio pipe at real time, one fresh session per clip,
with the live app's own recogniser settings, and logs every partial, final, word confidence and error against the
clip's clock. GET /sttprobe/log returns that log. It refuses unless the phone's microphone is off, and ships nothing.
"""
import json
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

from tqdm import tqdm

REMOTE = '/sdcard/Android/data/com.jay.siri/files/probe/'
ENDS = ('session-end', 'results', 'error', 'timeout', 'pipe-closed-early', 'start-failed', 'create-failed', 'wav-failed', 'pipe-failed')


class Phone:
    def __init__(self, url, adb, serial):
        self.url, self.adb, self.serial = url.rstrip('/'), adb, serial

    def get(self, route, timeout=20):
        with urllib.request.urlopen(self.url + route, timeout=timeout) as r:
            return json.loads(r.read() or b'{}')

    def push(self, clips):
        """The clips onto the phone, over an ADB connection that already exists (never opens a new one)."""
        subprocess.run([self.adb, '-s', self.serial, 'shell', 'mkdir', '-p', REMOTE], check=True, capture_output=True, timeout=30)
        for p in tqdm(clips, desc='push to phone', unit='clip', file=sys.stdout):
            r = subprocess.run([self.adb, '-s', self.serial, 'push', str(p), REMOTE + p.name], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError('adb push failed: ' + (r.stderr or r.stdout).strip())

    def run(self, clips, gap_ms=1500):
        """Every clip through the recogniser in turn; returns the phone's log. Two bars: all clips, and this one's seconds."""
        names = [p.stem for p in clips]
        secs = {}
        for p in clips:
            with wave.open(str(p)) as w:
                secs[p.stem] = w.getnframes() / w.getframerate()
        started = self.get('/sttprobe?wav=%s&gap=%d' % (','.join(names), gap_ms))
        if not started.get('ok'):
            raise RuntimeError('the phone refused: %s' % started.get('error'))
        outer = tqdm(total=len(names), desc='through the recogniser', unit='clip', position=0, file=sys.stdout)
        inner = tqdm(total=1, desc='this clip', unit='s', position=1, leave=False, file=sys.stdout,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f} s')
        log, cur = {}, None
        while True:
            time.sleep(1.0)
            log = self.get('/sttprobe/log')
            ev = log.get('events', [])
            outer.n = len({e['clip'] for e in ev if e['event'] in ENDS}); outer.refresh()
            i = log.get('clip', -1)
            if log.get('active') and 0 <= i < len(names):
                if names[i] != cur:
                    cur = names[i]; inner.reset(total=round(secs[cur], 1)); inner.set_description('this clip: ' + cur)
                fed = max([e['fed_ms'] for e in ev if e['clip'] == i] or [0]) / 1000
                inner.n = min(round(fed, 1), inner.total); inner.refresh()
            if not log.get('active'):
                break
        inner.close(); outer.close()
        return log
