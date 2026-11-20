"""Reading the CMU Direction-of-Voice recordings straight out of their zip.

Ahuja et al., "Direction-of-Voice (DoV) Estimation for Intuitive Speech Interaction with Smart Devices Ecosystems",
UIST 2020; data GPL v2, github.com/FIGLAB/DirectionOfVoice (subjectrecording.zip, 5.2 GB, not redistributed here).
Inside: data/s<P>/s<P>_<room>_<place>_<trial>/<spot>/<utterance>_<dov>_<channel>.wav, 48 kHz mono 16-bit.
  room: upstairs | downstairs      place: wall | nowall (device against a wall or not)      trial: trial1 | trial2
  spot: <pos>_<dist m>_<polar deg>, e.g. B1_3_45 = 3 m out, 45 deg off to the side
  utterance: recording0 = "hey assistant", recording1 = "a quick brown fox jumps over the lazy dog"
  dov: head angle, 0 = facing the device, 45 .. 315
  channel: a ReSpeaker USB array; 0 = its processed output, 1-4 = its raw mics, 5 = playback
"""
import io
import wave
from typing import NamedTuple

import numpy as np

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


class Take(NamedTuple):
    name: str
    person: str
    room: str
    place: str
    trial: str
    spot: str
    dist_m: int
    polar_deg: int
    utt: str
    dov_deg: int
    channel: int

    @property
    def key(self):
        """The take without its channel: the same moment on every mic."""
        return self.name.rsplit('_', 1)[0]


def parse(name):
    parts = name.split('/')
    person, session, spot, fn = parts[1], parts[2], parts[3], parts[4]
    _, room, place, trial = session.split('_')
    utt, dov, ch = fn[:-4].split('_')
    _, dist, polar = spot.split('_')
    return Take(name, person, room, place, trial, spot, int(dist), int(polar), utt, int(dov), int(ch))


def takes(z, utt='recording1', channel=1, dist=None):
    """Every take of one utterance on one channel, as parsed names, sorted."""
    out = []
    for n in z.namelist():
        if not n.endswith('_%d.wav' % channel) or n.count('/') != 4:
            continue
        t = parse(n)
        if t.utt == utt and (dist is None or t.dist_m == dist):
            out.append(t)
    return sorted(out)


def read(z, name):
    """One recording as float32 at 48 kHz."""
    with wave.open(io.BytesIO(z.read(name))) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768
