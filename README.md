# Blur

**How much of a sentence survives as the talker turns away, read with no language model.**

Ten people said *"a quick brown fox jumps over the lazy dog"* to a small four-mic device, facing it and then turned
eight ways around, at 1, 3 and 5 m: the CMU Direction-of-Voice recordings (Ahuja et al., UIST 2020). We decode every
take from **one** of the device's raw microphones with a phoneme recogniser that has **no language model**, and count
which of the sentence's sounds still arrive.

The page: [research/blur.html](https://jaysoma.dev/research/blur.html), built from this repo's step 7.

## The blur value

As a phone's speech recogniser listens, it writes words (its partial results), leaning on its language model. A
decoder with no language model writes down the sounds that actually arrived. **Blur** is the share of the consonants
in the phone's words that the plain decoder never heard in the same audio:

```
blur = 1 - (consonants in the phone's words that were heard) / (consonants in the phone's words)
```

0% means everything the phone wrote really arrived. On the CMU takes at 1 m (ten people, one take each per angle):
facing the device, 0-44%, seven of the ten at 11% or less; back turned, 42-83%. It needs no reference recording and
no known sentence: the phone's own words are the reference. Computed at every partial in step 6 (`scripts/06_gap.py`;
the words' consonants come from their spelling, `blur/spelling.py`).

## What it shows

- **Turning away costs consonants, and it follows the turn.** At 1 m the decoder misses about a fifth of the sentence's
  19 consonants facing the device and about seven tenths with the back turned; the share climbs as the head turns away
  and falls as it comes back round. Run by run, against the same person's own facing take, the back-turned take keeps
  fewer consonants almost every time at 1 m, and most of the time at 5 m, where much is already lost facing.
- **The buzzed consonants go first** (b, d, g, v, z, th, j) and often turn into their unvoiced twins (d -> t, z -> s);
  the s holds on longest.
- **It isn't the room's noise.** Decoding the device's own processed channel, or its four mics summed, leaves the
  turn's effect almost unchanged (step 3).
- **A phone's recogniser covers for it.** Played through a phone's on-device speech recogniser (which has a language
  model), the same takes come back as words. Facing, its words match what arrived. Turned away, it keeps writing a
  sentence whose sounds never reached the microphone, or it gives up. The share of the phone's consonants that never
  arrived separates facing from back-turned at a known distance, with no facing take to compare against (step 6).

What it is not: an angle estimator. A 45° turn is close to facing on every measure here, and beyond about 1 m the
distance has to be known before one fixed cut works. It is a cue that complements array-based facing estimators: it
reads how much of the speech got through, from one microphone, where those read where the sound came from.

## Reproduce

```
pip install -r requirements.txt
# the recordings: subjectrecording.zip (5.2 GB) from github.com/FIGLAB/DirectionOfVoice
python scripts/01_decode.py subjectrecording.zip results/fox_all.jsonl                    # ~30 min, CPU
python scripts/02_consonants.py results/fox_all.jsonl                                     # seconds
python scripts/03_cleanup_check.py subjectrecording.zip results/fox_all.jsonl results/fox_5m_clean.jsonl   # ~20 min
python scripts/04_phone_clips.py subjectrecording.zip clips/                              # 160 clips
python scripts/05_phone_run.py clips/ results/phone_log.json --phone http://PHONE:8792 --serial ADB_SERIAL  # ~26 min, real time
python scripts/06_gap.py results/phone_log.json clips/ results/gap.json                   # ~10 min
python scripts/07_page_data.py results/fox_all.jsonl results/gap.json clips/ path/to/HomePage/research
```

`results/` holds the outputs of steps 1, 3, 5 and 6, so steps 2, 6 and 7 can be rerun without the dataset or a
phone (7 needs `clips/` from step 4 for the audio).

Step 5 needs an Android phone running the SiriLean app, whose `SttProbe`
(`SiriLean/app/src/main/java/com/jay/siri/SttProbe.kt`) plays each clip into the on-device `SpeechRecognizer`'s audio
pipe at real time, one fresh session per clip, with the live app's own recogniser settings, and logs every partial and
final against the clip's clock. Any phone recogniser driven the same way would do.

## Layout

```
blur/
  decoder.py    the plain decoder: wav2vec2-lv-60-espeak-cv-ft, English sounds only, best sound per 20 ms, no language model
  sounds.py     the English sound set, splitting multi-sound tokens, plain-letter respelling
  sentence.py   the fox sentence sound by sound; which sounds a decode kept
  dov.py        reading the CMU recordings out of their zip
  spelling.py   the phone's words -> the consonants they promise; lined up against what was heard
  phone.py      driving the phone's recogniser over HTTP + ADB
scripts/        steps 1-7
results/        fox_all.jsonl (step 1), fox_5m_clean.jsonl (3), phone_log.json (5), gap.json (6)
```

## Credits

Recordings: CMU Direction-of-Voice dataset, Ahuja, Kong, Goel and Harrison, UIST 2020, GPL v2. Not redistributed
here; the results files hold decodes and counts only. Decoder: facebook/wav2vec2-lv-60-espeak-cv-ft (Apache 2.0).
