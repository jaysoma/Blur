"""The fox sentence, sound by sound, and which of its sounds a decode kept.

A decode is lined up against the sentence's 30 sounds (edit distance: a swap within consonants or within vowels costs
1, across 1.6, a drop or an insert 1). A sound is "kept" when its own sound was decoded in its place. Vowels accept
their near variants (ah/aah, the reduced "uh"s); consonants must match exactly, except the end of "jumps", where many
speakers say "jumped".
"""
from .sounds import CONS

FOX = 'a quick brown fox jumps over the lazy dog'

REF = ['ə', 'k', 'w', 'ɪ', 'k', 'b', 'ɹ', 'aʊ', 'n', 'f', 'ɑː', 'k', 's', 'dʒ', 'ʌ', 'm', 'p', 's', 'oʊ', 'v', 'ɚ', 'ð',
       'ə', 'l', 'eɪ', 'z', 'i', 'd', 'ɔ', 'ɡ']

NAME = {0: 'the "a"', 1: 'the k in quick', 2: 'the w in quick', 3: 'the i in quick', 4: 'the k ending quick',
        5: 'the b in brown', 6: 'the r in brown', 7: 'the "ow" in brown', 8: 'the n in brown', 9: 'the f in fox',
        10: 'the "ah" in fox', 11: 'the k in fox', 12: 'the s in fox', 13: 'the j in jumps', 14: 'the u in jumps',
        15: 'the m in jumps', 16: 'the p in jumps', 17: 'the s (or t) ending jumps', 18: 'the "oh" in over',
        19: 'the v in over', 20: 'the "er" in over', 21: 'the th in the', 22: 'the "uh" in the', 23: 'the l in lazy',
        24: 'the "ay" in lazy', 25: 'the z in lazy', 26: 'the "ee" in lazy', 27: 'the d in dog', 28: 'the "aw" in dog',
        29: 'the g in dog'}

# each letter of FOX -> the sounds it spells (None for spaces); "x" is two sounds, "ck" and "th" share one
LETTERS = [[0], None, [1], [2], [3], [4], [4], None, [5], [6], [7], [7], [8], None, [9], [10], [11, 12], None,
           [13], [14], [15], [16], [17], None, [18], [19], [20], [20], None, [21], [21], [22], None,
           [23], [24], [25], [26], None, [27], [28], [29]]

SAME = {17: {'s', 't'}, 10: {'ɑː', 'ɑ'}, 28: {'ɔ', 'ɔː', 'ɑ', 'ɑː'}, 26: {'i', 'iː', 'ɪ'}, 20: {'ɚ', 'ɜ', 'ɜː'},
        0: {'ə', 'ɐ', 'ʌ', 'ɪ', 'ᵻ', 'eɪ'}, 22: {'ə', 'ɐ', 'ʌ', 'ɪ', 'ᵻ', 'i'}, 3: {'ɪ', 'ᵻ', 'i'}, 14: {'ʌ', 'ə', 'ɐ'}}

CONS_SLOTS = [k for k, p in enumerate(REF) if p in CONS]          # the sentence's 19 consonants
WORDS = FOX.split(' ')
assert len(LETTERS) == len(FOX) and len(CONS_SLOTS) == 19


def _ok(k, b):
    return b == REF[k] or b in SAME.get(k, ())


def _cost(k, b):
    return 0 if _ok(k, b) else (1 if (REF[k] in CONS) == (b in CONS) else 1.6)


def kept_slots(sounds):
    """The sentence's sound slots (0..29) that a decode, as a list of sounds, kept."""
    R, H = len(REF), len(sounds)
    D = [[0.0] * (H + 1) for _ in range(R + 1)]
    for i in range(1, R + 1): D[i][0] = i
    for j in range(1, H + 1): D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i][j] = min(D[i - 1][j - 1] + _cost(i - 1, sounds[j - 1]), D[i - 1][j] + 1, D[i][j - 1] + 1)
    i, j, kept = R, H, set()
    while i > 0:
        if j > 0 and D[i][j] == D[i - 1][j - 1] + _cost(i - 1, sounds[j - 1]):
            if _ok(i - 1, sounds[j - 1]): kept.add(i - 1)
            i -= 1; j -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            j -= 1
        else:
            i -= 1
    return kept


def word_of_slot():
    """Each sound slot -> the index of its word in WORDS."""
    out, c = {}, 0
    for wi, w in enumerate(WORDS):
        for _ in w:
            for k in (LETTERS[c] or []):
                out[k] = wi
            c += 1
        c += 1
    return out
