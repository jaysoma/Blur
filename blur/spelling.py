"""The phone's words, turned into the consonants they promise, and lined up against what the decoder heard.

The phone writes words (231 different ones on the 160 CMU takes, most of them mishearings), and no pronunciation
dictionary is needed: consonants follow spelling closely enough. The rules are approximate and are applied the same way
at every head angle, so they can't make one angle look better than another.
"""
import re

VOW = set('aeiouy')
_VOICELESS_END = ('p', 'k', 'c', 's', 'x', 'f', 'sh', 'ch')

# decoder sounds -> the same consonant classes (r-coloured vowels carry no r, matching the spelling rule for "r")
IPA = {'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'ɡ': 'g', 'f': 'f', 'v': 'v', 'θ': 'th', 'ð': 'th', 's': 's', 'z': 'z',
       'ʃ': 'sh', 'ʒ': 'zh', 'tʃ': 'ch', 'dʒ': 'j', 'm': 'm', 'n': 'n', 'ŋ': 'ng', 'l': 'l', 'ɹ': 'r', 'w': 'w', 'j': 'y',
       'h': 'h', 'ɾ': 'ɾ', 'ʔ': 'ʔ', 'n̩': 'n', 'l̩': 'l', 'əl': 'l'}


def consonants(word):
    """A word's consonants from its spelling, in sound classes: "quick" -> k w k, "jumped" -> j m p t, "box" -> b k s."""
    w = re.sub(r"[^a-z]", '', word.lower())
    if w in ('a', 'i', 'o'): return []
    if w == 'of': return ['v']
    if w == 'the': return ['th']
    out, i, n = [], 0, len(w)

    def vowel_at(k):
        return k < n and w[k] in VOW and not (w[k] == 'e' and k == n - 1)

    while i < n:
        c, nx, tri, di = w[i], w[i + 1] if i + 1 < n else '', w[i:i + 3], w[i:i + 2]
        if tri == 'tch': out.append('ch'); i += 3; continue
        if tri == 'dge': out.append('j'); i += 3; continue
        if di in ('ch', 'sh', 'th'): out.append(di); i += 2; continue
        if di == 'ph': out.append('f'); i += 2; continue
        if di == 'wh': out.append('w'); i += 2; continue
        if di == 'ck': out.append('k'); i += 2; continue
        if di == 'qu': out += ['k', 'w']; i += 2; continue
        if di == 'ng': out.append('ng'); i += 2; continue
        if di == 'nk': out += ['ng', 'k']; i += 2; continue
        if di == 'gh': i += 2; continue
        if i == 0 and di in ('kn', 'wr'): out.append('n' if di == 'kn' else 'r'); i += 2; continue
        if c in VOW and c != 'y': i += 1; continue
        if c == 'y':
            if i == 0: out.append('y')
            i += 1; continue
        if c == 'w':
            if i == 0 or w[i - 1] not in VOW: out.append('w')
            i += 1; continue
        if c == 'r':                                             # an r after a vowel is part of the vowel
            if i == 0 or vowel_at(i + 1): out.append('r')
            i += 1; continue
        if c == 'h':
            if i == 0 or vowel_at(i + 1): out.append('h')
            i += 1; continue
        if c == 'x': out += ['k', 's']; i += 1; continue
        if c == 'c': out.append('s' if nx in ('e', 'i', 'y') else 'k'); i += 1; continue
        if c == 'g': out.append('j' if (i > 0 and nx in ('e', 'i', 'y')) else 'g'); i += 1; continue
        if c == 'd' and i == n - 1 and i >= 2 and w[i - 1] == 'e':                 # -ed: t after a voiceless end
            out.append('t' if w[:i - 1].endswith(_VOICELESS_END) else 'd'); i += 1; continue
        if c == 's' and i == n - 1 and i > 0 and (w[i - 1] in VOW or w[i - 1] in 'bdglmnrvw'):
            out.append('z'); i += 1; continue
        if out and out[-1] == c and i > 0 and w[i - 1] == c: i += 1; continue       # a doubled letter
        out.append(c); i += 1
    return out


def text_consonants(text):
    return [c for w in re.findall(r"[a-z']+", text.lower()) for c in consonants(w)]


def heard_consonants(tokens):
    return [IPA[t] for t in tokens if t in IPA]


def _match(a, b):
    return a == b or (b == 'ɾ' and a in ('t', 'd')) or (b == 'ʔ' and a == 't')


def found(promised, heard):
    """How many of the phone's consonants (in order) the decoder heard. Sounds after the phone's words cost nothing:
    the phone runs behind the audio."""
    R, H = len(promised), len(heard)
    if R == 0: return 0
    D = [[0] * (H + 1) for _ in range(R + 1)]
    M = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(1, R + 1): D[i][0] = i
    for j in range(1, H + 1): D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            s = 0 if _match(promised[i - 1], heard[j - 1]) else 1
            D[i][j], M[i][j] = min((D[i - 1][j - 1] + s, M[i - 1][j - 1] + (s == 0)), (D[i - 1][j] + 1, M[i - 1][j]),
                                   (D[i][j - 1] + 1, M[i][j - 1]), key=lambda t: (t[0], -t[1]))
    j = min(range(H + 1), key=lambda jj: (D[R][jj], -M[R][jj]))
    return M[R][j]
