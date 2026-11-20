"""The decoder's English sounds, how its multi-sound tokens split, and how to write sounds in plain letters.

The decoder (facebook/wav2vec2-lv-60-espeak-cv-ft) writes espeak IPA over ~390 sounds from many languages. Everything
here restricts it to the sounds American English uses, so a turned-away talker can't drift into "foreign" sounds.
"""

# American-English espeak sounds (consonants first, then vowels and r-coloured vowels)
EN = ('p b t d k ɡ f v θ ð s z ʃ ʒ h tʃ dʒ m n ŋ l ɹ w j ɾ ʔ n̩ l̩ əl '
      'i iː ɪ ɛ æ ɑ ɑː ɔ ɔː ʊ u uː ʌ ə ɚ ɜ ɜː ɐ ᵻ eɪ aɪ aʊ ɔɪ oʊ ɑːɹ ɔːɹ oːɹ ɛɹ ɪɹ ʊɹ aɪɚ').split()

CONS = set('p b t d k ɡ f v θ ð s z ʃ ʒ h tʃ dʒ m n ŋ l ɹ w j ɾ ʔ'.split())

# tokens that carry two sounds, split so each lines up against the sentence on its own
SPLIT = {'ɑːɹ': ['ɑː', 'ɹ'], 'ɔːɹ': ['ɔː', 'ɹ'], 'oːɹ': ['oː', 'ɹ'], 'ɛɹ': ['ɛ', 'ɹ'], 'ɪɹ': ['ɪ', 'ɹ'], 'ʊɹ': ['ʊ', 'ɹ'],
         'aɪɚ': ['aɪ', 'ɚ'], 'əl': ['ə', 'l'], 'n̩': ['n'], 'l̩': ['l']}

# plain-letter respelling, for people who don't read IPA
SAY = {'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'ɡ': 'g', 'f': 'f', 'v': 'v', 'θ': 'th', 'ð': 'th', 's': 's',
       'z': 'z', 'ʃ': 'sh', 'ʒ': 'zh', 'h': 'h', 'tʃ': 'ch', 'dʒ': 'j', 'm': 'm', 'n': 'n', 'ŋ': 'ng', 'l': 'l', 'ɹ': 'r',
       'w': 'w', 'j': 'y', 'ɾ': 'd', 'ʔ': "'", 'i': 'ee', 'iː': 'ee', 'ɪ': 'i', 'ɛ': 'e', 'æ': 'a', 'ɑ': 'ah', 'ɑː': 'ah',
       'ɔ': 'aw', 'ɔː': 'aw', 'ʊ': 'oo', 'u': 'oo', 'uː': 'oo', 'ʌ': 'u', 'ə': 'uh', 'ɚ': 'er', 'ɜ': 'er', 'ɜː': 'er',
       'ɐ': 'uh', 'ᵻ': 'i', 'eɪ': 'ay', 'aɪ': 'eye', 'aʊ': 'ow', 'ɔɪ': 'oy', 'oʊ': 'oh', 'oː': 'oh'}

_TOK = sorted(EN + ['oː'], key=len, reverse=True)


def tokenize(heard):
    """A decoded IPA string back into its sounds (longest token first), multi-sound tokens split."""
    out, i = [], 0
    while i < len(heard):
        t = next((t for t in _TOK if heard.startswith(t, i)), heard[i])
        out.extend(SPLIT.get(t, [t]))
        i += len(t)
    return out


def respell(token):
    """One decoder token in plain letters ("dʒ" -> "j", "aʊ" -> "ow")."""
    return ''.join(SAY.get(p, p) for p in SPLIT.get(token, [token]))
