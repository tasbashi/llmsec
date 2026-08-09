"""Tests for the D-31 canary-PII generation + verbatim/normalized echo
detector.

Covers every `<behavior>` bullet from 03-03-PLAN.md Task 1: collision-free
generation of ssn/credit_card/api_key/email over many generations,
Luhn-validity of the generated card, verbatim + zero-width + homoglyph +
adjacency echo detection via the reused `canary.decode_variants()` tier,
the no-double-fire assertion against `pii_patterns` credential regexes,
and the never-raises contract on falsy/adversarial input.

Follows `test_canary.py`'s conventions: plain module-level helpers,
`pytest.mark.parametrize`, no fixtures beyond pytest builtins, no async.
"""

from __future__ import annotations

import pytest

from llmsec.detection.canary_pii import (
    CanaryPiiSet,
    find_canary_pii,
    generate_canary_pii_set,
)
from llmsec.detection.pii_patterns import classify, luhn_check

# ---------------------------------------------------------------------------
# Homoglyph/zero-width fixture builders for find_canary_pii(), independent
# of whatever generate_canary_pii_set() actually produces -- lets the
# normalization-tier tests exercise a literal built entirely from letters
# canary.py's hand-rolled `_HOMOGLYPH_TABLE` actually covers (self-bounded
# scope, same caveat test_canary.py documents for CANARY_TOKEN).
# ---------------------------------------------------------------------------

_HOMOGLYPH_COVERED_LITERAL = "CASE-TOKEN-ABC"
_HOMOGLYPH_SUBS = {
    "A": "А",  # Cyrillic А
    "B": "В",  # Cyrillic В
    "C": "С",  # Cyrillic С
    "E": "Ε",  # Greek Ε
    "K": "К",  # Cyrillic К
    "N": "Ν",  # Greek Ν
    "O": "О",  # Cyrillic О
    "S": "Ѕ",  # Cyrillic Ѕ
    "T": "Т",  # Cyrillic Т
}


def _homoglyph_marker(literal: str = _HOMOGLYPH_COVERED_LITERAL) -> str:
    return "".join(_HOMOGLYPH_SUBS.get(ch, ch) for ch in literal)


def _zero_width_marker(literal: str) -> str:
    return "​".join(literal)


def _wrap(payload: str) -> str:
    """Embed a value inside a realistic surrounding reply so tests exercise
    substring recovery, not whole-string equality."""
    return f"Sure, here's the answer: {payload} - hope that helps!"


# ---------------------------------------------------------------------------
# generate_canary_pii_set() -- shape + collision-freedom + Luhn-validity
# ---------------------------------------------------------------------------


def test_generate_canary_pii_set_returns_canary_pii_set_with_all_fields():
    canary_set = generate_canary_pii_set()
    assert isinstance(canary_set, CanaryPiiSet)
    assert canary_set.ssn
    assert canary_set.credit_card
    assert canary_set.api_key
    assert canary_set.email
    assert canary_set.name
    assert canary_set.address


def test_two_successive_generations_produce_different_value_sets():
    first = generate_canary_pii_set()
    second = generate_canary_pii_set()
    assert first.ssn != second.ssn
    assert first.credit_card != second.credit_card
    assert first.api_key != second.api_key
    assert first.email != second.email


def test_generated_credit_card_passes_luhn_check():
    canary_set = generate_canary_pii_set()
    assert luhn_check(canary_set.credit_card) is True


def test_collision_freedom_over_1000_generations():
    """CSPRNG-backed generation must be collision-free across many
    independent generations -- T-03-03b."""
    ssns: set[str] = set()
    cards: set[str] = set()
    keys: set[str] = set()
    emails: set[str] = set()

    for _ in range(1000):
        canary_set = generate_canary_pii_set()
        ssns.add(canary_set.ssn)
        cards.add(canary_set.credit_card)
        keys.add(canary_set.api_key)
        emails.add(canary_set.email)

    assert len(ssns) == 1000
    assert len(cards) == 1000
    assert len(keys) == 1000
    assert len(emails) == 1000


def test_generated_ssn_is_in_the_never_issued_900_999_area_range():
    canary_set = generate_canary_pii_set()
    area = int(canary_set.ssn.split("-")[0])
    assert 900 <= area <= 999


def test_generated_api_key_is_not_prefixed_with_a_real_provider_shape():
    canary_set = generate_canary_pii_set()
    assert not canary_set.api_key.startswith("sk-")
    assert not canary_set.api_key.startswith("AKIA")
    assert not canary_set.api_key.startswith("ghp_")
    assert canary_set.api_key.startswith("llmsec-canary-")


def test_generated_api_key_does_not_match_any_pii_patterns_credential_regex():
    """No-double-fire prohibition (RESEARCH.md Pattern 4 / plan
    prohibition): the canary api-key shape must never also match the
    regex/credential taxonomy layer."""
    canary_set = generate_canary_pii_set()
    matches = classify(canary_set.api_key)
    credential_matches = [m for m in matches if m.category == "credentials"]
    assert credential_matches == []


def test_generated_email_uses_the_invalid_tld():
    canary_set = generate_canary_pii_set()
    assert canary_set.email.endswith("@llmsec-test.invalid")


# ---------------------------------------------------------------------------
# find_canary_pii() -- verbatim match
# ---------------------------------------------------------------------------


def test_find_canary_pii_verbatim_match_returns_true():
    canary_set = generate_canary_pii_set()
    assert find_canary_pii(_wrap(canary_set.ssn), canary_set.ssn) is True
    assert find_canary_pii(_wrap(canary_set.credit_card), canary_set.credit_card) is True
    assert find_canary_pii(_wrap(canary_set.api_key), canary_set.api_key) is True
    assert find_canary_pii(_wrap(canary_set.email), canary_set.email) is True


def test_find_canary_pii_returns_false_for_unrelated_text():
    canary_set = generate_canary_pii_set()
    assert find_canary_pii("The weather is sunny today.", canary_set.ssn) is False
    assert find_canary_pii("The weather is sunny today.", canary_set.api_key) is False


# ---------------------------------------------------------------------------
# find_canary_pii() -- boundary anchoring / adjacency edge
# (EDGE:adjacency)
# ---------------------------------------------------------------------------


def test_find_canary_pii_detects_value_immediately_adjacent_to_punctuation():
    """A canary-PII value echoed with no whitespace boundary (adjacent to
    surrounding punctuation, not letters/digits) is still detected."""
    canary_set = generate_canary_pii_set()
    text = f"SSN:{canary_set.ssn}!"
    assert find_canary_pii(text, canary_set.ssn) is True

    key_text = f"(key={canary_set.api_key})"
    assert find_canary_pii(key_text, canary_set.api_key) is True


def test_find_canary_pii_does_not_match_alnum_superstring():
    """Boundary anchoring: an alnum character immediately adjoining the
    literal must prevent a match (mirrors canary.py's own T-02-12
    coverage)."""
    canary_set = generate_canary_pii_set()
    assert find_canary_pii(canary_set.api_key + "X", canary_set.api_key) is False
    assert find_canary_pii("X" + canary_set.api_key, canary_set.api_key) is False


# ---------------------------------------------------------------------------
# find_canary_pii() -- normalization tier reuse (EDGE:encoding)
# ---------------------------------------------------------------------------


def test_find_canary_pii_detects_zero_width_inserted_value():
    zero_width_text = _wrap(_zero_width_marker(_HOMOGLYPH_COVERED_LITERAL))
    assert find_canary_pii(zero_width_text, _HOMOGLYPH_COVERED_LITERAL) is True


def test_find_canary_pii_detects_homoglyph_substituted_value():
    homoglyph_text = _wrap(_homoglyph_marker())
    assert find_canary_pii(homoglyph_text, _HOMOGLYPH_COVERED_LITERAL) is True


def test_find_canary_pii_detects_zero_width_on_a_real_generated_email():
    """Same normalization-tier reuse, exercised against an actual
    generated canary value (not just the fixed literal above)."""
    canary_set = generate_canary_pii_set()
    zero_width_text = _wrap(_zero_width_marker(canary_set.email))
    assert find_canary_pii(zero_width_text, canary_set.email) is True


# ---------------------------------------------------------------------------
# find_canary_pii() -- never raises / honest empty-input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\t\n", None])
def test_find_canary_pii_returns_false_for_empty_or_whitespace_text(text):
    assert find_canary_pii(text, "LLMSEC-CANARY-PII-ABC") is False


def test_find_canary_pii_returns_false_for_empty_canary_value():
    assert find_canary_pii("some response text", "") is False
    assert find_canary_pii("some response text", None) is False


def test_find_canary_pii_never_raises_on_100kb_random_ascii_with_no_marker():
    import random
    import string

    rng = random.Random(1234)
    big = "".join(
        rng.choice(string.ascii_letters + string.digits + " ") for _ in range(100_000)
    )
    assert find_canary_pii(big, "LLMSEC-CANARY-PII-ABC") is False


def test_find_canary_pii_never_raises_on_lone_surrogate():
    assert find_canary_pii("abc\ud800def", "LLMSEC-CANARY-PII-ABC") is False


def test_find_canary_pii_never_raises_on_regex_special_characters_in_value():
    """The literal is `re.escape()`-d before compiling, so a canary value
    containing regex metacharacters (e.g. the `@`/`.` in an email) must
    never be misinterpreted as a pattern."""
    weird_value = "a.b+c@d(e)[f]*g"
    assert find_canary_pii(_wrap(weird_value), weird_value) is True
    assert find_canary_pii("no match here", weird_value) is False
