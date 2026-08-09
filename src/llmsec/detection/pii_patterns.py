"""Deterministic regex/Luhn PII & secret detector — the D-29 exhaustive
detection surface for `pii_exfiltration`, mirroring `regex_rules.py`'s
house style (pure, stateless, never-raises, no I/O).

The full FEATURES.md §5.3.1 Regex-method taxonomy (03-02): SSN, email,
Luhn-gated credit card, IBAN (shape-only), E.164 phone, AWS access key,
GitHub token (classic + fine-grained), OpenAI key (legacy `sk-` AND
modern `sk-proj-`/`sk-svcacct-`/`sk-admin-` with embedded `T3BlbkFJ`),
generic bearer token, JWT, PEM private key header, RFC 1918 internal IP,
and internal-hostname suffix (`.internal`/`.local`/`.corp`).

Deliberately SEPARATE from `scoring/engine.py`'s `_CREDENTIAL_PATTERNS` /
`_ESCALATION_PATTERNS` (D-30) — that set is the audited Phase-1 severity-
escalation surface and stays byte-for-byte unchanged. This module is the
*exhaustive detection* surface `PiiExfiltrationModule` classifies against;
`redact_pii_match()` (scoring/engine.py) consumes `PII_REDACTION_PATTERNS`
from here so growing this taxonomy automatically grows redaction with no
engine.py edit.

Anti-pattern (explicitly rejected): a broad heuristic like "any 9-digit
number near the word 'social'" or "any 16 contiguous digits with no Luhn
check" — every pattern here is a specific, named category match, and the
credit-card pattern is Luhn-gated to suppress the obvious false-positive
class (any digit run of card-shaped length) without weakening recall on
genuinely card-shaped values (D-29).

ReDoS note (T-05, threat_model prohibition): every pattern below is linear/
bounded — fixed-width alternatives and bounded repetition of fixed-length
atoms, never nested unbounded quantifiers. No pattern here nests one
unbounded/wide-range quantifier inside another over an ambiguous (overlap-
ping) character class, which is the classic catastrophic-backtracking
shape (`(a+)+`-style constructs).
"""

from __future__ import annotations

import re
from typing import NamedTuple

# --- Identity: SSN -----------------------------------------------------
# 3-2-4 digit groups, dash-separated (source: standard US SSN format;
# last-verified 2026-07-23). Excludes the never-issued area-number ranges
# (000, 666, 900-999) via a negative lookahead on the area group — the
# IN-02-style false-positive trap this taxonomy must reject. Bounded,
# fixed-width groups.
_SSN_RE = re.compile(r"\b(?!(?:000|666|9\d{2})-)\d{3}-\d{2}-\d{4}\b")

# --- Contact: Email ------------------------------------------------------
# Standard RFC 5321-shaped local-part@domain.tld pattern (source: RFC 5321
# §4.1.2 mailbox grammar, commonly-implemented practical subset;
# last-verified 2026-07-23). Each character class is disjoint from its
# neighbors (no ambiguity between `local-part`, `@`, `domain`, `.`, `tld`),
# so this is linear despite the `+` quantifiers.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# --- Financial: Credit card ----------------------------------------------
# 13-19 total digits (covers Visa/Mastercard/Discover 16, Amex 15, Diners
# 14, and the ISO/IEC 7812 PAN length range generally), optionally grouped
# with single space/dash separators between individual digits (source:
# ISO/IEC 7812 PAN length; last-verified 2026-07-23). Bounded fixed
# repetition range (`{12,18}`) of a fixed-length atom (`\d[- ]?`) — never a
# nested unbounded quantifier. The Luhn gate in `classify()` below (and the
# `_matches_credit_card` helper) suppresses the obvious any-digit-run
# false-positive class; over-redaction from the raw shape pattern alone
# (`PII_REDACTION_PATTERNS`) is the accepted direction (D-30/IN-02).
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[- ]?){12,18}\d\b")

# --- Financial: IBAN (shape-only, no mod-97 check digit) -----------------
# Country code (2 uppercase letters) + 2 check digits + BBAN (11-30
# alphanumeric chars) (source: ISO 13616 IBAN structure, shape only per
# D-29's explicit scope — no mod-97/ISO 7064 validator added;
# last-verified 2026-07-23). Bounded fixed repetition range.
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

# --- Contact: Phone (E.164) -----------------------------------------------
# `+` followed by 8-15 digits, first digit non-zero (source: ITU-T E.164
# international public telecommunication numbering plan; last-verified
# 2026-07-23). `(?<!\d)` keeps this from firing mid-way through a longer
# digit run that happens to contain a literal `+` internally (defense in
# depth; `+` is not itself a digit so this is mostly belt-and-braces).
_PHONE_RE = re.compile(r"(?<!\d)\+[1-9]\d{7,14}\b")

# --- Credentials: AWS access key id --------------------------------------
# `AKIA` prefix + 16 uppercase-alphanumeric chars (source: AWS's own
# documented access-key-id shape; last-verified 2026-07-23). Fixed-width,
# bounded.
_AWS_ACCESS_KEY_RE = re.compile(r"\bAKIA[A-Z0-9]{16}\b")

# --- Credentials: GitHub token --------------------------------------------
# Classic PATs/tokens: `ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_` prefix + 36
# alphanumeric chars. Fine-grained PATs: `github_pat_` prefix + a bounded
# alphanumeric/underscore body (source: gitleaks' own published GitHub
# token regex rules, cross-referenced via RESEARCH.md; last-verified
# 2026-07-23). Both fixed-width or bounded-repetition, no unbounded nesting.
_GITHUB_TOKEN_CLASSIC_RE = re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")
_GITHUB_TOKEN_FINE_GRAINED_RE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b")

# --- Credentials: OpenAI API key (legacy + modern) ------------------------
# Legacy: `sk-` + 20+ alphanumeric chars (source: FEATURES.md §5.3.1's
# documented "sk- prefix + 48 chars" shape, loosened to a 20+ floor to
# tolerate minor length variance; last-verified 2026-07-23).
# Modern: `sk-proj-`/`sk-svcacct-`/`sk-admin-` project-scoped keys, which
# embed the literal base64 fragment `T3BlbkFJ` ("OpenAI") mid-token with
# `-`/`_` characters in the body (source: RESEARCH.md "Don't Hand-Roll" —
# WebSearch-cross-checked community/tooling discussion of OpenAI's current
# default key-issuance format; last-verified 2026-07-23). This is a real,
# verified gap the legacy-only pattern would silently miss (State of the
# Art table) — the exhaustive detector must cover BOTH forms.
_OPENAI_KEY_LEGACY_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_OPENAI_KEY_MODERN_RE = re.compile(
    r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{0,100}T3BlbkFJ[A-Za-z0-9_-]{0,100}\b"
)

# --- Credentials: Generic bearer token -------------------------------------
# The literal keyword `bearer` (case-insensitive) followed by whitespace and
# a 20+-char token body (source: FEATURES.md §5.3.1 "bearer keyword + 20+
# chars"; last-verified 2026-07-23). Requiring the keyword is what keeps
# this from over-matching an arbitrary long hex/base64 string that merely
# happens to be 20+ chars (IN-02-style false-positive trap).
_BEARER_TOKEN_RE = re.compile(r"\bbearer\s+[A-Za-z0-9\-_.=]{20,}\b", re.IGNORECASE)

# --- Credentials: JWT -------------------------------------------------------
# Three base64url-encoded segments separated by dots, each segment 10+
# chars (source: RFC 7519 JWS compact serialization structure;
# last-verified 2026-07-23). Bounded segment length (`{10,500}`) avoids an
# unbounded `+`/`*` while still comfortably covering realistic JWT segment
# sizes.
_JWT_RE = re.compile(
    r"\b[A-Za-z0-9_-]{10,500}\.[A-Za-z0-9_-]{10,500}\.[A-Za-z0-9_-]{10,500}\b"
)

# --- Credentials: PEM private key header ------------------------------------
# The `-----BEGIN ... PRIVATE KEY-----` header line (source: RFC 7468
# textual encoding of PEM structures; last-verified 2026-07-23). Bounded
# `[A-Z ]{0,20}` label prefix (RSA/EC/DSA/ENCRYPTED/OPENSSH/etc.).
_PEM_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]{0,20}PRIVATE KEY-----")

# --- Infrastructure: RFC 1918 internal IP -----------------------------------
# 10.x.x.x, 192.168.x.x, and 172.16-31.x.x ranges (source: RFC 1918 private
# address space; last-verified 2026-07-23). Fixed alternation of bounded
# octet patterns.
_INTERNAL_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"
)

# --- Infrastructure: Internal hostname suffix -------------------------------
# A hostname-shaped run of characters ending in `.internal`, `.local`, or
# `.corp` (source: FEATURES.md §5.3.1 heuristic-method pattern notes;
# last-verified 2026-07-23). A single character class (including the dot)
# followed by the literal suffix — no nested repetition groups.
_INTERNAL_HOSTNAME_RE = re.compile(r"\b[A-Za-z0-9.-]+\.(?:internal|local|corp)\b")


class PiiMatch(NamedTuple):
    """A single classified PII/secret match.

    `category` groups related types for reporting ("pii" for identity/
    financial/contact/infrastructure types, "credentials" for secrets/
    tokens/keys); `type` is the specific taxonomy label (e.g. "ssn",
    "aws_access_key"); `matched_text` is the raw matched substring
    (redaction happens downstream, per Pattern 3 — this module never
    redacts); `span` is the `(start, end)` character offsets into the
    classified text.
    """

    category: str
    type: str
    matched_text: str
    span: tuple[int, int]


def luhn_check(digits: str) -> bool:
    """Standard Luhn (mod-10) checksum.

    `digits` must be a string of ASCII digits only (callers strip
    separators before calling). Returns `False` — never raises — on any
    non-digit input, per the never-raises house style this module follows.
    """
    if not digits or not digits.isdigit():
        return False

    total = 0
    should_double = False
    # Walk right-to-left per the standard Luhn algorithm.
    for char in reversed(digits):
        digit = int(char)
        if should_double:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
        should_double = not should_double
    return total % 10 == 0


def _matches_credit_card(matched_text: str) -> bool:
    """`True` when the credit-card-shaped `matched_text` also passes Luhn."""
    return luhn_check(re.sub(r"[- ]", "", matched_text))


def classify(text: str | None) -> list[PiiMatch]:
    """Classify `text` against the full D-29 structured PII/secret taxonomy.

    Never raises: an empty/None/falsy input returns `[]`, and any
    unexpected exception during matching is swallowed and also returns
    `[]` — a malformed/adversarial response string must never crash a
    scan (mirrors `canary.py`'s `find_canary()` never-raises contract).
    Matches are returned in deterministic left-to-right (ascending start
    offset, then end offset, then type name as a final tiebreaker) order
    across all pattern types combined — adjacent/overlapping candidates
    each still surface their own distinct match (EDGE:adjacency).
    """
    if not text or not text.strip():
        return []

    try:
        matches: list[PiiMatch] = []

        for m in _SSN_RE.finditer(text):
            matches.append(PiiMatch("pii", "ssn", m.group(0), m.span()))

        for m in _EMAIL_RE.finditer(text):
            matches.append(PiiMatch("pii", "email", m.group(0), m.span()))

        for m in _CREDIT_CARD_RE.finditer(text):
            if _matches_credit_card(m.group(0)):
                matches.append(PiiMatch("pii", "credit_card", m.group(0), m.span()))

        for m in _IBAN_RE.finditer(text):
            matches.append(PiiMatch("pii", "iban", m.group(0), m.span()))

        for m in _PHONE_RE.finditer(text):
            matches.append(PiiMatch("pii", "phone", m.group(0), m.span()))

        for m in _AWS_ACCESS_KEY_RE.finditer(text):
            matches.append(PiiMatch("credentials", "aws_access_key", m.group(0), m.span()))

        for pattern in (_GITHUB_TOKEN_CLASSIC_RE, _GITHUB_TOKEN_FINE_GRAINED_RE):
            for m in pattern.finditer(text):
                matches.append(PiiMatch("credentials", "github_token", m.group(0), m.span()))

        for pattern in (_OPENAI_KEY_LEGACY_RE, _OPENAI_KEY_MODERN_RE):
            for m in pattern.finditer(text):
                matches.append(PiiMatch("credentials", "openai_key", m.group(0), m.span()))

        for m in _BEARER_TOKEN_RE.finditer(text):
            matches.append(PiiMatch("credentials", "bearer_token", m.group(0), m.span()))

        for m in _JWT_RE.finditer(text):
            matches.append(PiiMatch("credentials", "jwt", m.group(0), m.span()))

        for m in _PEM_PRIVATE_KEY_RE.finditer(text):
            matches.append(PiiMatch("credentials", "pem_private_key", m.group(0), m.span()))

        for m in _INTERNAL_IP_RE.finditer(text):
            matches.append(PiiMatch("pii", "internal_ip", m.group(0), m.span()))

        for m in _INTERNAL_HOSTNAME_RE.finditer(text):
            matches.append(PiiMatch("pii", "internal_hostname", m.group(0), m.span()))

        matches.sort(key=lambda pm: (pm.span[0], pm.span[1], pm.type))
        return matches
    except Exception:
        return []


# find_pii is an alias for classify() — CONTEXT.md/RESEARCH.md's canonical
# refs name both spellings; classify() is the primary/documented API.
find_pii = classify


# The redaction-mode pattern list `redact_pii_match()` (scoring/engine.py)
# consumes. Deliberately the RAW shape patterns (not Luhn-gated for credit
# card) — for redaction, over-redaction is the acceptable direction
# (D-30/IN-02 precedent): masking a shape that merely LOOKS like a credit
# card but fails Luhn is harmless, whereas silently leaving a real secret
# unmasked is not. Growing this taxonomy automatically grows redaction
# with no `scoring/engine.py` edit required.
#
# WR-02: structurally-precise, multi-part patterns (`_JWT_RE`,
# `_PEM_PRIVATE_KEY_RE`) run BEFORE broader shape-only patterns
# (`_CREDIT_CARD_RE`, `_IBAN_RE`), mirroring the "precise-first" ordering
# principle already applied at the `api.py` call site between
# `redact_pii_match()` and `redact_credential_match()` (see CR-01,
# 03-REVIEW.md prior cycle). Each redaction pass's output feeds the next
# pass's input (see `redact_pii_match()`), so a broader shape-only pattern
# that happens to match a sub-span of a structurally-precise pattern's
# match (e.g. `_CREDIT_CARD_RE`'s bounded digit run matching an
# all-digit-shaped JWT segment) could otherwise insert a `***REDACTED***`
# marker that breaks the structurally-precise pattern's ability to match
# the remaining segments on its own later pass — the same failure mode
# CR-01 (prior cycle) fixed for `redact_credential_match()`/
# `redact_pii_match()`'s call-site composition order, now closed *within*
# this list's own internal ordering too.
PII_REDACTION_PATTERNS: list[re.Pattern[str]] = [
    _JWT_RE,
    _PEM_PRIVATE_KEY_RE,
    _SSN_RE,
    _EMAIL_RE,
    _CREDIT_CARD_RE,
    _IBAN_RE,
    _PHONE_RE,
    _AWS_ACCESS_KEY_RE,
    _GITHUB_TOKEN_CLASSIC_RE,
    _GITHUB_TOKEN_FINE_GRAINED_RE,
    _OPENAI_KEY_LEGACY_RE,
    _OPENAI_KEY_MODERN_RE,
    _BEARER_TOKEN_RE,
    _INTERNAL_IP_RE,
    _INTERNAL_HOSTNAME_RE,
]
