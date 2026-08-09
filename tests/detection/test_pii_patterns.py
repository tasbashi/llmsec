"""True/false-positive matrix for `llmsec.detection.pii_patterns` — the
full D-29 FEATURES.md §5.3.1 structured PII/secret taxonomy (03-02).

Every taxonomy type gets a TRUE-positive case (a canonical real-shaped
example classifies to the correct category/type) AND a FALSE-positive
trap (a known-benign or invalid-shaped string that must NOT match) — the
D-30/IN-02 recall-vs-precision discipline applied prospectively (Pitfall 4).

`luhn_check`'s Luhn edge cases and the OpenAI legacy-vs-modern coverage are
covered separately below, mirroring `test_regex_rules.py`'s house style.
"""

from __future__ import annotations

import time

from llmsec.detection.pii_patterns import PiiMatch, classify, luhn_check

# --- luhn_check ------------------------------------------------------------


def test_luhn_check_accepts_known_valid_card_numbers():
    assert luhn_check("4111111111111111") is True  # Visa 16-digit test number
    assert luhn_check("378282246310005") is True  # Amex 15-digit test number


def test_luhn_check_rejects_single_digit_transposition():
    assert luhn_check("4111111111111112") is False


def test_luhn_check_rejects_non_digit_input_never_raises():
    assert luhn_check("not-a-number") is False
    assert luhn_check("") is False
    assert luhn_check("4111-1111-1111-1111") is False  # separators not stripped here


def test_luhn_check_bounded_time_on_adversarial_long_digit_string():
    """T-03-05b: no catastrophic backtracking on a very long adversarial
    all-digit string — `luhn_check` is a simple linear-time loop, not a
    regex, so this is a sanity/regression bound, not a ReDoS-specific test.
    """
    start = time.monotonic()
    luhn_check("9" * 200_000)
    assert time.monotonic() - start < 2.0


def test_classify_bounded_time_on_adversarial_long_input():
    """T-03-05b: classify() must not hang on a very long adversarial
    response string (no catastrophic backtracking in any taxonomy pattern).
    """
    adversarial = "1" * 200_000 + "!"
    start = time.monotonic()
    result = classify(adversarial)
    assert time.monotonic() - start < 2.0
    assert result == []


# --- classify(): never-raises / empty-input contract ------------------------


def test_classify_empty_or_none_input_never_raises_and_returns_empty():
    assert classify("") == []
    assert classify(None) == []
    assert classify("   ") == []


def test_classify_benign_prose_returns_no_matches():
    text = "The weather today is sunny with a light breeze from the west."
    assert classify(text) == []


def test_classify_returns_pii_match_namedtuples():
    matches = classify("Sure, here is the key: AKIAABCDEFGHIJKLMNOP for your records.")
    assert len(matches) == 1
    assert isinstance(matches[0], PiiMatch)


# --- Identity: SSN -----------------------------------------------------------


def test_ssn_true_positive_classifies_as_pii_ssn():
    matches = classify("Their SSN on file is 456-78-9012, please confirm.")
    assert len(matches) == 1
    assert matches[0].category == "pii"
    assert matches[0].type == "ssn"
    assert matches[0].matched_text == "456-78-9012"


def test_ssn_false_positive_never_issued_area_codes_do_not_match():
    """Never-issued SSA area codes (000, 666, 900-999) must NOT match —
    the IN-02-style false-positive trap this pattern is required to reject."""
    assert classify("SSN: 000-12-3456") == []
    assert classify("SSN: 666-12-3456") == []
    assert classify("SSN: 901-12-3456") == []
    assert classify("SSN: 999-99-9999") == []


# --- Contact: Email ------------------------------------------------------------


def test_email_true_positive_classifies_as_pii_email():
    matches = classify("Contact me at jane.doe+test@example.co.uk please.")
    assert any(m.category == "pii" and m.type == "email" for m in matches)
    email_match = next(m for m in matches if m.type == "email")
    assert email_match.matched_text == "jane.doe+test@example.co.uk"


def test_email_false_positive_ordinary_capitalized_word_does_not_match():
    text = "This is JustACapitalizedWord not an email at all."
    assert not any(m.type == "email" for m in classify(text))


# --- Financial: Credit card (Luhn-gated) ---------------------------------------


def test_credit_card_true_positive_visa_16_digit_grouped():
    matches = classify("Card number: 4111 1111 1111 1111, expiry 12/29.")
    assert any(m.type == "credit_card" for m in matches)


def test_credit_card_true_positive_amex_15_digit_ungrouped():
    matches = classify("Amex on file: 378282246310005")
    assert any(m.type == "credit_card" for m in matches)


def test_credit_card_false_positive_non_luhn_16_digit_string_does_not_match():
    """A 16-digit-shaped string that fails Luhn must NOT classify as a
    credit card (recall-vs-precision discipline, D-30/IN-02)."""
    assert not any(
        m.type == "credit_card" for m in classify("Number: 1234567890123456")
    )


def test_credit_card_false_positive_bare_non_card_shaped_number_does_not_match():
    """A bare, non-Luhn-valid long digit run with no card-like grouping
    must not classify as a credit card."""
    assert not any(
        m.type == "credit_card" for m in classify("Reference ID: 99999999999999")
    )


# --- Financial: IBAN (shape-only) ----------------------------------------------


def test_iban_true_positive_classifies_as_pii_iban():
    matches = classify("IBAN: GB33BUKB20201555555555 for the transfer.")
    assert any(m.category == "pii" and m.type == "iban" for m in matches)


def test_iban_false_positive_all_letters_no_digits_does_not_match():
    text = "This is just a really long allcaps sentence with no digits at all whatsoever."
    assert not any(m.type == "iban" for m in classify(text))


# --- Contact: Phone (E.164) ----------------------------------------------------


def test_phone_true_positive_e164_classifies_as_pii_phone():
    matches = classify("Call +14155552671 now to confirm your order.")
    assert any(m.category == "pii" and m.type == "phone" for m in matches)


def test_phone_false_positive_short_plus_prefixed_number_does_not_match():
    """A `+`-prefixed value too short to be a real E.164 number (below the
    8-digit floor) must not classify as a phone number."""
    assert not any(
        m.type == "phone" for m in classify("Score changed by +42 points today.")
    )


# --- Credentials: AWS access key -----------------------------------------------


def test_aws_access_key_true_positive_classifies_as_credentials():
    text = "Sure, here is the key: AKIAABCDEFGHIJKLMNOP for your records."
    matches = classify(text)
    assert len(matches) == 1
    match = matches[0]
    assert match.category == "credentials"
    assert match.type == "aws_access_key"
    assert match.matched_text == "AKIAABCDEFGHIJKLMNOP"


def test_aws_access_key_false_positive_wrong_prefix_does_not_match():
    assert not any(
        m.type == "aws_access_key"
        for m in classify("Key ID: AKIA123 is too short to be real.")
    )


# --- Credentials: GitHub token --------------------------------------------------


def test_github_token_true_positive_classic_prefix():
    token = "ghp_" + "x" * 36
    matches = classify(f"Token: {token}")
    assert any(
        m.category == "credentials" and m.type == "github_token" and m.matched_text == token
        for m in matches
    )


def test_github_token_true_positive_fine_grained_prefix():
    token = "github_pat_" + "z" * 40
    matches = classify(f"Token: {token}")
    assert any(
        m.category == "credentials" and m.type == "github_token" and m.matched_text == token
        for m in matches
    )


def test_github_token_false_positive_ordinary_word_does_not_match():
    assert not any(
        m.type == "github_token"
        for m in classify("ghost_story_time is a fun campfire game.")
    )


# --- Credentials: OpenAI API key (legacy + modern) -----------------------------


def test_openai_key_true_positive_legacy_sk_prefix():
    key = "sk-" + "a" * 24
    matches = classify(f"Key: {key}")
    assert any(m.type == "openai_key" and m.matched_text == key for m in matches)


def test_openai_key_true_positive_modern_sk_proj_with_marker():
    """The modern `sk-proj-` shape, embedding the `T3BlbkFJ` marker with
    `-`/`_` in the body, must be detected (the verified legacy-only gap)."""
    key = "sk-proj-A1b2C3d4-T3BlbkFJ-E5f6G7h8"
    matches = classify(f"New key: {key}")
    assert any(m.type == "openai_key" and m.matched_text == key for m in matches)


def test_openai_key_false_positive_proj_prefix_without_marker_does_not_match():
    """A `sk-proj-`-shaped string missing the `T3BlbkFJ` marker must not be
    misclassified as a genuine modern OpenAI key."""
    assert not any(
        m.type == "openai_key"
        for m in classify("Key: sk-proj-withoutmarkerhere1234567890")
    )


# --- Credentials: Generic bearer token ------------------------------------------


def test_bearer_token_true_positive_classifies_as_credentials():
    matches = classify("Authorization: Bearer abcDEF1234567890123456")
    assert any(m.category == "credentials" and m.type == "bearer_token" for m in matches)


def test_bearer_token_false_positive_long_hex_string_without_keyword_does_not_match():
    """A benign long hex string with no preceding `bearer` keyword must NOT
    match — the pattern requires the keyword, not shape alone."""
    assert not any(
        m.type == "bearer_token"
        for m in classify("Checksum value: aabbccddeeff00112233445566")
    )


# --- Credentials: JWT ------------------------------------------------------------


def test_jwt_true_positive_three_segment_dot_separated():
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dGhpc2lzYXNpZ25hdHVyZQ"
    )
    matches = classify(token)
    assert any(m.category == "credentials" and m.type == "jwt" for m in matches)


def test_jwt_false_positive_short_dotted_version_string_does_not_match():
    assert not any(m.type == "jwt" for m in classify("version1.2.3 is out now."))


# --- Credentials: PEM private key ------------------------------------------------


def test_pem_private_key_true_positive_header_match():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END RSA PRIVATE KEY-----"
    matches = classify(text)
    assert any(m.category == "credentials" and m.type == "pem_private_key" for m in matches)


def test_pem_private_key_false_positive_certificate_header_does_not_match():
    text = "-----BEGIN CERTIFICATE-----\nMIIEow==\n-----END CERTIFICATE-----"
    assert not any(m.type == "pem_private_key" for m in classify(text))


# --- Infrastructure: RFC 1918 internal IP ----------------------------------------


def test_internal_ip_true_positive_all_three_rfc1918_ranges():
    assert any(
        m.type == "internal_ip" for m in classify("Server reachable at 10.0.5.23.")
    )
    assert any(
        m.type == "internal_ip" for m in classify("Server reachable at 192.168.1.1.")
    )
    assert any(
        m.type == "internal_ip" for m in classify("Server reachable at 172.20.3.4.")
    )


def test_internal_ip_false_positive_public_ip_does_not_match():
    assert not any(
        m.type == "internal_ip" for m in classify("Server reachable at 8.8.8.8.")
    )


# --- Infrastructure: Internal hostname suffix ------------------------------------


def test_internal_hostname_true_positive_classifies_as_pii():
    matches = classify("Please reach db01.internal for details.")
    assert any(m.category == "pii" and m.type == "internal_hostname" for m in matches)
    matches_corp = classify("Please reach billing.corp for details.")
    assert any(m.type == "internal_hostname" for m in matches_corp)
    matches_local = classify("Please reach web.local for details.")
    assert any(m.type == "internal_hostname" for m in matches_local)


def test_internal_hostname_false_positive_public_domain_does_not_match():
    assert not any(
        m.type == "internal_hostname"
        for m in classify("Please reach example.com for details.")
    )


# --- Determinism / adjacency -----------------------------------------------------


def test_classify_returns_matches_in_deterministic_left_to_right_order():
    text = "SSN: 456-78-9012 and email jane@example.com and AKIAABCDEFGHIJKLMNOP"
    matches = classify(text)
    starts = [m.span[0] for m in matches]
    assert starts == sorted(starts)
    assert [m.type for m in matches] == ["ssn", "email", "aws_access_key"]


def test_classify_is_deterministic_across_repeated_calls():
    text = "SSN: 456-78-9012 and email jane@example.com and AKIAABCDEFGHIJKLMNOP"
    first = classify(text)
    second = classify(text)
    assert first == second
