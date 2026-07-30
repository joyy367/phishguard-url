"""
test_features.py
----------------
Tests for URL validation, normalisation and 22-feature extraction.
"""

import math
import pytest

from app.features import (
    FEATURE_NAMES,
    _shannon_entropy,
    extract_url_features,
    feature_vector,
    normalize_url,
)


# ---------------------------------------------------------------------------
# FEATURE_NAMES contract
# ---------------------------------------------------------------------------

def test_feature_names_count():
    assert len(FEATURE_NAMES) == 22


def test_feature_names_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_feature_names_exact_order():
    expected = [
        "url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt",
        "letter_cnt", "digit_cnt", "special_cnt", "eq_cnt", "qm_cnt",
        "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt", "letter_ratio",
        "digit_ratio", "spec_ratio", "is_https", "slash_cnt", "entropy",
        "path_len", "query_len",
    ]
    assert FEATURE_NAMES == expected


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:

    def test_https_passthrough(self):
        url = normalize_url("https://example.com/path")
        assert url.startswith("https://")

    def test_missing_scheme_becomes_https(self):
        url = normalize_url("example.com")
        assert url.startswith("https://")

    def test_http_accepted(self):
        url = normalize_url("http://example.com")
        assert url.startswith("http://")

    def test_fragment_preserved_for_feature_contract(self):
        url = normalize_url("https://example.com/page#section")
        assert url.endswith("#section")

    def test_whitespace_stripped(self):
        url = normalize_url("  https://example.com  ")
        assert url == "https://example.com"

    @pytest.mark.parametrize("bad_url, expected_fragment", [
        ("ftp://example.com", "Unsupported scheme"),
        ("http://127.0.0.1/login", "private"),
        ("http://192.168.1.1/", "private"),
        ("http://10.0.0.1/", "private"),
        ("", "empty"),
        ("hello", "public suffix"),
        ("https://example.com:99999", "invalid port"),
    ])
    def test_invalid_urls_raise(self, bad_url, expected_fragment):
        with pytest.raises(ValueError, match=expected_fragment):
            normalize_url(bad_url)


# ---------------------------------------------------------------------------
# extract_url_features
# ---------------------------------------------------------------------------

class TestExtractUrlFeatures:

    def _get(self, url: str) -> dict:
        normed = normalize_url(url)
        return extract_url_features(normed)

    def test_returns_all_22_keys(self):
        feats = self._get("https://example.com")
        assert set(feats.keys()) == set(FEATURE_NAMES)

    def test_all_values_finite(self):
        feats = self._get("https://example.com/path?q=1&a=2#frag")
        for name, val in feats.items():
            assert math.isfinite(val), f"{name} is not finite: {val}"

    def test_is_https_flag_true(self):
        feats = self._get("https://example.com")
        assert feats["is_https"] == 1

    def test_is_https_flag_false(self):
        feats = self._get("http://example.com")
        assert feats["is_https"] == 0

    def test_is_ip_false_for_domain(self):
        feats = self._get("https://example.com")
        assert feats["is_ip"] == 0

    def test_url_len_correct(self):
        url = "https://example.com"
        normed = normalize_url(url)
        feats = extract_url_features(normed)
        assert feats["url_len"] == len(normed)

    def test_eq_cnt(self):
        feats = self._get("https://example.com/path?a=1&b=2")
        assert feats["eq_cnt"] == 2

    def test_amp_cnt(self):
        feats = self._get("https://example.com/path?a=1&b=2&c=3")
        assert feats["amp_cnt"] == 2

    def test_qm_cnt(self):
        feats = self._get("https://example.com/search?q=hello")
        assert feats["qm_cnt"] == 1

    def test_dot_cnt(self):
        feats = self._get("https://www.sub.example.com/page")
        assert feats["dot_cnt"] >= 3

    def test_ratios_sum_to_one(self):
        feats = self._get("https://example.com/path")
        total = feats["letter_ratio"] + feats["digit_ratio"] + feats["spec_ratio"]
        assert abs(total - 1.0) < 1e-6

    def test_ratios_in_0_1(self):
        feats = self._get("https://example.com/some/path?q=123")
        for ratio_name in ["letter_ratio", "digit_ratio", "spec_ratio"]:
            assert 0.0 <= feats[ratio_name] <= 1.0

    def test_query_len_zero_no_query(self):
        feats = self._get("https://example.com/path")
        assert feats["query_len"] == 0

    def test_query_len_positive_with_query(self):
        feats = self._get("https://example.com/path?q=hello&page=2")
        assert feats["query_len"] > 0

    def test_fragment_contributes_to_lexical_counts(self):
        without_fragment = self._get("https://example.com/path")
        with_fragment = self._get("https://example.com/path#account-login")
        assert with_fragment["url_len"] > without_fragment["url_len"]
        assert with_fragment["letter_cnt"] > without_fragment["letter_cnt"]

    def test_entropy_positive(self):
        feats = self._get("https://example.com")
        assert feats["entropy"] > 0

    def test_subdom_cnt_no_subdomain(self):
        feats = self._get("https://example.com")
        assert feats["subdom_cnt"] == 0

    def test_subdom_cnt_with_www(self):
        feats = self._get("https://www.example.com")
        assert feats["subdom_cnt"] == 1

    def test_subdom_cnt_multi(self):
        feats = self._get("https://a.b.example.com")
        assert feats["subdom_cnt"] == 2

    def test_dash_cnt(self):
        feats = self._get("https://my-phishing-site.example.com")
        assert feats["dash_cnt"] >= 2


# ---------------------------------------------------------------------------
# feature_vector
# ---------------------------------------------------------------------------

class TestFeatureVector:

    def test_returns_22_values(self):
        vec = feature_vector("https://example.com")
        assert len(vec) == 22

    def test_returns_list(self):
        vec = feature_vector("https://example.com")
        assert isinstance(vec, list)

    def test_order_matches_feature_names(self):
        url = "https://example.com/path?q=1"
        vec = feature_vector(url)
        normed = normalize_url(url)
        feat_dict = extract_url_features(normed)
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == feat_dict[name]

    def test_raises_on_invalid_url(self):
        with pytest.raises(ValueError):
            feature_vector("not-a-valid-hostname")


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

class TestEntropy:

    def test_empty_string_zero(self):
        assert _shannon_entropy("") == 0.0

    def test_uniform_string_max(self):
        # All same char → entropy = 0
        assert _shannon_entropy("aaaa") == 0.0

    def test_entropy_positive_for_varied(self):
        assert _shannon_entropy("abcdef") > 0

    def test_entropy_increases_with_variety(self):
        low = _shannon_entropy("aaaaab")
        high = _shannon_entropy("abcdef")
        assert high > low
