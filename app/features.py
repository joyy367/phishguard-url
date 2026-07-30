"""
features.py
-----------
URL validation, normalisation and deterministic 22-feature extraction.
The FEATURE_NAMES list defines the fixed contract shared between training
and inference.  feature_vector() always returns values in this exact order.
"""

import math
import re
import ipaddress
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit, urlparse

import tldextract

# Use the bundled Public Suffix List snapshot. This keeps feature extraction
# deterministic and avoids an outbound network request on first use.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# ---------------------------------------------------------------------------
# Fixed feature contract – must match the dataset column order exactly
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "url_len",
    "dom_len",
    "is_ip",
    "tld_len",
    "subdom_cnt",
    "letter_cnt",
    "digit_cnt",
    "special_cnt",
    "eq_cnt",
    "qm_cnt",
    "amp_cnt",
    "dot_cnt",
    "dash_cnt",
    "under_cnt",
    "letter_ratio",
    "digit_ratio",
    "spec_ratio",
    "is_https",
    "slash_cnt",
    "entropy",
    "path_len",
    "query_len",
]

# Allowed schemes
_ALLOWED_SCHEMES = {"http", "https"}

# Private / loopback ranges that should be rejected
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_private_ip(hostname: str) -> bool:
    """Return True if *hostname* resolves to a private/local IP address."""
    try:
        addr = ipaddress.ip_address(hostname)
        return any(addr in net for net in _PRIVATE_NETWORKS) or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _is_ip_host(hostname: str) -> bool:
    """Return True if hostname is a raw IP address (v4 or v6)."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=4096)
def _domain_parts_for_hostname(hostname: str) -> tuple[str, str, str]:
    """Cached public-suffix decomposition for a hostname."""
    if _is_ip_host(hostname):
        return hostname, "", ""
    ext = _TLD_EXTRACTOR(hostname)
    registrable_domain = ext.domain + ("." + ext.suffix if ext.suffix else "")
    return registrable_domain, ext.suffix, ext.subdomain


def get_domain_parts(url: str) -> tuple[str, str, str]:
    """Return (registrable_domain, public_suffix, subdomain) for *url*."""
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    return _domain_parts_for_hostname(hostname)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_url(raw: str) -> str:
    """
    Validate and normalise a URL string.

    Rules
    -----
    * Strip leading/trailing whitespace.
    * If no scheme present, prepend 'https://'.
    * Accept only http:// and https:// schemes.
    * Require a non-empty, non-private hostname.
    * Require a recognised public suffix (tldextract).
    * Preserve the fragment because the training dataset's lexical features
      were calculated from the complete URL string.

    Returns the normalised URL string.
    Raises ValueError with a user-friendly message on any violation.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("URL must not be empty.")

    # Prepend scheme if missing so that urlsplit works properly
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw):
        raw = "https://" + raw

    parsed = urlsplit(raw)

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported scheme '{scheme}'. Only http and https are accepted."
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("URL does not contain a valid hostname.")

    # Accessing .port validates malformed/out-of-range ports.
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"URL contains an invalid port: {exc}") from exc

    # Reject private / loopback addresses
    if _is_private_ip(hostname):
        raise ValueError(
            f"'{hostname}' is a private or loopback address. "
            "Private addresses are outside the public-URL model scope."
        )

    # Require a recognised public suffix (unless it's a raw IP)
    if not _is_ip_host(hostname):
        _, suffix, _ = get_domain_parts(raw)
        if not suffix:
            raise ValueError(
                f"'{hostname}' does not contain a recognised public suffix. "
                "Please enter a valid public URL."
            )

    # Keep every lexical component used by the training feature contract.
    normalised = urlunsplit(
        (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return normalised


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_url_features(url: str) -> dict:
    """
    Extract the 22 numerical features from a *normalised* URL string.

    Parameters
    ----------
    url : str
        A URL that has already been processed by normalize_url().

    Returns
    -------
    dict  {feature_name: numeric_value}  in FEATURE_NAMES order.
    """
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""

    # Public-suffix-aware domain decomposition
    registrable_domain, suffix, subdomain = get_domain_parts(url)

    # --- Basic length features ---
    url_len = len(url)
    dom_len = len(registrable_domain)
    tld_len = len(suffix) if suffix else 0
    subdom_cnt = len([s for s in subdomain.split(".") if s]) if subdomain else 0

    # --- IP flag ---
    is_ip = 1 if _is_ip_host(hostname) else 0

    # --- Character composition ---
    letter_cnt = sum(1 for c in url if c.isalpha())
    digit_cnt = sum(1 for c in url if c.isdigit())
    special_cnt = sum(1 for c in url if not c.isalpha() and not c.isdigit())

    # --- Punctuation counts ---
    eq_cnt = url.count("=")
    qm_cnt = url.count("?")
    amp_cnt = url.count("&")
    dot_cnt = url.count(".")
    dash_cnt = url.count("-")
    under_cnt = url.count("_")
    slash_cnt = url.count("/")

    # --- Ratios ---
    letter_ratio = letter_cnt / url_len if url_len > 0 else 0.0
    digit_ratio = digit_cnt / url_len if url_len > 0 else 0.0
    spec_ratio = special_cnt / url_len if url_len > 0 else 0.0

    # --- Protocol flag ---
    is_https = 1 if parsed.scheme.lower() == "https" else 0

    # --- Entropy ---
    entropy = _shannon_entropy(url)

    # --- Path and query lengths ---
    path_len = len(parsed.path) if parsed.path else 0
    query_len = len(parsed.query) if parsed.query else 0

    features = {
        "url_len": url_len,
        "dom_len": dom_len,
        "is_ip": is_ip,
        "tld_len": tld_len,
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": eq_cnt,
        "qm_cnt": qm_cnt,
        "amp_cnt": amp_cnt,
        "dot_cnt": dot_cnt,
        "dash_cnt": dash_cnt,
        "under_cnt": under_cnt,
        "letter_ratio": letter_ratio,
        "digit_ratio": digit_ratio,
        "spec_ratio": spec_ratio,
        "is_https": is_https,
        "slash_cnt": slash_cnt,
        "entropy": entropy,
        "path_len": path_len,
        "query_len": query_len,
    }

    # Validate that all values are finite numbers
    for name, val in features.items():
        if not math.isfinite(val):
            features[name] = 0.0

    return features


def feature_vector(url: str) -> list:
    """
    Return the feature values as an ordered list matching FEATURE_NAMES.
    Raises ValueError if the URL is invalid.
    """
    normalised = normalize_url(url)
    feat_dict = extract_url_features(normalised)
    return [feat_dict[name] for name in FEATURE_NAMES]
