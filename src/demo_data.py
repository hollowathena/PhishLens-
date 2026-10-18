"""Safe, fabricated data for demonstrating the pipeline and UI only."""
from __future__ import annotations

import pandas as pd


def demo_urls() -> pd.DataFrame:
    legitimate_domains = [
        "library.example.edu", "portal.example.edu", "courses.example.edu", "news.example.org",
        "museum.example.org", "weather.example.net", "docs.example.net", "shop.example.com",
        "support.example.com", "careers.example.com", "research.example.edu", "events.example.edu",
        "health.example.org", "books.example.net", "maps.example.com", "community.example.org",
    ]
    paths = ["/", "/about", "/help/contact", "/events", "/resources/guide", "/account/settings"]
    legitimate = [f"https://{domain}{paths[i % len(paths)]}" for i, domain in enumerate(legitimate_domains * 3)]
    suspicious = [
        "http://198.51.100.24/verify/account/login", "http://secure-login.example.test/confirm-account",
        "https://account-update.security.example.test/verify", "http://portal.example.edu.login-check.example.test/signin",
        "http://verify-wallet.example.test/secure/login", "https://delivery-update.example.test/confirm/invoice",
        "http://login.example.test@198.51.100.81/secure", "https://secure-account-check.example.test/update/password",
        "http://bank-alert.example.test/verify/account", "https://signin-check.example.test/login/confirm",
        "http://192.0.2.55/account/update", "https://billing-notice.example.test/verify/wallet",
        "http://account-security.example.test/login", "https://portal.verify.example.test/confirm/account",
        "http://safe-login-check.example.test/secure/verify", "https://invoice-update.example.test/confirm/login",
    ] * 3
    return pd.DataFrame({"url": legitimate + suspicious, "label": [0] * len(legitimate) + [1] * len(suspicious)})


def demo_survey() -> pd.DataFrame:
    rows = [
        ("https://library.example.edu/", 0, "legitimate", 5),
        ("http://198.51.100.24/verify/account/login", 1, "suspicious", 5),
        ("https://portal.example.edu.login-check.example.test/signin", 1, "legitimate", 4),
        ("https://museum.example.org/about", 0, "suspicious", 2),
        ("https://secure-account-check.example.test/update/password", 1, "suspicious", 4),
        ("https://events.example.edu/events", 0, "legitimate", 4),
    ]
    return pd.DataFrame(rows, columns=["url", "true_label", "decision", "confidence"])
