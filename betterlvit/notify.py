# -*- coding: utf-8 -*-
"""Bark push notifications. Hardcoded key matches legacy behavior.

Call sites should gate on config.enable_bark before invoking bark_notify so
training-time push side-effects can be disabled without editing code.
"""
import requests

__all__ = ['bark_notify']

_BARK_KEY = "uAnJRvt7pxbzE9KK6bCVva"


def bark_notify(body: str, title: str = "训练通知") -> None:
    """Best-effort push: 3s timeout, exceptions swallowed.

    Failures are logged to stdout instead of raising — Bark is a non-critical
    side channel and must never block training.
    """
    url = f"https://api.day.app/{_BARK_KEY}/{title}/{body}"
    try:
        requests.get(url, timeout=3)
    except Exception as e:
        print(f"推送失败: {e}")
