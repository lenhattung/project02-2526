from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from playwright.sync_api import sync_playwright


def has_facebook_session(cookies: list[dict]) -> bool:
    names = {cookie.get("name") for cookie in cookies if "facebook.com" in cookie.get("domain", "")}
    return "c_user" in names and "xs" in names


def main() -> int:
    if len(sys.argv) < 2:
        print("COOKIE_ERROR:Missing cookies.json path")
        return 1

    output_path = Path(sys.argv[1]).resolve()
    timeout_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("COOKIE_INFO:Opening Facebook browser for cookie capture.")
    print("COOKIE_INFO:Please sign in to Facebook in the opened browser if needed.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.facebook.com/", wait_until="domcontentloaded")
        deadline = time.time() + timeout_seconds
        status_tick = 0

        while time.time() < deadline:
            page.wait_for_timeout(2000)
            all_cookies = context.cookies()
            if has_facebook_session(all_cookies):
                output_path.write_text(json.dumps(all_cookies, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"COOKIE_CAPTURED:{output_path}")
                browser.close()
                return 0

            status_tick += 1
            if status_tick % 5 == 0:
                if "login" in page.url.lower():
                    print("COOKIE_INFO:Waiting for Facebook sign-in...")
                else:
                    print("COOKIE_INFO:Waiting for Facebook session cookies...")

        print("COOKIE_TIMEOUT:Timed out before session cookies were captured.")
        browser.close()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
