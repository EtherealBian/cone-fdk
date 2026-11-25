"""Optional browser QA for the published measured atlas (Playwright + local Chromium)."""

import argparse
import base64
import io
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


def verify(file, browser_path, screenshot):
    from playwright.sync_api import sync_playwright

    # Decode the reference bytes separately; never use the page's JS to predict its own pixels.
    payload = json.loads(
        re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            file.read_text(encoding="utf8"),
            re.S,
        )[1]
    )
    d, h, w = payload["shape_rha"]
    errors = []
    network = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=str(browser_path), args=["--disable-gpu"], headless=True
        )
        tab = browser.new_page(
            viewport={"width": 1120, "height": 820}, offline=True, device_scale_factor=1
        )
        tab.on("pageerror", lambda e: errors.append(str(e)))
        tab.on(
            "request",
            lambda r: network.append(r.url) if r.url.startswith(("http:", "https:")) else None,
        )
        tab.goto(file.resolve().as_uri())
        tab.wait_for_function("images.every(im=>im.complete && im.naturalWidth===W)")
        cv = tab.locator("#sheet")

        def slider(name, n):
            tab.locator("#" + name).fill(str(n))

        def tap(i, j):
            b = cv.bounding_box()
            tab.mouse.click(
                round(b["x"] + (i + 0.5) * b["width"] / w),
                round(b["y"] + (j + 0.5) * b["height"] / h),
            )

        # Verify visible pixels even after a quarter turn and at both radius limits.
        for k in (0, d // 2, d - 1):
            slider("shell", k)
            ref = np.array(Image.open(io.BytesIO(base64.b64decode(payload["frames"][k]))))
            for off in (0, w // 4):
                slider("seam", off)
                got = tab.evaluate(
                    """() => Array.from(document.querySelector('canvas').getContext('2d').getImageData(0,5,document.querySelector('canvas').width,1).data.filter((_,i)=>i%4===0))"""
                )
                np.testing.assert_array_equal(got, np.roll(ref[5], -off))
        slider("shell", d // 2)
        slider("seam", 0)
        r = payload["first_radius_mm"] + (d // 2) * payload["radial_step_mm"]
        pitch = payload["axial_pitch_mm"]
        tap(w - 1, h // 2 - 1)
        tap(0, h // 2)
        expected = math.hypot(r * math.tau / w, pitch)
        assert tab.locator("#dist").inner_text() == f"{expected:.3f} mm", (
            tab.locator("#dist").inner_text(),
            tab.evaluate(
                "[pins, i0, j0, rad(), +seam.value, cv.getBoundingClientRect().toJSON()]"
            ),
        )
        # Source fixture has axis +Z and seam +X. Read what is actually displayed.
        assert tab.locator("#xyz").inner_text() == f"XYZ ({r:.2f}, 0.00, {pitch / 2:.2f}) mm"
        slider("seam", w // 4)
        assert tab.locator("#dist").inner_text() == f"{expected:.3f} mm", (
            tab.locator("#dist").inner_text(),
            tab.evaluate(
                "[pins, i0, j0, rad(), +seam.value, cv.getBoundingClientRect().toJSON()]"
            ),
        )
        slider("shell", d // 2 + 1)
        assert tab.locator("#dist").inner_text() == "— mm"
        slider("seam", 0)
        tap(0, 0)
        cv.focus()
        cv.press("ArrowLeft")
        cv.press("Enter")
        assert (
            tab.locator("#dist").inner_text()
            == f"{(r + payload['radial_step_mm']) * math.tau / w:.3f} mm"
        )
        cv.press("Escape")
        assert tab.locator("#dist").inner_text() == "— mm"
        tab.set_viewport_size({"width": 390, "height": 844})
        assert tab.evaluate("document.documentElement.scrollWidth<=innerWidth")
        tab.set_viewport_size({"width": 1120, "height": 820})
        slider("shell", d // 2)
        slider("seam", w // 8)
        tap(w - 22, 135)
        tap(18, 118)
        tab.mouse.move(0, 0)
        tab.locator("main").screenshot(path=str(screenshot), animations="disabled")
        assert not errors and not network, (errors, network)
        print(
            f"Chromium {browser.version}; 6 decoded scanlines; wrapped ruler {expected:.3f} mm; coordinates, keyboard, sliders, mobile width, offline: PASS."
        )
        browser.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file", type=Path)
    p.add_argument("--browser", type=Path, required=True)
    p.add_argument("--screenshot", type=Path, required=True)
    args = p.parse_args()
    verify(args.file, args.browser, args.screenshot)
