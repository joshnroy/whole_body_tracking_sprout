"""Manual browser regression: run against a running browser simulator.

uv run --no-project --with playwright --with msgspec python tests/browser_controls.py
Set WBT_BROWSER_URL and, if needed, WBT_CHROMIUM_EXECUTABLE.
"""

import json
import os

import msgspec
from playwright.sync_api import expect, sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=os.environ.get("WBT_CHROMIUM_EXECUTABLE"),
        headless=True,
        args=["--no-sandbox", "--enable-unsafe-swiftshader"],
    )
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors, cameras = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def frame(payload):
        try:
            value = msgspec.msgpack.decode(payload) if isinstance(payload, bytes) else json.loads(payload)
            if isinstance(value, dict) and value.get("type") == "ViewerCameraMessage":
                cameras.append(value)
        except Exception:
            pass

    page.on("websocket", lambda ws: ws.on("framesent", frame))
    page.goto(os.environ.get("WBT_BROWSER_URL", "http://localhost:8080"), wait_until="networkidle")
    body = page.locator("body")
    expect(page.get_by_role("button", name="Play", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_text("Connecting keyboard controls…")).not_to_be_visible(timeout=15000)
    canvas = page.locator("canvas[data-engine]").first
    canvas.click(position={"x": 400, "y": 400})
    page.keyboard.press("r")
    expect(body).to_contain_text("Paused · 0.00 s")
    page.wait_for_timeout(1000)
    before = cameras[-1] if cameras else None
    page.keyboard.down("w")
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.wait_for_timeout(800)
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.keyboard.down("a")
    expect(body).to_contain_text("Turn +0.50 rad/s")
    page.wait_for_timeout(400)
    page.keyboard.up("w")
    expect(body).to_contain_text("Forward +0.00 m/s")
    expect(body).to_contain_text("Turn +0.50 rad/s")
    page.keyboard.up("a")
    expect(body).to_contain_text("Turn +0.00 rad/s")
    for key, text in [
        ("s", "Forward -0.20 m/s"),
        ("d", "Turn -0.50 rad/s"),
        ("ArrowUp", "Forward +0.40 m/s"),
        ("ArrowLeft", "Turn +0.50 rad/s"),
    ]:
        page.keyboard.down(key)
        expect(body).to_contain_text(text)
        page.wait_for_timeout(300)
        page.keyboard.up(key)
        expect(body).to_contain_text("Forward +0.00 m/s")
        expect(body).to_contain_text("Turn +0.00 rad/s")
    page.wait_for_timeout(400)
    assert before is not None, "No camera messages captured"
    after = cameras[-1]
    for prop in ["position", "look_at", "wxyz"]:
        assert before[prop] == after[prop], (prop, before[prop], after[prop])
    print("PASS: hold/release, combined controls, arrows, camera unaffected by keyboard")
    # Mouse orbit should still change the camera.
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + 400, box["y"] + 400)
    page.mouse.down()
    page.mouse.move(box["x"] + 560, box["y"] + 450, steps=15)
    page.mouse.up()
    page.wait_for_timeout(700)
    assert cameras[-1]["wxyz"] != before["wxyz"]
    print("PASS: mouse orbit remains enabled")
    # Opposite keys cancel until one is released.
    page.keyboard.down("w")
    page.keyboard.down("s")
    expect(body).to_contain_text("Forward +0.00 m/s")
    page.keyboard.up("s")
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.keyboard.up("w")
    expect(body).to_contain_text("Forward +0.00 m/s")
    # Simulate lost drive packets while a key is still held; the server must expire it.
    page.keyboard.down("w")
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.evaluate("""() => {
        window.savedWebSocketSend = WebSocket.prototype.send;
        WebSocket.prototype.send = function(data) {
            if (typeof data === 'string' && JSON.parse(data).action === 'drive') return;
            return window.savedWebSocketSend.call(this, data);
        };
    }""")
    page.wait_for_timeout(900)
    expect(body).to_contain_text("Forward +0.00 m/s")
    page.keyboard.up("w")
    page.evaluate("() => { WebSocket.prototype.send = window.savedWebSocketSend; }")
    print("PASS: opposite keys and server timeout after lost keyboard packets")
    page.keyboard.down("w")
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    expect(body).to_contain_text("Forward +0.00 m/s")
    page.keyboard.up("w")
    page.keyboard.down("w")
    page.locator("input[inputmode=decimal]").nth(0).click()
    expect(body).to_contain_text("Forward +0.00 m/s")
    page.keyboard.up("w")
    page.keyboard.press("p")
    expect(page.get_by_role("button", name="Play", exact=True)).to_be_visible()
    canvas.click(position={"x": 400, "y": 400})
    page.keyboard.press("p")
    expect(page.get_by_role("button", name="Pause", exact=True)).to_be_visible()
    page.keyboard.down("w")
    page.wait_for_timeout(1200)
    expect(body).to_contain_text("Running")
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.keyboard.press(" ")
    expect(body).to_contain_text("Forward +0.00 m/s")
    page.wait_for_timeout(400)
    expect(body).to_contain_text("Forward +0.00 m/s")
    page.keyboard.up("w")
    page.keyboard.press("r")
    expect(body).to_contain_text("Paused · 0.00 s")
    # A reload replaces the bridge and its listeners; a key must still work once.
    page.reload(wait_until="networkidle")
    expect(page.get_by_role("button", name="Play", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_text("Connecting keyboard controls…")).not_to_be_visible(timeout=15000)
    page.locator("canvas[data-engine]").first.click(position={"x": 400, "y": 400})
    page.keyboard.down("w")
    expect(body).to_contain_text("Forward +0.40 m/s")
    page.keyboard.up("w")
    expect(body).to_contain_text("Forward +0.00 m/s")
    assert not errors, errors
    print("PASS: blur, input focus, live walking, Space, reset; no browser errors")
    browser.close()
