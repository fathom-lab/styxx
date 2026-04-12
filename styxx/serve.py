# -*- coding: utf-8 -*-
"""
styxx.serve - local live dashboard for the agent personality card.

0.2.0+. The missing distribution surface: the agent card used to
live on your desktop as a PNG until you manually shared it. Now it
has a home. `styxx agent-card --serve` spins up a local http server
that renders your personality card and auto-refreshes as your
audit log grows.

Usage
-----

    $ styxx agent-card --serve
    [styxx serve] running at http://localhost:9797
    [styxx serve] card re-renders every 30s
    [styxx serve] press ctrl+c to stop

Open your browser to localhost:9797 and leave it open in a side
panel. As your agent accumulates observations, the card updates
automatically. The page meta-refreshes every N seconds; the
background renderer rewrites the PNG at the same cadence.

Design
------

  * http.server stdlib, no Flask. one request handler, two routes:
      GET /          -> HTML dashboard page (meta-refresh on)
      GET /card.png  -> latest rendered PNG

  * Background renderer thread rewrites the PNG every
    `refresh_seconds` even if no request comes in. this is the
    "live" part — you don't have to refresh the page to see new
    data, the page auto-refreshes on a timer AND the renderer
    is already writing the new data.

  * Fail-open: if the audit log is empty, renders a "waiting for
    data" placeholder card instead of crashing.

  * Foreground blocks until ctrl+c. Cleanup is automatic.

No dependencies beyond stdlib + Pillow (for the rendering itself —
serve.py doesn't add any new deps).
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Optional


# The HTML page template. Styled to match the landing page aesthetic
# so the serve view feels like a direct extension of the brand.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="{refresh_seconds}">
<title>styxx - live agent card - {agent_name}</title>
<style>
body {{
  margin: 0;
  padding: 0;
  min-height: 100vh;
  background: #080306;
  color: #b0a8ac;
  font-family: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
}}
header {{
  width: 100%;
  padding: 1rem 2rem;
  background: #120810;
  border-bottom: 1px solid #2a1218;
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
header .brand {{ color: #ff0033; font-weight: 700; }}
header .brand span {{ color: #ff2a8a; }}
header .meta {{ color: #6a5a60; }}
header .live {{
  color: #ff0033;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
}}
header .live::before {{
  content: '';
  width: 6px;
  height: 6px;
  background: #ff0033;
  border-radius: 50%;
  box-shadow: 0 0 6px #ff0033;
  animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.3; }}
}}
main {{
  padding: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 100%;
}}
.card-frame {{
  border: 1px solid #3b1a22;
  padding: 1rem;
  background: #050103;
  box-shadow: 0 0 40px rgba(255, 0, 51, 0.12),
              inset 0 0 80px rgba(255, 0, 51, 0.04);
  position: relative;
}}
.card-frame::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #ff0033, #ff2a8a, #ff0033, transparent);
  box-shadow: 0 0 8px #ff0033;
}}
.card-frame img {{
  display: block;
  max-width: 100%;
  height: auto;
  image-rendering: pixelated;
}}
.tagline {{
  margin-top: 1rem;
  color: #6a5a60;
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  text-align: center;
}}
footer {{
  margin-top: 2rem;
  padding: 1rem 2rem;
  color: #3a2c30;
  font-size: 0.55rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}
footer a {{ color: #6a5a60; text-decoration: none; }}
footer a:hover {{ color: #ff0033; }}
</style>
</head>
<body>
<header>
  <div class="brand"><span>STYXX</span> - LIVE AGENT CARD</div>
  <div class="meta">{agent_name} &nbsp;&middot;&nbsp; refreshing every {refresh_seconds}s</div>
  <div class="live">LIVE</div>
</header>
<main>
  <div class="card-frame">
    <img src="/card.png?ts={cache_bust}" alt="styxx agent card">
  </div>
  <div class="tagline">&middot; &middot; &middot; nothing crosses unseen &middot; &middot; &middot;</div>
</main>
<footer>
  a fathom lab product &middot;
  <a href="https://fathom.darkflobi.com/styxx">fathom.darkflobi.com/styxx</a> &middot;
  pip install styxx
</footer>
</body>
</html>
"""


def _make_handler(serve_dir: Path, agent_name: str, refresh_seconds: int):
    """Build a request handler closure that serves / and /card.png."""

    class _StyxxHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args):
            # Suppress the default access log spam; we print our own
            # messages in the render loop.
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._serve_html()
            elif path == "/card.png":
                self._serve_png()
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found")

        def _serve_html(self) -> None:
            html = _HTML_TEMPLATE.format(
                agent_name=agent_name,
                refresh_seconds=refresh_seconds,
                cache_bust=int(time.time()),
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_png(self) -> None:
            png_path = serve_dir / "card.png"
            if not png_path.exists():
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"card not rendered yet")
                return
            try:
                body = png_path.read_bytes()
            except OSError:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"could not read card file")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return _StyxxHandler


def _render_loop(
    serve_dir: Path,
    agent_name: str,
    days: float,
    refresh_seconds: int,
    stop_flag: threading.Event,
) -> None:
    """Background thread: re-render the card every refresh_seconds."""
    from .card_image import render_agent_card

    png_path = serve_dir / "card.png"

    while not stop_flag.is_set():
        try:
            render_agent_card(
                out_path=png_path,
                agent_name=agent_name,
                days=days,
            )
        except RuntimeError:
            # No audit data yet — the page will show a 503 on /card.png
            # until observations accumulate. That's correct behavior.
            pass
        except Exception as e:
            # Any other error: log it to stdout but keep looping.
            print(f"  [styxx serve] render error: {type(e).__name__}: {e}")

        # Sleep in small increments so ctrl+c is responsive
        for _ in range(refresh_seconds * 2):
            if stop_flag.is_set():
                return
            time.sleep(0.5)


def run_serve(
    *,
    port: int = 9797,
    agent_name: str = "styxx agent",
    days: float = 7.0,
    refresh_seconds: int = 30,
    open_browser: bool = True,
) -> int:
    """Run the local styxx live dashboard.

    Blocks the foreground thread until ctrl+c. Returns 0 on clean
    shutdown, non-zero on bind failure.
    """
    serve_dir = Path.home() / ".styxx" / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    # Start the background render loop
    stop_flag = threading.Event()
    render_thread = threading.Thread(
        target=_render_loop,
        args=(serve_dir, agent_name, days, refresh_seconds, stop_flag),
        daemon=True,
    )
    render_thread.start()

    # Render once synchronously so the first page load has data
    try:
        from .card_image import render_agent_card
        render_agent_card(
            out_path=serve_dir / "card.png",
            agent_name=agent_name,
            days=days,
        )
    except RuntimeError:
        # no audit data yet — placeholder will serve 503 until rendered
        pass
    except Exception as e:
        print(f"  [styxx serve] initial render failed: {e}")

    # Try to bind the http server
    handler = _make_handler(serve_dir, agent_name, refresh_seconds)
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as e:
        print(f"  [styxx serve] could not bind port {port}: {e}")
        stop_flag.set()
        return 1

    url = f"http://localhost:{port}"
    print()
    print(f"  [styxx serve] running at {url}")
    print(f"  [styxx serve] agent: {agent_name}")
    print(f"  [styxx serve] re-renders every {refresh_seconds}s")
    print(f"  [styxx serve] press ctrl+c to stop")
    print()

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url, new=2)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
        print("  [styxx serve] shutting down...")
    finally:
        stop_flag.set()
        httpd.server_close()

    return 0
