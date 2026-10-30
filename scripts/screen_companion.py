"""Serve Bentebot's capture page on localhost and relay frames to its configured server.

Browser screen capture is unavailable from a plain http://LAN-IP page. This stdlib-only
launcher keeps the page on localhost, a browser secure context, without user account tokens.
"""

from __future__ import annotations

import argparse
import ipaddress
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PATHS = {
    "/screen",
    "/screen/",
    "/screen/app.js",
    "/screen/style.css",
    "/screen/pair",
    "/screen/frame",
    "/screen/stop",
    "/screen/status",
}
MAX_BODY = 1_000_000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_target(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Use an HTTP(S) server origin without credentials, query or path")
    if parsed.scheme == "http":
        host = parsed.hostname
        try:
            address = ipaddress.ip_address(host)
            local = address.is_private or address.is_loopback
        except ValueError:
            local = host == "localhost" or "." not in host or host.endswith((".lan", ".local", ".internal"))
        if not local:
            raise ValueError("Use HTTPS for a public server; HTTP is allowed only for local addresses")
    return value.rstrip("/")


def make_handler(target: str):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    class Handler(BaseHTTPRequestHandler):
        timeout = 15

        def do_GET(self):
            self.relay()

        def do_POST(self):
            self.relay()

        def relay(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            if host not in allowed or (origin and origin not in {"http://" + item for item in allowed}):
                self.send_error(403, "Only this localhost capture page may use the companion")
                return
            path = "/screen" if self.path == "/" else self.path
            if path not in PATHS:
                self.send_error(404)
                return
            if (
                self.headers.get("Transfer-Encoding")
                or self.headers.get("Content-Encoding", "identity") != "identity"
            ):
                self.send_error(400, "Unsupported request encoding")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= length <= MAX_BODY:
                self.send_error(413)
                return
            body = self.rfile.read(length) if self.command == "POST" else None
            headers = {
                name: self.headers[name]
                for name in ("Content-Type", "Authorization", "X-Bentebot-Sequence")
                if name in self.headers
            }
            request = urllib.request.Request(target + path, data=body, method=self.command, headers=headers)
            try:
                response = opener.open(request, timeout=15)
            except urllib.error.HTTPError as error:
                response = error
            except (urllib.error.URLError, TimeoutError, OSError):
                self.send_error(
                    502, "Bentebot screen service is unreachable; check target URL and published port"
                )
                return
            with response:
                data = response.read(2_000_001)
                if len(data) > 2_000_000:
                    self.send_error(502, "Unexpected response size")
                    return
                self.send_response(response.status)
                for name in (
                    "Content-Type",
                    "Content-Security-Policy",
                    "X-Content-Type-Options",
                    "Referrer-Policy",
                ):
                    if name in response.headers:
                        self.send_header(name, response.headers[name])
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, format, *args):
            # No request bodies, pairing codes, authorization headers, or screenshots are logged.
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True, help="Bentebot's screen server origin, e.g. http://192.168.50.10:8768"
    )
    parser.add_argument("--port", type=int, default=8888, help="Local capture page port (default 8888)")
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser")
    args = parser.parse_args()
    try:
        target = validate_target(args.target)
    except ValueError as error:
        parser.error(str(error))
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port from 1024 to 65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(target))
    url = f"http://localhost:{args.port}/screen"
    print(f"Open {url}")
    print("Use /screen start in Discord to get a one-time code. Ctrl+C stops this local relay.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
