"""Local prediction API for the web dashboard's Manual Predict page.

    python -m src.api                # http://127.0.0.1:8000

GET  /api/meta     features, units, bounds, training ranges, latest observed 24 h window
POST /api/predict  {"window": [[11 values] x 24 hours]} -> forecast from every trained model

Standard library only (no extra web framework). Binds to localhost.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from src.prediction.manual import ManualPredictor
from src.utils import get_logger, load_config

log = get_logger("api", "api.log")
MAX_BODY = 1_000_000


def make_handler(predictor: ManualPredictor):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/api/meta":
                self._send(200, predictor.meta())
            elif self.path == "/api/health":
                self._send(200, {"ok": True, "models": list(predictor.models)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/predict":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(400, {"error": "Request body missing or too large"})
            try:
                window = json.loads(self.rfile.read(n))["window"]
                result = predictor.predict(window)
            except (json.JSONDecodeError, KeyError, TypeError):
                return self._send(400, {"error": 'Body must be JSON: {"window": [[...], ...]}'})
            except ValueError as e:
                return self._send(422, {"error": str(e)})
            log.info("predict -> %s", {k: v["value"] for k, v in result["predictions"].items()})
            self._send(200, result)

        def log_message(self, fmt, *args):  # route access logs to our logger
            log.debug(fmt, *args)

    return Handler


def main(port: int = 8000) -> int:
    cfg = load_config()
    torch.set_num_threads(cfg["training"].get("num_threads", 4))
    predictor = ManualPredictor(cfg)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(predictor))
    log.info("Prediction API on http://127.0.0.1:%d (models: %s)", port, ", ".join(predictor.models))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(*(int(a) for a in sys.argv[1:])))
