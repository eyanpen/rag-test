"""Adaptive concurrency controller with httpx monkey-patching."""
import asyncio
import json
import logging
import threading
import time
from typing import Dict, List

import httpx

log = logging.getLogger(__name__)


class AdaptiveConcurrencyController:
    def __init__(self, init=10, min_val=2, max_val=50):
        self.current = init
        self.min_val = min_val
        self.max_val = max_val
        self._success_streak = 0
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(init)
        self._stats: List[float] = []
        self._stats_lock = threading.Lock()
        self._req_start_times: Dict[int, float] = {}

    def adjust(self, is_error: bool):
        with self._lock:
            old = self.current
            if is_error:
                self._success_streak = 0
                self.current = max(self.min_val, self.current - 1)
            else:
                self._success_streak += 1
                if self._success_streak >= 5:
                    self._success_streak = 0
                    self.current = min(self.max_val, self.current + 1)
            if self.current != old:
                log.info(f"[ADAPTIVE] concurrency {old} → {self.current}")
                self._semaphore = asyncio.Semaphore(self.current)

    def install_hooks(self):
        ctrl = self
        _orig_init = httpx.AsyncClient.__init__
        _orig_send = httpx.AsyncClient.send

        async def _on_request(request):
            ctrl._req_start_times[id(request)] = time.time()
            # Strip encoding_format=None from embedding requests
            if request.method == "POST" and request.content:
                try:
                    body = json.loads(request.content)
                    if "encoding_format" in body and body["encoding_format"] is None:
                        del body["encoding_format"]
                        request._content = json.dumps(body).encode("utf-8")
                        request.headers["content-length"] = str(len(request._content))
                except Exception:
                    pass

        async def _on_response(response):
            start = ctrl._req_start_times.pop(id(response.request), None)
            if start:
                dur = time.time() - start
                log.debug(
                    f"[TIMING] {response.request.method} {response.request.url} "
                    f"→ {response.status_code} in {dur:.2f}s"
                )
                with ctrl._stats_lock:
                    ctrl._stats.append(dur)
                ctrl.adjust(response.status_code >= 500)

        def patched_init(self_client, *args, **kwargs):
            hooks = kwargs.get("event_hooks") or {}
            hooks.setdefault("request", []).append(_on_request)
            hooks.setdefault("response", []).append(_on_response)
            kwargs["event_hooks"] = hooks
            _orig_init(self_client, *args, **kwargs)

        async def throttled_send(self_client, request, *args, **kwargs):
            async with ctrl._semaphore:
                return await _orig_send(self_client, request, *args, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        httpx.AsyncClient.send = throttled_send

        # Also patch sync Client to strip encoding_format=None
        _orig_sync_send = httpx.Client.send

        def patched_sync_send(self_client, request, *args, **kwargs):
            if request.method == "POST" and request.content:
                try:
                    body = json.loads(request.content)
                    if "encoding_format" in body and body["encoding_format"] is None:
                        del body["encoding_format"]
                        new_content = json.dumps(body).encode("utf-8")
                        request = httpx.Request(
                            method=request.method,
                            url=request.url,
                            headers={k: v for k, v in request.headers.items() if k.lower() != "content-length"},
                            content=new_content,
                        )
                except Exception:
                    pass
            return _orig_sync_send(self_client, request, *args, **kwargs)

        httpx.Client.send = patched_sync_send
        self._start_stats_printer()

    def _start_stats_printer(self):
        def _run():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._print_stats())
        threading.Thread(target=_run, daemon=True).start()

    async def _print_stats(self):
        while True:
            await asyncio.sleep(60)
            with self._stats_lock:
                snap = self._stats.copy()
                self._stats.clear()
            if snap:
                log.info(
                    f"[STATS] Last 60s: count={len(snap)}, "
                    f"min={min(snap):.2f}s, max={max(snap):.2f}s, "
                    f"avg={sum(snap)/len(snap):.2f}s"
                )
