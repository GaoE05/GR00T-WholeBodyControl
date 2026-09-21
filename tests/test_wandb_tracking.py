"""CPU tests for mandatory online-first W&B initialization."""
import os

from gear_sonic.utils.wandb_tracking import PROXY_KEYS, init_wandb_online_first


class FakeWandb:
    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def __init__(self, fail_online=False):
        self.fail_online = fail_online
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["mode"] == "online" and self.fail_online:
            raise ConnectionError("offline test")
        return object()

    def finish(self, **kwargs):
        self.calls.append({"finish": kwargs})


def test_online_clears_stale_proxy_and_mode(monkeypatch):
    for key in PROXY_KEYS:
        monkeypatch.setenv(key, "http://127.0.0.1:7890")
    monkeypatch.setenv("WANDB_MODE", "offline")
    fake = FakeWandb()
    _, status = init_wandb_online_first(fake, project="test")
    assert fake.calls[0]["mode"] == "online"
    assert status["actual_mode"] == "online"
    assert status["fallback_reason"] is None
    assert "WANDB_MODE" in status["removed_environment"]
    assert not any(key in os.environ for key in PROXY_KEYS)


def test_online_failure_is_explicit_offline_fallback(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    fake = FakeWandb(fail_online=True)
    _, status = init_wandb_online_first(fake, project="test")
    assert fake.calls[0]["mode"] == "online"
    assert fake.calls[-1]["mode"] == "offline"
    assert status["actual_mode"] == "offline"
    assert "ConnectionError" in status["fallback_reason"]
