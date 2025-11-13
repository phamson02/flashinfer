import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch

import flashinfer.autotuner as autotuner_module
from flashinfer.autotuner import (
    AutoTuner,
    TuningConfig,
    TunableRunner,
    default_profiling_cache_file,
)


class DummyRunner(TunableRunner):
    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile
    ) -> List[int]:  # pragma: no cover - unused in test
        return [0]

    def forward(  # pragma: no cover - unused in test
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        return None


def _reset_autotuner_state() -> Tuple[Optional[AutoTuner], bool]:
    old_instance = AutoTuner._instance
    old_registered = AutoTuner._atexit_registered
    AutoTuner._instance = None
    AutoTuner._atexit_registered = False
    return old_instance, old_registered


def _restore_autotuner_state(
    old_instance: AutoTuner, old_registered: bool
) -> None:
    AutoTuner._instance = old_instance
    AutoTuner._atexit_registered = old_registered


def test_autotuner_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_TUNING_MODE", "1")
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE", "1")
    cache_file = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE_PATH", str(cache_file))
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)

    old_instance, old_registered = _reset_autotuner_state()
    try:
        tuner = AutoTuner.get()
        tuning_config = TuningConfig()
        shapes = (torch.Size([2, 3]),)
        cache_key = AutoTuner._get_cache_key(
            "dummy::op", DummyRunner(), shapes, tuning_config
        )
        tuner.profiling_cache[cache_key] = (0, 13, None)
        tuner._cache_dirty = True
        tuner.maybe_persist_cache()

        payload = json.loads(cache_file.read_text())
        assert payload["entries"][0]["tactic"] == 13

        AutoTuner._instance = None
        tuner_reloaded = AutoTuner.get()
        hit, runner_id, tactic, stored_profile = tuner_reloaded.search_cache(
            "dummy::op", [DummyRunner()], shapes, tuning_config
        )

        assert hit is True
        assert runner_id == 0
        assert tactic == 13
        assert stored_profile is None
    finally:
        _restore_autotuner_state(old_instance, old_registered)


def test_autotuner_cache_respects_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)
    cache_file = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE_PATH", str(cache_file))

    monkeypatch.setenv("FLASHINFER_AUTOTUNER_TUNING_MODE", "1")
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_CACHE", raising=False)

    old_instance, old_registered = _reset_autotuner_state()
    try:
        tuner = AutoTuner.get()
        shapes = (torch.Size([2, 3]),)
        tuning_config = TuningConfig()
        cache_key = AutoTuner._get_cache_key(
            "dummy::op", DummyRunner(), shapes, tuning_config
        )
        tuner.profiling_cache[cache_key] = (0, 7, None)
        tuner._cache_dirty = True
        tuner.maybe_persist_cache()

        assert not cache_file.exists()

        payload = {
            "version": 1,
            "entries": [
                {
                    "custom_op": "dummy::op",
                    "runner": "DummyRunner",
                    "profile": [[2, 3]],
                    "tactic": 5,
                }
            ],
        }
        cache_file.write_text(json.dumps(payload))

        AutoTuner._instance = None
        tuner = AutoTuner.get()

        hit, runner_id, tactic, stored_profile = tuner.search_cache(
            "dummy::op", [DummyRunner()], shapes, tuning_config
        )
        assert hit is False
        assert runner_id == 0
        assert tactic == -1
        assert stored_profile is None
    finally:
        _restore_autotuner_state(old_instance, old_registered)


def test_autotuner_default_path_and_exit_flush(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_CACHE_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_TUNING_MODE", "1")
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE", "1")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    registered = []

    def fake_register(func):
        registered.append(func)
        return func

    monkeypatch.setattr(autotuner_module.atexit, "register", fake_register)

    old_instance, old_registered = _reset_autotuner_state()
    try:
        tuner = AutoTuner.get()
        expected_path = default_profiling_cache_file()
        assert expected_path == tmp_path / ".cache" / "flashinfer" / "autotuner_cache.json"
        assert registered, "atexit.register should be called when persistence is enabled"

        shapes = (torch.Size([2, 3]),)
        tuning_config = TuningConfig()
        cache_key = AutoTuner._get_cache_key(
            "dummy::op", DummyRunner(), shapes, tuning_config
        )
        tuner.profiling_cache[cache_key] = (0, 23, None)
        tuner._cache_dirty = True

        registered[0]()

        payload = json.loads(expected_path.read_text())
        assert payload["entries"][0]["tactic"] == 23
    finally:
        _restore_autotuner_state(old_instance, old_registered)
