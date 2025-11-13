import json
from typing import List

import torch

from flashinfer.autotuner import AutoTuner, TuningConfig, TunableRunner


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


def test_autotuner_persistent_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_TUNING_MODE", "1")
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE", "1")
    cache_file = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE_PATH", str(cache_file))
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)

    old_instance = AutoTuner._instance
    AutoTuner._instance = None
    try:
        tuner = AutoTuner.get()
        runner = DummyRunner()
        tuning_config = TuningConfig()
        shapes = (torch.Size([2, 3]),)

        cache_key = AutoTuner._get_cache_key("dummy::op", runner, shapes, tuning_config)
        tuner.profiling_cache[cache_key] = (0, 13, None)
        tuner._cache_dirty = True
        tuner.maybe_persist_cache()

        assert cache_file.exists()
        payload = json.loads(cache_file.read_text())
        assert payload["entries"][0]["custom_op"] == "dummy::op"
        assert payload["entries"][0]["runner"] == "DummyRunner"
        assert payload["entries"][0]["profile"] == [[2, 3]]
        assert payload["entries"][0]["tactic"] == 13

        AutoTuner._instance = None
        tuner_reloaded = AutoTuner.get()
        runner_reloaded = DummyRunner()
        hit, runner_id, tactic, stored_profile = tuner_reloaded.search_cache(
            "dummy::op", [runner_reloaded], shapes, tuning_config
        )

        assert hit is True
        assert runner_id == 0
        assert tactic == 13
        assert stored_profile is None
    finally:
        AutoTuner._instance = old_instance


def test_autotuner_persistent_cache_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_TUNING_MODE", "1")
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_CACHE", raising=False)
    cache_file = tmp_path / "cache.json"
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE_PATH", str(cache_file))
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)

    old_instance = AutoTuner._instance
    AutoTuner._instance = None
    try:
        tuner = AutoTuner.get()
        runner = DummyRunner()
        tuning_config = TuningConfig()
        shapes = (torch.Size([2, 3]),)

        cache_key = AutoTuner._get_cache_key("dummy::op", runner, shapes, tuning_config)
        tuner.profiling_cache[cache_key] = (0, 7, None)
        tuner._cache_dirty = True
        tuner.maybe_persist_cache()

        assert not cache_file.exists()
    finally:
        AutoTuner._instance = old_instance


def test_autotuner_skips_loading_without_flag(tmp_path, monkeypatch):
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
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps(payload))

    monkeypatch.setenv("FLASHINFER_AUTOTUNER_CACHE_PATH", str(cache_file))
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)

    old_instance = AutoTuner._instance
    AutoTuner._instance = None
    try:
        tuner = AutoTuner.get()
        runner = DummyRunner()
        tuning_config = TuningConfig()
        shapes = (torch.Size([2, 3]),)

        hit, runner_id, tactic, stored_profile = tuner.search_cache(
            "dummy::op", [runner], shapes, tuning_config
        )

        assert hit is False
        assert runner_id == 0
        assert tactic == -1
        assert stored_profile is None
    finally:
        AutoTuner._instance = old_instance
