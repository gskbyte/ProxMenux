"""Tests for user-configurable Vital Signs sampling intervals."""

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from health_fixture_modules import load_health_modules, load_source  # noqa: E402

Times = namedtuple("Times", "user nice system idle iowait irq softirq steal guest guest_nice")


def _load_isolated_thresholds(temp_dir):
    ht = load_source("health_thresholds")
    ht._DB_DIR = temp_dir
    ht._CONFIG_PATH = os.path.join(temp_dir, "health_thresholds.json")
    ht.invalidate_cache()
    return ht


def _load_isolated_vital_signs():
    with mock.patch.dict(sys.modules, {"psutil": types.ModuleType("psutil")}):
        return load_source("vital_signs")


def _extract_flask_function(name, namespace):
    source = (SCRIPTS_DIR / "flask_server.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="flask_server.py")
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    exec(compile(module, "flask_server.py", "exec"), namespace)
    return namespace[name]


class TestHealthSamplingIntervals(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sampling-intervals-")
        self.addCleanup(self.temporary.cleanup)
        self.ht = _load_isolated_thresholds(self.temporary.name)

    def test_default_sampling_intervals(self):
        self.assertIn("sampling_intervals", self.ht.DEFAULTS)
        self.assertEqual(self.ht.get("sampling_intervals", "cpu"), 5.0)
        self.assertEqual(self.ht.get("sampling_intervals", "temperature"), 15.0)
        self.assertEqual(self.ht.get("sampling_intervals", "memory"), 30.0)

        effective = self.ht.load_effective()["sampling_intervals"]
        self.assertEqual(effective["cpu"]["unit"], "s")
        self.assertEqual(effective["cpu"]["min"], 1)
        self.assertEqual(effective["cpu"]["max"], 300)
        self.assertEqual(effective["temperature"]["min"], 5)
        self.assertEqual(effective["temperature"]["max"], 300)
        self.assertEqual(effective["memory"]["min"], 5)
        self.assertEqual(effective["memory"]["max"], 600)
        self.assertFalse(effective["cpu"]["customised"])

    def test_override_sampling_intervals_persists(self):
        res = self.ht.save({
            "sampling_intervals": {
                "cpu": 15,
                "temperature": 45,
                "memory": 60,
            }
        })
        self.assertTrue(res["sampling_intervals"]["cpu"]["customised"])
        self.assertEqual(res["sampling_intervals"]["cpu"]["value"], 15.0)

        self.ht.invalidate_cache()
        self.assertEqual(self.ht.get("sampling_intervals", "cpu"), 15.0)
        self.assertEqual(self.ht.get("sampling_intervals", "temperature"), 45.0)
        self.assertEqual(self.ht.get("sampling_intervals", "memory"), 60.0)

    def test_validation_out_of_range(self):
        invalid_payloads = [
            {"sampling_intervals": {"cpu": 0}},
            {"sampling_intervals": {"cpu": 301}},
            {"sampling_intervals": {"temperature": 4}},
            {"sampling_intervals": {"temperature": 301}},
            {"sampling_intervals": {"memory": 4}},
            {"sampling_intervals": {"memory": 601}},
            {"sampling_intervals": {"cpu": True}},
            {"sampling_intervals": {"cpu": False}},
            {"sampling_intervals": {"cpu": float("nan")}},
            {"sampling_intervals": {"cpu": "fast"}},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(self.ht.ThresholdValidationError):
                    self.ht.save(payload)

    def test_reset_sampling_intervals_section_preserves_other_sections(self):
        self.ht.save({
            "cpu": {"warning": 80},
            "sampling_intervals": {"cpu": 20, "temperature": 60, "memory": 120},
        })
        self.ht.reset_section("sampling_intervals")

        self.assertEqual(self.ht.get("sampling_intervals", "cpu"), 5.0)
        self.assertEqual(self.ht.get("sampling_intervals", "temperature"), 15.0)
        self.assertEqual(self.ht.get("sampling_intervals", "memory"), 30.0)
        self.assertEqual(self.ht.get("cpu", "warning"), 80.0)

    def test_reset_all_restores_sampling_intervals(self):
        self.ht.save({"sampling_intervals": {"cpu": 25}})
        self.ht.reset_all()
        self.assertEqual(self.ht.get("sampling_intervals", "cpu"), 5.0)


class TestVitalSignsIntervalSync(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vital-signs-intervals-")
        self.addCleanup(self.temporary.cleanup)
        self.ht = _load_isolated_thresholds(self.temporary.name)
        self.vs = _load_isolated_vital_signs()

    def test_refresh_intervals_applies_overrides_and_signals_only_on_change(self):
        self.vs.clear_interval_change()
        self.assertEqual(self.vs.refresh_intervals(self.ht), (5, 15, 30))
        self.assertFalse(self.vs.wait_for_interval_change(0))

        self.ht.save({"sampling_intervals": {"cpu": 20, "temperature": 45, "memory": 90}})
        self.assertEqual(self.vs.refresh_intervals(self.ht), (20, 45, 90))
        self.assertEqual(self.vs.CPU_INTERVAL, 20)
        self.assertEqual(self.vs.TEMPERATURE_INTERVAL, 45)
        self.assertEqual(self.vs.MEMORY_INTERVAL, 90)
        self.assertTrue(self.vs.wait_for_interval_change(0))

        # No-op refresh does not re-trigger the wake signal.
        self.vs.clear_interval_change()
        self.assertEqual(self.vs.refresh_intervals(self.ht), (20, 45, 90))
        self.assertFalse(self.vs.wait_for_interval_change(0))

        # Resetting restores defaults and signals change.
        self.ht.reset_section("sampling_intervals")
        self.assertEqual(self.vs.refresh_intervals(self.ht), (5, 15, 30))
        self.assertEqual(self.vs.CPU_INTERVAL, 5)
        self.assertEqual(self.vs.TEMPERATURE_INTERVAL, 15)
        self.assertEqual(self.vs.MEMORY_INTERVAL, 30)
        self.assertTrue(self.vs.wait_for_interval_change(0))

    def test_refresh_intervals_falls_back_on_corrupt_disk_values(self):
        with open(self.ht._CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"sampling_intervals": {"cpu": True, "temperature": 0, "memory": 9999}},
                f,
            )
        self.ht.invalidate_cache()

        # cpu: true must NOT coerce to 1s (min=1); all invalid values fall back to defaults.
        self.assertEqual(self.ht.get("sampling_intervals", "cpu"), 5.0)
        self.assertEqual(self.vs.refresh_intervals(self.ht), (5, 15, 30))
        self.assertEqual(self.vs.CPU_INTERVAL, 5)
        self.assertEqual(self.vs.TEMPERATURE_INTERVAL, 15)
        self.assertEqual(self.vs.MEMORY_INTERVAL, 30)

    def test_cpu_cache_window_follows_configured_cpu_interval(self):
        times = [
            Times(0, 0, 0, 100, 0, 0, 0, 0, 0, 0),
            Times(50, 0, 0, 150, 0, 0, 0, 0, 0, 0),
        ]
        clock = [1000.0]
        fake_psutil = types.SimpleNamespace(cpu_times=lambda: times.pop(0))

        self.ht.save({"sampling_intervals": {"cpu": 60}})
        self.vs.refresh_intervals(self.ht)

        with mock.patch.object(self.vs, "psutil", fake_psutil), \
                mock.patch.object(self.vs.time, "monotonic", lambda: clock[0]):
            self.assertIsNone(self.vs.sample_cpu())
            sampled = self.vs.sample_cpu()
            # 120s is well beyond the default 3 * 5s window, but within 3 * 60s.
            clock[0] += 120.0
            self.assertEqual(self.vs.cpu(), sampled)
            self.assertEqual(times, [])


class TestVitalSignsSamplerDynamicSchedule(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sampler-schedule-")
        self.addCleanup(self.temporary.cleanup)
        self.ht = _load_isolated_thresholds(self.temporary.name)
        self.vs = _load_isolated_vital_signs()

    def test_sampler_reschedules_when_intervals_shorten_or_lengthen(self):
        self.ht.save({"sampling_intervals": {"cpu": 300, "temperature": 300, "memory": 600}})

        clock = [1000.0]
        events = []
        sleeps = []

        class StopSampler(BaseException):
            pass

        fake_monitor = types.SimpleNamespace(
            _sample_cpu_usage=lambda: events.append(("cpu", clock[0])),
            _sample_cpu_temperature=lambda: events.append(("temp", clock[0])),
            _sample_memory_usage=lambda: events.append(("mem", clock[0])),
        )
        fake_hm_module = types.ModuleType("health_monitor")
        fake_hm_module.health_monitor = fake_monitor

        def fake_wait(timeout):
            sleeps.append((clock[0], timeout))
            step = len(sleeps)
            if step == 1:
                # 10s later (while sleeping up to 300s), user shortens CPU to 5s.
                clock[0] += 10.0
                self.ht.save({"sampling_intervals": {"cpu": 5}})
                self.vs.refresh_intervals(self.ht)
                return True
            if step == 2:
                # 2s later, user lengthens CPU to 120s.
                clock[0] += 2.0
                self.ht.save({"sampling_intervals": {"cpu": 120}})
                self.vs.refresh_intervals(self.ht)
                return True
            raise StopSampler()

        fake_time = types.SimpleNamespace(
            sleep=lambda _s: None,
            monotonic=lambda: clock[0],
        )
        namespace = {
            "time": fake_time,
            "vital_signs": self.vs,
        }
        sampler = _extract_flask_function("_vital_signs_sampler", namespace)

        with mock.patch.dict(
            sys.modules,
            {"health_monitor": fake_hm_module, "health_thresholds": self.ht},
        ), mock.patch.object(
            self.vs, "wait_for_interval_change", side_effect=fake_wait
        ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(StopSampler):
                sampler()

        # Tick 1 (t=1000): CPU sampled immediately; next earliest is temp stagger at +7s.
        self.assertEqual(events[0], ("cpu", 1000.0))
        self.assertAlmostEqual(sleeps[0][1], 7.0)

        # Tick 2 (t=1010): after shortening CPU 300s -> 5s, last_cpu + 5 = 1005 <= 1010,
        # so CPU samples immediately and temp (+7s = 1007) also fires; next sleep is 5s.
        self.assertIn(("cpu", 1010.0), events)
        self.assertIn(("temp", 1010.0), events)
        self.assertAlmostEqual(sleeps[1][1], 5.0)

        # Tick 3 (t=1012): after lengthening CPU 5s -> 120s, CPU does NOT sample at 1012;
        # next earliest is mem stagger at 1015 (3s away), not 1015 from old 5s CPU cadence.
        self.assertNotIn(("cpu", 1012.0), events)
        self.assertAlmostEqual(sleeps[2][1], 3.0)

    def test_threshold_api_routes_refresh_vital_signs(self):
        source = (SCRIPTS_DIR / "flask_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="flask_server.py")
        for route_name in ("api_health_thresholds_put", "api_health_thresholds_reset"):
            func = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == route_name
            )
            calls = [
                node for node in ast.walk(func)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "refresh_intervals"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "vital_signs"
            ]
            self.assertTrue(calls, f"{route_name} must call vital_signs.refresh_intervals")

    def test_health_monitor_refresh_thresholds_syncs_vital_signs(self):
        hm, _, ht = load_health_modules(self.temporary.name)
        ht._DB_DIR = self.temporary.name
        ht.invalidate_cache()
        monitor = hm.HealthMonitor.__new__(hm.HealthMonitor)
        orig = (
            hm.vital_signs.CPU_INTERVAL,
            hm.vital_signs.TEMPERATURE_INTERVAL,
            hm.vital_signs.MEMORY_INTERVAL,
        )
        self.addCleanup(
            lambda: (
                setattr(hm.vital_signs, "CPU_INTERVAL", orig[0]),
                setattr(hm.vital_signs, "TEMPERATURE_INTERVAL", orig[1]),
                setattr(hm.vital_signs, "MEMORY_INTERVAL", orig[2]),
            )
        )

        ht.save({"sampling_intervals": {"cpu": 25, "temperature": 50, "memory": 120}})
        with mock.patch.dict(sys.modules, {"health_thresholds": ht}):
            monitor._refresh_thresholds()

        self.assertEqual(hm.vital_signs.CPU_INTERVAL, 25)
        self.assertEqual(hm.vital_signs.TEMPERATURE_INTERVAL, 50)
        self.assertEqual(hm.vital_signs.MEMORY_INTERVAL, 120)

    def test_max_sampling_intervals_do_not_drop_samples_between_ticks(self):
        hm, _, _ = load_health_modules(self.temporary.name)
        monitor = hm.HealthMonitor.__new__(hm.HealthMonitor)
        monitor.cached_results = {}
        monitor.last_check_times = {}
        now = hm.time.time()

        # Samples taken right before the next tick at max configured intervals (300s / 300s / 600s).
        monitor.state_history = {
            "cpu_usage": [{"value": 97.0, "time": now - 305}],
            "cpu_temp_history": [{"value": 86.0, "time": now - 305}],
            "memory_usage": [
                {"mem_percent": 96.0, "swap_percent": 90.0, "available_percent": 5.0, "time": now - 605}
            ],
        }

        mem = types.SimpleNamespace(
            percent=96.0, available=1 * 1024**3, total=16 * 1024**3
        )
        swap = types.SimpleNamespace(
            percent=90.0, used=9 * 1024**3, total=10 * 1024**3
        )

        with mock.patch.object(hm.vital_signs, "CPU_INTERVAL", 300), \
                mock.patch.object(hm.vital_signs, "TEMPERATURE_INTERVAL", 300), \
                mock.patch.object(hm.vital_signs, "MEMORY_INTERVAL", 600), \
                mock.patch.object(hm.vital_signs, "cpu", return_value={"value": 97.0}), \
                mock.patch.object(hm.vital_signs, "memory", return_value=(mem, swap)), \
                mock.patch.object(hm, "psutil", types.SimpleNamespace(cpu_count=lambda *a, **k: 4), create=True), \
                mock.patch.object(hm.health_persistence, "record_error"), \
                mock.patch.object(hm.health_persistence, "resolve_error"), \
                mock.patch.object(hm.health_persistence, "is_error_active", return_value=False):
            temp_res = monitor._check_cpu_temperature()
            self.assertIsNotNone(temp_res)
            self.assertEqual(temp_res["status"], "WARNING")

            cpu_res = monitor._check_cpu_with_hysteresis()
            self.assertEqual(cpu_res["status"], "CRITICAL")

            mem_res = monitor._check_memory_comprehensive()
            self.assertEqual(mem_res["status"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
