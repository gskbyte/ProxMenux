"""Tests for user-configurable Vital Signs sampling intervals."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import health_thresholds as ht
from health_fixture_modules import load_source


class TestHealthSamplingIntervals(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.orig_db_dir = ht._DB_DIR
        self.orig_config_path = ht._CONFIG_PATH
        ht._DB_DIR = self.temporary.name
        ht._CONFIG_PATH = os.path.join(self.temporary.name, "health_thresholds.json")
        ht.invalidate_cache()

    def tearDown(self):
        ht._DB_DIR = self.orig_db_dir
        ht._CONFIG_PATH = self.orig_config_path
        ht.invalidate_cache()

    def test_default_sampling_intervals(self):
        self.assertIn("sampling_intervals", ht.DEFAULTS)
        self.assertEqual(ht.get("sampling_intervals", "cpu"), 5.0)
        self.assertEqual(ht.get("sampling_intervals", "temperature"), 15.0)
        self.assertEqual(ht.get("sampling_intervals", "memory"), 30.0)

    def test_override_sampling_intervals(self):
        payload = {
            "sampling_intervals": {
                "cpu": 15,
                "temperature": 45,
                "memory": 60,
            }
        }
        res = ht.save(payload)
        self.assertTrue(res["sampling_intervals"]["cpu"]["customised"])
        self.assertEqual(res["sampling_intervals"]["cpu"]["value"], 15.0)
        self.assertEqual(ht.get("sampling_intervals", "cpu"), 15.0)
        self.assertEqual(ht.get("sampling_intervals", "temperature"), 45.0)
        self.assertEqual(ht.get("sampling_intervals", "memory"), 60.0)

    def test_validation_out_of_range(self):
        # min for cpu is 1, max is 300
        with self.assertRaises(ht.ThresholdValidationError):
            ht.save({"sampling_intervals": {"cpu": 0}})

        with self.assertRaises(ht.ThresholdValidationError):
            ht.save({"sampling_intervals": {"cpu": 500}})

        # temperature: min 5, max 300
        with self.assertRaises(ht.ThresholdValidationError):
            ht.save({"sampling_intervals": {"temperature": 2}})

        with self.assertRaises(ht.ThresholdValidationError):
            ht.save({"sampling_intervals": {"temperature": 400}})

        # memory: min 5, max 600
        with self.assertRaises(ht.ThresholdValidationError):
            ht.save({"sampling_intervals": {"memory": 1}})

        with self.assertRaises(ht.ThresholdValidationError):
            ht.save({"sampling_intervals": {"memory": 700}})

    def test_reset_sampling_intervals_section(self):
        ht.save({"sampling_intervals": {"cpu": 20}})
        self.assertEqual(ht.get("sampling_intervals", "cpu"), 20.0)

        ht.reset_section("sampling_intervals")
        self.assertEqual(ht.get("sampling_intervals", "cpu"), 5.0)

    def test_reset_all_resets_sampling_intervals(self):
        ht.save({"sampling_intervals": {"cpu": 20, "temperature": 40}})
        self.assertEqual(ht.get("sampling_intervals", "cpu"), 20.0)
        self.assertEqual(ht.get("sampling_intervals", "temperature"), 40.0)

        ht.reset_all()
        self.assertEqual(ht.get("sampling_intervals", "cpu"), 5.0)
        self.assertEqual(ht.get("sampling_intervals", "temperature"), 15.0)

    def test_vital_signs_refresh_intervals(self):
        with mock.patch.dict(sys.modules, {"psutil": types.ModuleType("psutil")}):
            vital_signs = load_source("vital_signs")

            vital_signs.refresh_intervals()
            self.assertEqual(vital_signs.CPU_INTERVAL, 5)
            self.assertEqual(vital_signs.TEMPERATURE_INTERVAL, 15)
            self.assertEqual(vital_signs.MEMORY_INTERVAL, 30)

            ht.save({
                "sampling_intervals": {
                    "cpu": 10,
                    "temperature": 25,
                    "memory": 50,
                }
            })
            vital_signs.refresh_intervals()
            self.assertEqual(vital_signs.CPU_INTERVAL, 10)
            self.assertEqual(vital_signs.TEMPERATURE_INTERVAL, 25)
            self.assertEqual(vital_signs.MEMORY_INTERVAL, 50)

            ht.reset_section("sampling_intervals")
            vital_signs.refresh_intervals()
            self.assertEqual(vital_signs.CPU_INTERVAL, 5)
            self.assertEqual(vital_signs.TEMPERATURE_INTERVAL, 15)
            self.assertEqual(vital_signs.MEMORY_INTERVAL, 30)

            # Resilient against invalid/boolean types
            with mock.patch.object(ht, "get", side_effect=lambda sec, key, **kw: False if key == "cpu" else 0):
                vital_signs.refresh_intervals()
                self.assertEqual(vital_signs.CPU_INTERVAL, 5)  # boolean falls back to default 5
                self.assertEqual(vital_signs.TEMPERATURE_INTERVAL, 1)  # clamped to min 1
                self.assertEqual(vital_signs.MEMORY_INTERVAL, 1)  # clamped to min 1


if __name__ == "__main__":
    unittest.main()
