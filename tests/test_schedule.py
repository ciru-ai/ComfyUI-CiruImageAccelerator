"""Contracts for the adjustable full-evaluation control."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("ciru_turbo_schedule", Path(__file__).resolve().parents[1] / "schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class ScheduleTests(unittest.TestCase):
    def test_known_seven_evaluation_presets(self):
        for steps, expected in schedule.SEVEN_EVALUATION_SCHEDULES.items():
            self.assertEqual(schedule.full_evaluation_schedule(steps, 7), expected)

    def test_default_is_twelve(self):
        self.assertEqual(len(schedule.full_evaluation_schedule(30)), 12)

    def test_every_count_is_nested_and_all_full_is_exact(self):
        for steps in range(20, 61):
            previous = set()
            for count in range(6, steps + 1):
                current = schedule.full_evaluation_schedule(steps, count)
                self.assertEqual(len(current), count)
                self.assertTrue(previous.issubset(current))
                self.assertEqual(current[:2], (0, 1))
                self.assertLess(current[-1], steps)
                previous = set(current)
            self.assertEqual(schedule.full_evaluation_schedule(steps, steps), tuple(range(steps)))

    def test_failed_five_evaluation_setting_is_unavailable(self):
        with self.assertRaises(ValueError):
            schedule.full_evaluation_schedule(25, 5)


if __name__ == "__main__":
    unittest.main()
