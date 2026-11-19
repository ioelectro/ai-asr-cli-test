import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from persian_voice_controller import PersianCommandEngine, TIMEZONE_NAME


class PersianCommandEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PersianCommandEngine()
        self.now = datetime(2026, 11, 19, 20, 0, tzinfo=ZoneInfo(TIMEZONE_NAME))

    def check(self, phrase, status, device, action):
        result = self.engine.parse(phrase, self.now).to_dict()
        self.assertEqual(result["status"], status, result)
        self.assertEqual(result["device"], device, result)
        self.assertEqual(result["action"], action, result)
        return result

    def test_basic_devices(self):
        self.check("چراغ اتاق رو خاموش کن", "ACCEPTED", "light", "off")
        self.check("لامپ هال رو روشن کن", "ACCEPTED", "light", "on")
        self.check("پنکه اتاق رو نصف کن", "ACCEPTED", "fan", "fan_speed_set")
        self.check("سرعت کولر رو نصف کن", "ACCEPTED", "ac", "fan_speed_set")
        self.check("دمای کولر رو روی 24 درجه تنظیم کن", "ACCEPTED", "ac", "temperature_set")

    def test_percent_and_location(self):
        result = self.check("لامپ هال رو روی پنجاه درصد روشن کن", "ACCEPTED", "light", "brightness_set")
        self.assertEqual(result["value"], 50)
        self.assertEqual(result["location"], "living_room")

    def test_open_close_and_confirmation(self):
        result = self.check("درب ورودی رو باز کن", "NEEDS_CONFIRMATION", "door", "open")
        self.assertTrue(result["confirmation_required"])
        result = self.check("قفل در ورودی رو باز کن", "NEEDS_CONFIRMATION", "lock", "unlock")
        self.assertTrue(result["confirmation_required"])
        self.check("پنجره اتاق رو باز کن", "ACCEPTED", "window", "open")
        self.check("پرده پذیرایی رو 30 درصد باز کن", "ACCEPTED", "curtain", "set_position")

    def test_relative_schedule(self):
        result = self.check("بعد از 2 ساعت و 30 دقیقه پنکه رو خاموش کن", "ACCEPTED", "fan", "off")
        self.assertEqual(result["schedule"]["delay_seconds"], 9000)
        result = self.check("فردا ساعت 7:30 چراغ رو روشن کن", "ACCEPTED", "light", "on")
        self.assertTrue(result["schedule"]["execute_at"].endswith("T07:30:00+01:00"))

    def test_timer_and_alarm(self):
        result = self.check("تایمر رو برای 10 دقیقه تنظیم کن", "ACCEPTED", "timer", "start")
        self.assertEqual(result["schedule"]["duration_seconds"], 600)
        result = self.check("هشدار برای فردا صبح", "ACCEPTED", "alarm", "set")
        self.assertTrue(result["schedule"]["execute_at"].endswith("T08:00:00+01:00"))

    def test_reminders(self):
        result = self.check("یادم بنداز سه روز دیگه قبض رو پرداخت کنم", "ACCEPTED", "reminder", "create")
        self.assertEqual(result["schedule"]["delay_seconds"], 3 * 86400)
        result = self.check("یادآوری هر روز ساعت 8 صبح به مدت 3 روز دارو", "ACCEPTED", "reminder", "create")
        self.assertEqual(result["schedule"]["rule"], "daily")
        self.assertEqual(result["schedule"]["occurrences"], 3)

    def test_gregorian_date(self):
        result = self.check("یادآوری برای تاریخ 2026-11-25 ساعت 10 خرید", "ACCEPTED", "reminder", "create")
        self.assertIn("2026-11-25T10:00:00", result["schedule"]["execute_at"])

    def test_jalali_date(self):
        result = self.check("چراغ را در تاریخ 1405/09/05 ساعت 10 روشن کن", "ACCEPTED", "light", "on")
        self.assertIn("2026-11-26T10:00:00", result["schedule"]["execute_at"])

    def test_negation_is_not_executed(self):
        result = self.engine.parse("چراغ رو خاموش نکن", self.now).to_dict()
        self.assertEqual(result["status"], "NEEDS_CONFIRMATION")


if __name__ == "__main__":
    unittest.main(verbosity=2)
