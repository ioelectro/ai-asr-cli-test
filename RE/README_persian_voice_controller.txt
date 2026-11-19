Persian Offline Voice Command Engine
====================================

Files
-----
persian_voice_controller.py       Main parser
requirements_voice_controller.txt Optional timezone/Jalali dependencies
test_persian_voice_controller.py  Automated regression tests

Python
------
Python 3.10 or newer.

Install optional dependencies
------------------------------
python -m pip install -r requirements_voice_controller.txt

Run
---
python persian_voice_controller.py

Run tests
---------
python -m unittest -v test_persian_voice_controller.py

Timezone
--------
The default timezone is Europe/Berlin. Set VOICE_TIMEZONE to the timezone of
where the controlled home is located, for example:

Linux / WSL:
    export VOICE_TIMEZONE=Asia/Tehran

Windows PowerShell:
    $env:VOICE_TIMEZONE = "Asia/Tehran"

The setting must be applied before starting Python. Restart the program after
changing it.

Morning defaults
----------------
When only a part of day is spoken, the parser uses:
  morning: 08:00
  noon:    12:00
  afternoon: 15:00
  evening: 19:00
  night:   21:00

Change these constants at the top of the Python file if needed.

Output contract
---------------
Each parsed sentence produces one JSON object. Important fields:
  status: ACCEPTED or NEEDS_CONFIRMATION or UNKNOWN
  domain: device, alarm, timer, or reminder
  device: stable equipment ID
  location: stable room ID (or default)
  action: stable operation ID
  value/unit: optional setting
  schedule: optional one-time, timer, or recurrence data
  confirmation_required: safety gate for door/lock operations

A missing schedule means execute the device command immediately.
A one-time schedule contains local execute_at, UTC execute_at_utc, and timezone.
A timer contains duration_seconds and its expiration timestamp.
A recurrence contains rule, time, start_date, and optional interval/count.

Important execution note
------------------------
This program parses and emits commands. It does not persist future jobs, wake
up at the scheduled time, generate desktop notifications, or send UART data.
The host application must store scheduled commands and dispatch the device
part (device/location/action/parameter/value) to the ESP32 at execute_at.
Do not dispatch a device/lock command unless status is ACCEPTED. A door/lock
operation is intentionally returned as NEEDS_CONFIRMATION.

Jalali dates
------------
Numeric Jalali dates such as 1405/09/05 and month-name phrases such as
25 آبان 1405 are converted internally using the Python standard library.
No Jalali package or internet connection is needed at runtime.
