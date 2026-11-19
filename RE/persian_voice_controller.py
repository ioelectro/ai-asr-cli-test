#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Persian voice-command parser for home automation.

Python: 3.10+
Optional Jalali date support: pip install jdatetime

The program parses text produced by an ASR engine (for example Shenava) and
prints one JSON command. It does not directly operate hardware. A host-side
scheduler/dispatcher must execute scheduled jobs and send device commands to
ESP32 when due. Door opening/unlocking is marked confirmation_required=True.

Configure VOICE_TIMEZONE (IANA name), e.g. Asia/Tehran or Europe/Berlin.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
TIMEZONE_NAME = os.getenv("VOICE_TIMEZONE", "Asia/Tehran")
MORNING_DEFAULT = "08:00"       # Used only for e.g. "فردا صبح" with no hour.
AFTERNOON_DEFAULT = "15:00"     # Used only for e.g. "فردا بعدازظهر".
EVENING_DEFAULT = "19:00"       # Used only for e.g. "فردا عصر".
NIGHT_DEFAULT = "21:00"         # Used only for e.g. "فردا شب".

try:
    LOCAL_TZ = ZoneInfo(TIMEZONE_NAME)
except ZoneInfoNotFoundError as exc:
    raise RuntimeError(
        f"Unknown timezone {TIMEZONE_NAME!r}. Set VOICE_TIMEZONE to a valid "
        "IANA timezone such as Asia/Tehran or Europe/Berlin."
    ) from exc


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------------------
_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_REPLACEMENTS = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه",
    "إ": "ا", "أ": "ا", "ٱ": "ا", "ؤ": "و",
    "\u200c": " ", "\u200f": " ", "\ufeff": " ",
}


def normalize(text: str) -> str:
    text = str(text).translate(_DIGITS)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.lower()
    text = text.replace("٪", "%")
    text = re.sub(r"[،؛,!?؟()\[\]{}\"'`،]", " ", text)
    text = re.sub(r"[^\w\s./%+:-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\S){re.escape(phrase)}(?!\S)", text))


# ---------------------------------------------------------------------------
# PERSIAN NUMBERS
# ---------------------------------------------------------------------------
ONES = {
    "صفر": 0, "یک": 1, "یه": 1, "یکی": 1,
    "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
    "شش": 6, "شیش": 6, "هفت": 7, "هشت": 8, "نه": 9,
}
TENS = {
    "ده": 10, "یازده": 11, "دوازده": 12, "سیزده": 13,
    "چهارده": 14, "پانزده": 15, "شانزده": 16, "هفده": 17,
    "هجده": 18, "نوزده": 19, "بیست": 20, "سی": 30,
    "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70,
    "هشتاد": 80, "نود": 90,
}
NUMBER_WORDS = {**ONES, **TENS, "صد": 100}


def parse_number_phrase(value: str) -> Optional[int]:
    """Parse an integer from digits or common Persian words (e.g. بیست و پنج)."""
    value = normalize(value)
    digit_match = re.search(r"(?<!\S)\d+(?!\S)", value)
    if digit_match:
        return int(digit_match.group())

    tokens = value.split()
    # Find the first run of number words, allowing conjunction 'و'.
    for start, token in enumerate(tokens):
        if token not in NUMBER_WORDS:
            continue
        total = 0
        current = NUMBER_WORDS[token]
        i = start + 1
        if current == 100:
            total = 100
            current = 0
        elif current >= 20 and current < 100:
            total = current
            current = 0
        else:
            total = current
            current = 0

        consumed = 1
        while i < len(tokens):
            if tokens[i] == "و" and i + 1 < len(tokens) and tokens[i + 1] in NUMBER_WORDS:
                nxt = NUMBER_WORDS[tokens[i + 1]]
                if total >= 100:
                    if 0 < nxt < 100:
                        total += nxt
                    else:
                        break
                elif total >= 20 and total < 100 and nxt in ONES.values():
                    total += nxt
                elif total < 20 and nxt >= 20:
                    total += nxt
                else:
                    break
                i += 2
                consumed += 2
            elif tokens[i] in NUMBER_WORDS:
                nxt = NUMBER_WORDS[tokens[i]]
                if total >= 100 and 0 < nxt < 100:
                    total += nxt
                    i += 1
                    consumed += 1
                elif total >= 20 and total < 100 and nxt in ONES.values():
                    total += nxt
                    i += 1
                    consumed += 1
                else:
                    break
            else:
                break
        if consumed:
            return total
    return None


def _number_before_unit(text: str, unit: str) -> Optional[int]:
    """Get number immediately before a unit; supports digits and Persian words."""
    # Digits first.
    m = re.search(rf"(?<!\S)(\d+)\s*{re.escape(unit)}(?!\S)", text)
    if m:
        return int(m.group(1))

    # Walk backwards from the unit and isolate only the contiguous number phrase.
    # This avoids reading '2 ساعت و 30 دقیقه' as 2 minutes.
    tokens = text.split()
    for i, token in enumerate(tokens):
        if token != unit or i == 0:
            continue
        j = i - 1
        if tokens[j] not in NUMBER_WORDS and not tokens[j].isdigit():
            continue
        start = j
        while start >= 2 and tokens[start - 1] == "و" and (
            tokens[start - 2] in NUMBER_WORDS or tokens[start - 2].isdigit()
        ):
            start -= 2
        return parse_number_phrase(" ".join(tokens[start:i]))
    return None


# ---------------------------------------------------------------------------
# ENTITY REGISTRY
# Add aliases here; output IDs remain stable for firmware.
# ---------------------------------------------------------------------------
DEVICES: dict[str, list[str]] = {
    "light": ["چراغ روشنایی", "چراغ برق", "چراغ", "لامپ", "روشنایی", "نور"],
    "fan": ["پنکه", "هواکش", "فن"],
    "ac": ["کولر گازی", "کولر آبی", "اسپلیت", "تهویه", "کولر"],
    "window": ["پنجره"],
    "curtain": ["پرده", "کرکره"],
    "lock": ["قفل در", "قفل"],
    # Bare 'در' is deliberately handled separately to avoid matching the
    # preposition in sentences such as 'چراغ در اتاق را روشن کن'.
    "door": ["درب ورودی", "درب حیاط", "درب اتاق", "درب", "در ورودی", "در حیاط", "در پارکینگ"],
    "clock": ["ساعت"],
    "alarm": ["آلارم", "هشدار", "زنگ بیدارباش", "زنگ ساعت"],
    "timer": ["تایمر", "شمارش معکوس"],
}

LOCATIONS: dict[str, list[str]] = {
    "bedroom": ["اتاق خواب", "اتاق", "خواب"],
    "living_room": ["اتاق نشیمن", "اتاق پذیرایی", "نشیمن", "پذیرایی", "هال", "حال", "سالن"],
    "kitchen": ["آشپزخانه", "آشپزخونه", "مطبخ"],
    "bathroom": ["حمام", "دستشویی", "سرویس بهداشتی", "سرویس"],
    "hallway": ["راه پله", "راهرو"],
    "entrance": ["ورودی", "دم در", "جلوی در"],
    "outdoor": ["فضای بیرون", "بیرون", "حیاط", "محوطه"],
    "balcony": ["بالکن", "تراس", "ایوان"],
    "office": ["اتاق کار", "دفتر"],
    "garage": ["پارکینگ", "گاراژ"],
}


def find_entities(text: str, registry: dict[str, list[str]]) -> list[str]:
    candidates = sorted(
        ((entity, alias) for entity, aliases in registry.items() for alias in aliases),
        key=lambda item: len(item[1]), reverse=True,
    )
    occupied: list[tuple[int, int]] = []
    found: list[str] = []
    for entity, alias in candidates:
        for match in re.finditer(rf"(?<!\S){re.escape(alias)}(?!\S)", text):
            start, end = match.span()
            if any(start < old_end and end > old_start for old_start, old_end in occupied):
                continue
            occupied.append((start, end))
            if entity not in found:
                found.append(entity)
    return found


# ---------------------------------------------------------------------------
# ACTION ALIASES
# ---------------------------------------------------------------------------
ACTIONS: dict[str, list[str]] = {
    "off": ["خاموش کن", "خاموشش کن", "خاموش", "قطع کن", "قطعش کن", "غیرفعال کن"],
    "on": ["روشن کن", "روشنش کن", "روشن", "وصل کن", "فعال کن", "راه بنداز"],
    "open": ["باز کن", "بازش کن", "باز"],
    "close": ["ببند", "ببندش", "بسته کن", "بسته"],
    "increase": ["زیاد کن", "بیشتر کن", "بالا ببر", "افزایش بده"],
    "decrease": ["کم کن", "کمتر کن", "پایین بیار", "کاهش بده"],
    "half": ["نصف کن", "نصفش کن", "نصف"],
    "status": ["وضعیت", "وضعیتش", "چقدر است", "چند است", "روشنه", "خاموشه"],
    "stop": ["متوقف کن", "توقف کن", "نگه دار", "بایست"],
    "set": ["تنظیم کن", "تنظیمش کن", "قرار بده", "بذار روی", "بگذار روی", "بذار", "بگذار", "تنظیم"],
    "unlock": ["قفل را باز کن", "قفل رو باز کن", "قفل در را باز کن", "قفل در رو باز کن", "قفل گشا", "قفل رو باز"],
    "lock": ["قفل کن", "قفلش کن", "در را قفل کن", "در رو قفل کن"],
}

WEEKDAYS = {
    "دوشنبه": 0, "سه شنبه": 1, "چهارشنبه": 2, "پنجشنبه": 3,
    "جمعه": 4, "شنبه": 5, "یکشنبه": 6,
}
JALALI_MONTHS = {
    "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4,
    "مرداد": 5, "شهریور": 6, "مهر": 7, "آبان": 8,
    "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12,
}
ORDINAL_DAYS = {
    "اول": 1, "یکم": 1, "دوم": 2, "سوم": 3, "چهارم": 4,
    "پنجم": 5, "ششم": 6, "هفتم": 7, "هشتم": 8, "نهم": 9,
    "دهم": 10, "یازدهم": 11, "دوازدهم": 12, "سیزدهم": 13,
    "چهاردهم": 14, "پانزدهم": 15, "شانزدهم": 16,
    "هفدهم": 17, "هجدهم": 18, "نوزدهم": 19, "بیستم": 20,
    "بیست و یکم": 21, "بیست و دوم": 22, "بیست و سوم": 23,
    "بیست و چهارم": 24, "بیست و پنجم": 25, "بیست و ششم": 26,
    "بیست و هفتم": 27, "بیست و هشتم": 28, "بیست و نهم": 29,
    "سی ام": 30, "سی و یکم": 31,
}


# ---------------------------------------------------------------------------
# DATE / TIME / DURATION PARSING
# ---------------------------------------------------------------------------
def local_now() -> datetime:
    return datetime.now(LOCAL_TZ)


def _parse_hhmm(value: str) -> time:
    h, m = map(int, value.split(":"))
    return time(h, m)


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> date:
    """Convert a modern Jalali date to Gregorian using integer arithmetic."""
    if not 1 <= jm <= 12 or jd < 1 or jd > (31 if jm <= 6 else 30):
        raise ValueError("invalid Jalali date")
    jy += 1595
    days = -355668 + (365 * jy) + (jy // 33) * 8 + ((jy % 33 + 3) // 4) + jd
    if jm < 7:
        days += (jm - 1) * 31
    else:
        days += ((jm - 7) * 30) + 186

    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        gy += 100 * ((days - 1) // 36524)
        days = (days - 1) % 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1

    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_lengths = [31, 29 if leap else 28, 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31]
    gm = 1
    while gm <= 12 and gd > month_lengths[gm - 1]:
        gd -= month_lengths[gm - 1]
        gm += 1
    if gm > 12:
        raise ValueError("invalid Jalali date")
    result = date(gy, gm, gd)
    if gregorian_to_jalali(result) != (jy - 1595, jm, jd):
        raise ValueError("invalid Jalali date")
    return result


def gregorian_to_jalali(gdate: date) -> tuple[int, int, int]:
    """Convert Gregorian date to Jalali (year, month, day)."""
    g_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy, gm, gd = gdate.year, gdate.month, gdate.day
    gy2 = gy + 1 if gm > 2 else gy
    days = (355666 + 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100
            + (gy2 + 399) // 400 + gd + g_days[gm - 1])
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def parse_explicit_date(text: str) -> tuple[Optional[date], Optional[str]]:
    """
    Parse Gregorian YYYY-MM-DD, numeric Jalali YYYY/MM/DD, and common Jalali
    month phrases such as '25 آبان 1405' or 'بیست و پنجم آبان'.
    Jalali conversion requires the optional jdatetime package.
    """
    m = re.search(r"(?<!\d)(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)", text)
    if m:
        year, month, day = map(int, m.groups())
        # Current-era Gregorian years are > 1600; Jalali years are normally
        # 1300..1499. This keeps ISO dates such as 2026-11-25 Gregorian.
        if 1300 <= year < 1700:
            try:
                return jalali_to_gregorian(year, month, day), None
            except ValueError:
                return None, "تاریخ شمسی نامعتبر است."
        try:
            return date(year, month, day), None
        except ValueError:
            return None, "تاریخ میلادی نامعتبر است."

    def parse_day_of_month(prefix: str) -> Optional[int]:
        words = normalize(prefix).split()
        # Check only the suffix adjacent to the month name, so an earlier
        # hour or date elsewhere in the sentence cannot be mistaken for day.
        for width in (3, 2, 1):
            if len(words) < width:
                continue
            suffix = " ".join(words[-width:])
            if suffix in ORDINAL_DAYS:
                return ORDINAL_DAYS[suffix]
            number = parse_number_phrase(suffix)
            if number is not None and 1 <= number <= 31:
                return number
        return None

    for month_name, month_number in JALALI_MONTHS.items():
        # Day before month: '25 آبان 1405' / 'بیست و پنجم آبان'
        match = re.search(rf"(.+?)\s+{re.escape(month_name)}(?:\s+(\d{{4}}))?(?!\S)", text)
        if not match:
            # Month before day: 'آبان 25 1405'
            match = re.search(rf"{re.escape(month_name)}\s+(\d{{1,2}})(?:\s+(\d{{4}}))?(?!\S)", text)
            if match:
                day_value = parse_number_phrase(match.group(1))
                year_value = int(match.group(2)) if match.group(2) else None
            else:
                continue
        else:
            day_phrase = match.group(1).strip()
            day_value = parse_day_of_month(day_phrase)
            year_value = int(match.group(2)) if match.group(2) else None
        if day_value is None or not 1 <= day_value <= 31:
            return None, "روز ماه شمسی معتبر تشخیص داده نشد."
        if year_value is None:
            year_value = gregorian_to_jalali(datetime.now(LOCAL_TZ).date())[0]
        try:
            gregorian = jalali_to_gregorian(year_value, month_number, day_value)
        except ValueError:
            return None, "تاریخ شمسی نامعتبر است."
        # With no year, interpret a date earlier than today as next year's date.
        if match.group(2) is None and gregorian < datetime.now(LOCAL_TZ).date():
            try:
                gregorian = jalali_to_gregorian(year_value + 1, month_number, day_value)
            except ValueError:
                return None, "تاریخ شمسی نامعتبر است."
        return gregorian, None
    return None, None


def _day_period(text: str) -> Optional[str]:
    if any(x in text for x in ("صبح", "بامداد")):
        return "morning"
    if any(x in text for x in ("ظهر", "نیمروز")):
        return "noon"
    if any(x in text for x in ("بعدازظهر", "بعد از ظهر", "عصر", "غروب")):
        return "afternoon"
    if any(x in text for x in ("شب", "شبانه")):
        return "night"
    return None


def parse_clock_time(text: str) -> tuple[Optional[time], Optional[str], bool]:
    """Return (time, error, was_explicit)."""
    # 07:30 or 7:30
    m = re.search(r"(?<!\d)(\d{1,2})\s*:\s*(\d{2})(?!\d)", text)
    if m:
        h, minute = map(int, m.groups())
        if h > 23 or minute > 59:
            return None, "ساعت نامعتبر است.", True
        period = _day_period(text)
        if h <= 12 and period in ("afternoon", "night") and h < 12:
            h += 12
        elif h == 12 and period in ("morning", "night"):
            h = 0
        return time(h, minute), None, True

    # A clock time must follow the word 'ساعت' to avoid mistaking room numbers
    # or dates for hours.
    # Numeric or word hour, optionally phrased as 'ساعت رو روی هفت کوک کن'.
    m = re.search(
        r"ساعت(?:\s+رو|\s+را)?(?:\s+روی)?\s+(\d{1,2}|[آ-ی]+)"
        r"(?:\s+و\s+(نیم|\d{1,2}|[آ-ی]+)(?:\s*دقیقه)?)?",
        text,
    )
    if m:
        h = parse_number_phrase(m.group(1))
        if h is None:
            m = None
        else:
            minute_token = m.group(2)
            minute = 0
            if minute_token == "نیم":
                minute = 30
            elif minute_token:
                parsed = parse_number_phrase(minute_token)
                if parsed is None:
                    return None, "دقیقه ساعت قابل تشخیص نیست.", True
                minute = parsed
            if h > 23 or minute > 59:
                return None, "ساعت نامعتبر است.", True
            period = _day_period(text)
            if h <= 12 and period in ("afternoon", "night") and h < 12:
                h += 12
            elif h == 12 and period == "morning":
                h = 0
            elif h == 12 and period == "night":
                h = 0
            return time(h, minute), None, True

    # Word-number hour: 'ساعت هفت و نیم' / 'ساعت هفت و سی دقیقه'.
    m = re.search(
        r"ساعت(?:\s+رو|\s+را)?(?:\s+روی)?\s+([آ-ی]+)"
        r"(?:\s+و\s+(نیم|[آ-ی]+)(?:\s*دقیقه)?)?",
        text,
    )
    if m:
        h = parse_number_phrase(m.group(1))
        if h is not None:
            minute_token = m.group(2)
            minute = 0 if not minute_token else (30 if minute_token == "نیم" else parse_number_phrase(minute_token))
            if minute is None or h > 23 or minute > 59:
                return None, "ساعت نامعتبر است.", True
            period = _day_period(text)
            if h <= 12 and period in ("afternoon", "night") and h < 12:
                h += 12
            elif h == 12 and period == "morning":
                h = 0
            elif h == 12 and period == "night":
                h = 0
            return time(h, minute), None, True

    # Hour followed by a day-period without the word 'ساعت': 'فردا 7 صبح'.
    m = re.search(
        r"(?<!\S)(\d{1,2}|[آ-ی]+)\s*(صبح|بامداد|ظهر|بعدازظهر|بعد از ظهر|عصر|غروب|شب)(?!\S)",
        text,
    )
    if m:
        h = parse_number_phrase(m.group(1))
        if h is not None:
            period_word = m.group(2)
            if h > 23:
                return None, "ساعت نامعتبر است.", True
            if period_word in ("عصر", "غروب", "شب", "بعدازظهر", "بعد از ظهر") and 1 <= h < 12:
                h += 12
            elif period_word == "شب" and h == 12:
                h = 0
            elif period_word in ("صبح", "بامداد") and h == 12:
                h = 0
            return time(h, 0), None, True

    # Day-period phrase without a numeric hour: configurable default.
    period = _day_period(text)
    defaults = {
        "morning": MORNING_DEFAULT,
        "noon": "12:00",
        "afternoon": AFTERNOON_DEFAULT,
        "night": NIGHT_DEFAULT,
    }
    if period:
        return _parse_hhmm(defaults[period]), None, False
    return None, None, False


def parse_duration_seconds(text: str) -> Optional[int]:
    """Parse combined durations such as '2 ساعت و 30 دقیقه' or 'ده دقیقه'."""
    total = 0
    found = False
    for unit, multiplier in (("روز", 86400), ("ساعت", 3600), ("دقیقه", 60), ("ثانیه", 1)):
        n = _number_before_unit(text, unit)
        if n is not None:
            total += n * multiplier
            found = True
    # 'نیم ساعت' may follow a whole-hour amount: 'یک ساعت و نیم'.
    if "نیم ساعت" in text or "ساعت و نیم" in text:
        total += 1800
        found = True
    if not found:
        return None
    if total <= 0 or total > 366 * 86400:
        return None
    return total


def _parse_relative_days(text: str) -> Optional[int]:
    if "پس فردا" in text:
        return 2
    if "فردا" in text or "روز بعد" in text:
        return 1
    # e.g. سه روز دیگه / 5 روز بعد / بعد از سه روز
    m = re.search(r"(?:بعد از|پس از)\s*(.+?)\s+روز", text)
    if m:
        return parse_number_phrase(m.group(1))
    m = re.search(r"(.+?)\s+روز\s+(?:دیگه|دیگر|بعد)", text)
    if m:
        return parse_number_phrase(m.group(1))
    return None


def _next_weekday(today: date, weekday: int, clock: Optional[time] = None,
                  now: Optional[datetime] = None) -> date:
    days = (weekday - today.weekday()) % 7
    candidate_date = today + timedelta(days=days)
    if clock is not None and now is not None:
        candidate_dt = datetime.combine(candidate_date, clock, tzinfo=LOCAL_TZ)
        if candidate_dt <= now:
            candidate_date += timedelta(days=7)
    elif days == 0:
        candidate_date += timedelta(days=7)
    return candidate_date


def parse_schedule(
    text: str,
    now: Optional[datetime] = None,
    *,
    allow_day_period_default: bool = True,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Parse one-time, timer, daily/weekly recurrence and dated schedules."""
    now = now or local_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)
    now = now.astimezone(LOCAL_TZ)

    # Recurrence (simple daily / weekly) takes precedence over one-shot.
    is_daily = any(x in text for x in ("هر روز", "روزانه", "هرشب", "هر شب"))
    is_weekly = any(x in text for x in ("هر هفته", "هفتگی"))
    interval_days = None
    interval_match = re.search(r"هر\s+(\d+|[آ-ی]+(?:\s+و\s+[آ-ی]+)?)\s+روز", text)
    if interval_match:
        interval_days = parse_number_phrase(interval_match.group(1))
        if interval_days is not None:
            if not 1 <= interval_days <= 366:
                return None, "فاصله تکرار باید بین ۱ تا ۳۶۶ روز باشد."
            is_daily = True
    weekday_name = next((name for name in sorted(WEEKDAYS, key=len, reverse=True) if contains_phrase(text, name)), None)
    if weekday_name and (is_weekly or "هر" in text):
        is_weekly = True

    # A duration used by an explicit timer is parsed elsewhere as a timer.
    explicit_date, date_error = parse_explicit_date(text)
    if date_error:
        return None, date_error

    clock, clock_error, explicit_clock = parse_clock_time(text)
    if clock_error:
        return None, clock_error

    if is_daily or is_weekly:
        if clock is None:
            return None, "برای برنامه تکرارشونده، ساعت را هم مشخص کن."
        start_date = explicit_date or now.date()
        if is_weekly and weekday_name and explicit_date is None:
            start_date = _next_weekday(now.date(), WEEKDAYS[weekday_name], clock, now)
        if datetime.combine(start_date, clock, tzinfo=LOCAL_TZ) <= now:
            if explicit_date:
                return None, "زمان شروع برنامه تکرارشونده در گذشته است."
            start_date = start_date + timedelta(days=1)
        schedule: dict[str, Any] = {
            "kind": "recurring",
            "rule": "weekly" if is_weekly else ("interval_days" if interval_days else "daily"),
            "time": clock.strftime("%H:%M:%S"),
            "timezone": TIMEZONE_NAME,
            "start_date": start_date.isoformat(),
        }
        if is_weekly:
            if weekday_name:
                schedule["weekday"] = WEEKDAYS[weekday_name]
                schedule["weekday_name"] = weekday_name
            else:
                schedule["weekday"] = start_date.weekday()
        if interval_days:
            schedule["interval_days"] = interval_days
        # 'برای 3 روز' / 'به مدت 3 روز' limits a daily recurrence.
        m = re.search(r"(?:برای|به مدت|تا)\s*(\d+|[آ-ی]+(?:\s+و\s+[آ-ی]+)?)\s*روز", text)
        if m:
            count = parse_number_phrase(m.group(1))
            if count is not None:
                if not 1 <= count <= 366:
                    return None, "تعداد روزهای تکرار باید بین ۱ تا ۳۶۶ باشد."
                schedule["occurrences"] = count
        return schedule, None

    # Resolve calendar date before relative durations so 'سه روز دیگر ساعت 8'
    # means 08:00 three days from now, not now + 3 days + 8 hours.
    target_date = explicit_date
    if target_date is None and "امروز" in text:
        target_date = now.date()
    relative_days = _parse_relative_days(text) if target_date is None else None
    if relative_days is not None:
        if not 0 <= relative_days <= 3660:
            return None, "فاصله روزها خارج از محدوده مجاز است."
        target_date = now.date() + timedelta(days=relative_days)

    # Relative duration: 'بعد از 2 ساعت و 30 دقیقه'. If a calendar day was
    # explicitly stated together with a clock time, use the calendar branch.
    relative_duration = parse_duration_seconds(text)
    has_relative_marker = any(x in text for x in ("بعد", "دیگه", "دیگر", "پس از"))
    has_calendar_day = explicit_date is not None or relative_days is not None or any(
        x in text for x in ("امروز", "فردا", "پس فردا")
    )
    if relative_duration is not None and has_relative_marker and not (has_calendar_day and clock is not None):
        when = now + timedelta(seconds=relative_duration)
        return {
            "kind": "once",
            "execute_at": when.isoformat(timespec="seconds"),
            "execute_at_utc": when.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "timezone": TIMEZONE_NAME,
            "delay_seconds": relative_duration,
        }, None

    if target_date is None and weekday_name:
        target_date = _next_weekday(now.date(), WEEKDAYS[weekday_name], clock, now)

    # A clock-only alarm means the next occurrence of that clock time.
    if target_date is None and clock is not None:
        candidate = datetime.combine(now.date(), clock, tzinfo=LOCAL_TZ)
        if candidate <= now:
            candidate += timedelta(days=1)
        return {
            "kind": "once",
            "execute_at": candidate.isoformat(timespec="seconds"),
            "execute_at_utc": candidate.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "timezone": TIMEZONE_NAME,
        }, None

    if target_date is None:
        return None, None

    if clock is None:
        if allow_day_period_default and _day_period(text):
            clock, _, _ = parse_clock_time(text)
        elif relative_days is not None and relative_days >= 2:
            # For 'یادم بنداز سه روز دیگه' use the same local wall-clock time.
            clock = now.timetz().replace(tzinfo=None)
        else:
            return None, "تاریخ مشخص است اما ساعت دقیق گفته نشده؛ ساعت را هم بیان کن."

    candidate = datetime.combine(target_date, clock, tzinfo=LOCAL_TZ)
    if candidate <= now:
        return None, "زمان تعیین‌شده گذشته است؛ تاریخ یا ساعت آینده را مشخص کن."

    return {
        "kind": "once",
        "execute_at": candidate.isoformat(timespec="seconds"),
        "execute_at_utc": candidate.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "timezone": TIMEZONE_NAME,
    }, None


# ---------------------------------------------------------------------------
# COMMAND RESULT
# ---------------------------------------------------------------------------
@dataclass
class Command:
    schema_version: int = 1
    command_id: str = ""
    status: str = "UNKNOWN"
    domain: Optional[str] = None
    device: Optional[str] = None
    location: Optional[str] = None
    action: Optional[str] = None
    parameter: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    schedule: Optional[dict[str, Any]] = None
    confirmation_required: bool = False
    original_text: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------------------------
class PersianCommandEngine:
    def __init__(self):
        self.timezone = TIMEZONE_NAME

    def _result(self, status: str, original: str, message: str,
                **kwargs: Any) -> Command:
        return Command(
            command_id=str(uuid.uuid4()), status=status,
            original_text=original, message=message, **kwargs
        )

    @staticmethod
    def _negative_or_ambiguous(text: str) -> bool:
        # Conservative handling: never execute a sentence containing a clear
        # prohibition/negation token. More sophisticated negation can be added.
        return any(contains_phrase(text, x) for x in (
            "نکن", "نذار", "نگذار", "نمیخوام", "نمی خواهم", "نمیخواهم"
        ))

    @staticmethod
    def _device_action(text: str, device: str, action: str):
        """Return action, parameter, value, unit, error."""
        if device == "ac" and any(x in text for x in ("دما", "درجه", "حرارت")):
            if action in ("increase", "decrease"):
                delta = parse_number_phrase(text)
                return ("temperature_increase" if action == "increase" else "temperature_decrease",
                        "temperature", delta, "celsius", None)
            value = parse_number_phrase(text)
            if value is None or not 16 <= value <= 30:
                return action, "temperature", value, "celsius", "دمای کولر باید بین ۱۶ تا ۳۰ درجه باشد."
            return "temperature_set", "temperature", value, "celsius", None

        parameter = "brightness" if device == "light" else "fan_speed"
        if action == "on":
            requested = PersianCommandEngine._extract_percent(text)
            if requested is not None:
                if not 0 <= requested <= 100:
                    return action, parameter, requested, "percent", "مقدار باید بین صفر تا صد باشد."
                return parameter + "_set", parameter, requested, "percent", None
            return action, None, None, None, None
        if action in ("off", "status", "stop"):
            return action, None, None, None, None
        if action == "half":
            return parameter + "_set", parameter, 50, "percent", None
        if action == "set":
            value = PersianCommandEngine._extract_percent(text)
            if value is None:
                value = parse_number_phrase(text)
            if value is None or not 0 <= value <= 100:
                return action, parameter, value, "percent", "مقدار باید بین صفر تا صد باشد."
            return parameter + "_set", parameter, value, "percent", None
        if action in ("increase", "decrease"):
            delta = PersianCommandEngine._extract_percent(text)
            return parameter + "_" + action, parameter, delta, "percent", None
        return action, None, None, None, "عملیات برای این دستگاه پشتیبانی نمی‌شود."

    @staticmethod
    def _extract_percent(text: str) -> Optional[int]:
        if "نصف" in text or "نیمه" in text:
            return 50
        if "سه چهارم" in text:
            return 75
        if "ربع" in text or "یک چهارم" in text:
            return 25
        m = re.search(r"(?<!\S)(\d{1,3})\s*(?:درصد|%)", text)
        if m:
            return int(m.group(1))
        word_percent = _number_before_unit(text, "درصد")
        if word_percent is not None:
            return word_percent
        if "صد درصد" in text:
            return 100
        return None

    @staticmethod
    def _detect_action(text: str, device: str) -> Optional[str]:
        if device == "lock":
            if (any(contains_phrase(text, x) for x in ACTIONS["unlock"]) or
                    ("قفل" in text and any(contains_phrase(text, x) for x in ACTIONS["open"]))):
                return "unlock"
            if any(contains_phrase(text, x) for x in ACTIONS["lock"]):
                return "lock"
        order = ("status", "stop", "open", "close", "off", "on", "half",
                 "increase", "decrease", "set")
        for action in order:
            if any(contains_phrase(text, x) for x in ACTIONS[action]):
                return action
        if ("%" in text or "درصد" in text or
                ("درجه" in text and re.search(r"\d+", text)) or
                ("روی" in text and re.search(r"\d+", text))):
            return "set"
        return None

    def parse(self, raw_text: str, now: Optional[datetime] = None) -> Command:
        text = normalize(raw_text)
        if not text:
            return self._result("UNKNOWN", raw_text, "متن ورودی خالی است.")
        if self._negative_or_ambiguous(text):
            return self._result(
                "NEEDS_CONFIRMATION", raw_text,
                "جمله منفی یا منع اجرای عمل تشخیص داده شد؛ فرمان اجرا نشد."
            )

        # Timer, alarm and reminder are separate scheduler commands.
        is_reminder = any(x in text for x in ("یادآوری", "یادآور", "یادم بنداز", "یادآوری کن"))
        is_alarm = any(x in text for x in ("آلارم", "هشدار", "زنگ بیدارباش", "زنگ ساعت", "کوک"))
        is_timer = "تایمر" in text or "شمارش معکوس" in text

        if is_timer:
            duration = parse_duration_seconds(text)
            if duration is None:
                return self._result("NEEDS_CONFIRMATION", raw_text, "مدت تایمر مشخص یا معتبر نیست.")
            now_local = now or local_now()
            if now_local.tzinfo is None:
                now_local = now_local.replace(tzinfo=LOCAL_TZ)
            else:
                now_local = now_local.astimezone(LOCAL_TZ)
            expires = now_local + timedelta(seconds=duration)
            return self._result(
                "ACCEPTED", raw_text, "تایمر آماده ثبت است.",
                domain="timer", device="timer", action="start",
                value={"duration_seconds": duration},
                schedule={
                    "kind": "timer", "duration_seconds": duration,
                    "expires_at": expires.isoformat(timespec="seconds"),
                    "expires_at_utc": expires.astimezone(timezone.utc).isoformat(timespec="seconds"),
                    "timezone": TIMEZONE_NAME,
                },
            )

        # Clock winding / alarm phrases such as 'ساعت را روی 7 کوک کن'.
        if is_alarm or ("بیدار" in text and "ساعت" in text):
            schedule, error = parse_schedule(text, now)
            if error:
                return self._result("NEEDS_CONFIRMATION", raw_text, error)
            if schedule is None:
                return self._result("NEEDS_CONFIRMATION", raw_text, "زمان هشدار مشخص نیست.")
            return self._result(
                "ACCEPTED", raw_text, "هشدار آماده ثبت است.",
                domain="alarm", device="alarm", action="set", schedule=schedule,
            )

        if is_reminder:
            schedule, error = parse_schedule(text, now)
            if error:
                return self._result("NEEDS_CONFIRMATION", raw_text, error)
            if schedule is None:
                return self._result("NEEDS_CONFIRMATION", raw_text, "زمان یادآوری مشخص نیست.")
            reminder_text = raw_text.strip()
            reminder_text = re.sub(r"^(لطفا\s*)?(برای من\s*)?یادآوری( کن)?\s*", "", reminder_text)
            reminder_text = re.sub(r"^یادم بنداز\s*", "", reminder_text)
            return self._result(
                "ACCEPTED", raw_text, "یادآور آماده ثبت است.",
                domain="reminder", device="reminder", action="create",
                value={"text": reminder_text or raw_text.strip()}, schedule=schedule,
            )

        # Find device and location entities.
        devices = find_entities(text, DEVICES)
        # 'فن کولر' means the AC's internal fan, not a separate room fan.
        if "ac" in devices and "fan" in devices and any(
            phrase in text for phrase in ("فن کولر", "فن اسپلیت", "فن تهویه")
        ):
            devices = [d for d in devices if d != "fan"]
        # 'قفل در ورودی را باز کن' is an unlock request, not two devices.
        if "lock" in devices and "قفل" in text:
            devices = ["lock"]
        # 'ساعت' as an appliance entity is not used for time setting here.
        devices = [d for d in devices if d not in ("clock", "alarm", "timer")]

        # Special handling for generic bare 'در': only classify as a door when
        # no other device was identified and the sentence has open/close intent.
        if not devices and re.search(r"(?<!\S)در(?!\S)", text):
            if any(contains_phrase(text, x) for x in ACTIONS["open"] + ACTIONS["close"]):
                devices = ["door"]

        locations = find_entities(text, LOCATIONS)
        if len(devices) == 0:
            # A clock-only command is treated as an alarm, not system clock set.
            if "ساعت" in text and any(x in text for x in ("تنظیم", "بذار", "بگذار", "روی")):
                schedule, error = parse_schedule(text, now)
                if error:
                    return self._result("NEEDS_CONFIRMATION", raw_text, error)
                if schedule:
                    return self._result("ACCEPTED", raw_text, "هشدار آماده ثبت است.",
                                        domain="alarm", device="alarm", action="set", schedule=schedule)
            return self._result("UNKNOWN", raw_text, "وسیله یا نوع فرمان مشخص نیست.")
        if len(devices) != 1:
            return self._result("NEEDS_CONFIRMATION", raw_text,
                                "چند دستگاه متفاوت تشخیص داده شد؛ هر فرمان را جداگانه بیان کن.")
        if len(locations) > 1:
            return self._result("NEEDS_CONFIRMATION", raw_text,
                                "چند محل متفاوت تشخیص داده شد؛ محل دقیق را مشخص کن.")

        device = devices[0]
        location = locations[0] if locations else "default"
        action = self._detect_action(text, device)
        if action is None:
            return self._result("UNKNOWN", raw_text, "عملیات موردنظر مشخص نیست.")

        parameter = None
        value = None
        unit = None
        confirmation = False

        if device == "lock":
            if action not in ("unlock", "lock", "status"):
                return self._result("NEEDS_CONFIRMATION", raw_text, "عملیات قفل معتبر نیست.")
            confirmation = action in ("unlock", "lock")

        elif device in ("door", "window", "curtain"):
            if action not in ("open", "close", "stop", "status", "set", "half"):
                return self._result("NEEDS_CONFIRMATION", raw_text,
                                    "برای این دستگاه فقط باز، بسته، توقف، وضعیت یا تعیین موقعیت مجاز است.")
            if action in ("open", "close") and device == "door":
                confirmation = True
            pct = self._extract_percent(text)
            if action == "half":
                pct = 50
            if action == "set" or pct is not None:
                if pct is None:
                    pct = parse_number_phrase(text)
                if pct is None or not 0 <= pct <= 100:
                    return self._result("NEEDS_CONFIRMATION", raw_text,
                                        "موقعیت باید به‌صورت درصدی بین صفر تا صد مشخص شود.")
                action, parameter, value, unit = "set_position", "position", pct, "percent"

        elif device in ("light", "fan", "ac"):
            action, parameter, value, unit, error = self._device_action(text, device, action)
            if error:
                return self._result("NEEDS_CONFIRMATION", raw_text, error)

        else:
            return self._result("UNKNOWN", raw_text, "این وسیله در رجیستری تعریف نشده است.")

        # Parse optional scheduling phrase attached to a device command:
        # 'فردا ساعت 8 چراغ اتاق روشن کن', 'بعد از 10 دقیقه پنکه را خاموش کن'.
        schedule, schedule_error = parse_schedule(text, now)
        if schedule_error:
            return self._result("NEEDS_CONFIRMATION", raw_text, schedule_error)

        final_status = "NEEDS_CONFIRMATION" if confirmation else "ACCEPTED"
        final_message = (
            "برای اجرای فرمان در یا قفل، تأیید صریح کاربر لازم است."
            if confirmation else "فرمان معتبر است."
        )
        return self._result(
            final_status, raw_text, final_message,
            domain="device", device=device, location=location,
            action=action, parameter=parameter, value=value, unit=unit,
            schedule=schedule, confirmation_required=confirmation,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    engine = PersianCommandEngine()
    print("Persian Offline Voice Command Engine")
    print(f"Timezone: {TIMEZONE_NAME}")
    print("Type exit to quit. Output is one JSON object per command.")
    while True:
        try:
            raw = input("\nفرمان: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.lower() in {"exit", "quit"}:
            break
        result = engine.parse(raw)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
