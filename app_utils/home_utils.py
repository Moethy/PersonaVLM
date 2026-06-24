import datetime
import html
import re

from app_utils.schedule_utils import get_schedule_table


def extract_name_from_profile(initial_profile: str):
    if not initial_profile:
        return "Maria"

    for line in initial_profile.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)

            if key.strip().lower() == "name":
                name = value.strip()
                return name if name else "Maria"

    return "Maria"


def get_time_greeting():
    current_hour = datetime.datetime.now().hour

    if 5 <= current_hour < 12:
        return "Goedemorgen"

    if 12 <= current_hour < 18:
        return "Goedemiddag"

    return "Goedenavond"


def get_home_header(initial_profile: str):
    name = html.escape(extract_name_from_profile(initial_profile))
    greeting = get_time_greeting()

    return f"""
    <div class="home-header-wrapper">
        <div class="home-title">{greeting} {name}</div>
        <div class="home-subtitle">Hier zie je wat nu belangrijk is</div>
    </div>
    """


def get_row_value(row, index, key):
    if isinstance(row, dict):
        return row.get(key, "")

    if isinstance(row, (list, tuple)) and len(row) > index:
        return row[index]

    return ""


def extract_times_from_text(text):
    if text is None:
        return []

    text = str(text)
    matches = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)

    times = []

    for hour, minute in matches:
        times.append((int(hour), int(minute)))

    return times


def format_home_time(hour, minute):
    return f"{int(hour)}:{int(minute):02d}"


def get_next_medication_from_schedule():
    schedule = get_schedule_table()

    if schedule is None:
        return None

    if hasattr(schedule, "to_dict"):
        rows = schedule.to_dict("records")
    else:
        rows = schedule

    if not rows:
        return None

    now = datetime.datetime.now()
    candidates = []

    for row in rows:
        medication_name = get_row_value(row, 0, "Medicatie")
        dosage = get_row_value(row, 1, "Dosering")
        intake_times = get_row_value(row, 2, "Innamemomenten")
        reminder_active = get_row_value(row, 4, "Herinnering actief")

        active_text = str(reminder_active).strip().lower()
        if active_text in ["nee", "false", "0", "no", "niet actief"]:
            continue

        found_times = extract_times_from_text(intake_times)

        for hour, minute in found_times:
            candidate_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if candidate_time < now:
                candidate_time += datetime.timedelta(days=1)

            candidates.append({
                "time": candidate_time,
                "display_time": format_home_time(hour, minute),
                "medication_name": medication_name,
                "dosage": dosage
            })

    if candidates:
        candidates.sort(key=lambda item: item["time"])
        return candidates[0]

    first_row = rows[0]
    medication_name = get_row_value(first_row, 0, "Medicatie")
    dosage = get_row_value(first_row, 1, "Dosering")
    intake_times = get_row_value(first_row, 2, "Innamemomenten")
    found_times = extract_times_from_text(intake_times)

    if found_times:
        hour, minute = found_times[0]
        display_time = format_home_time(hour, minute)
    else:
        display_time = "Nu"

    return {
        "display_time": display_time,
        "medication_name": medication_name,
        "dosage": dosage
    }


def get_home_medication_card():
    medication = get_next_medication_from_schedule()

    if medication is None:
        return """
        <div class="home-medication-card">
            <div class="home-time">Geen medicatie</div>
            <div class="home-med-name">Er staat nu geen medicatie klaar.</div>

            <div class="home-divider"></div>

            <div class="home-status-row">
                <span class="home-radio-circle"></span>
                <span>Geen actie nodig</span>
            </div>
        </div>
        """

    display_time = html.escape(str(medication.get("display_time", "Nu")))
    medication_name = html.escape(str(medication.get("medication_name", "")))
    dosage = html.escape(str(medication.get("dosage", "")))

    return f"""
    <div class="home-medication-card">
        <div class="home-time">{display_time}</div>
        <div class="home-med-name">{medication_name}</div>
        <div class="home-dosage">{dosage}</div>

        <div class="home-divider"></div>

        <div class="home-status-row">
            <span class="home-radio-circle"></span>
            <span>Nog niet ingenomen</span>
        </div>
    </div>
    """


def update_home_view(initial_profile):
    return (
        get_home_header(initial_profile),
        get_home_medication_card()
    )


def refresh_schedule_and_home():
    return (
        get_schedule_table(),
        get_home_medication_card()
    )