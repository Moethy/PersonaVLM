import datetime
import json
import os


def load_medication_schedule(path="medication_schedule.json"):
    if not os.path.exists(path):
        return "Er is geen medicatieschema gevonden."

    with open(path, "r", encoding="utf-8") as file:
        schedule = json.load(file)

    now = datetime.datetime.now()
    today = now.date()

    schedule_lines = []
    next_moments = []

    for medication in schedule.get("medications", []):
        name = medication.get("name", "Onbekende medicatie")
        dose = medication.get("dose", "")
        times = medication.get("times", [])
        instructions = medication.get("instructions", "")
        reminder_enabled = medication.get("reminder_enabled", False)

        schedule_lines.append(f"Medicatie: {name}")
        schedule_lines.append(f"Dosering volgens schema: {dose}")
        schedule_lines.append(f"Innamemomenten vandaag: {', '.join(times)}")
        schedule_lines.append(f"Instructie: {instructions}")
        schedule_lines.append(f"Herinnering actief: {'ja' if reminder_enabled else 'nee'}")

        for time_str in times:
            hour, minute = map(int, time_str.split(":"))
            moment = datetime.datetime.combine(today, datetime.time(hour, minute))

            if moment < now:
                moment = moment + datetime.timedelta(days=1)

            next_moments.append({
                "medication": name,
                "dose": dose,
                "time": moment,
                "instructions": instructions,
                "reminder_enabled": reminder_enabled
            })

    if not next_moments:
        return "\n".join(schedule_lines)

    next_moment = min(next_moments, key=lambda item: item["time"])
    minutes_until = int((next_moment["time"] - now).total_seconds() // 60)

    schedule_lines.append("")
    schedule_lines.append("Volgende innamemoment:")
    schedule_lines.append(f"Medicatie: {next_moment['medication']}")
    schedule_lines.append(f"Dosering: {next_moment['dose']}")
    schedule_lines.append(f"Tijd: {next_moment['time'].strftime('%Y-%m-%d %H:%M')}")
    schedule_lines.append(f"Over ongeveer: {minutes_until} minuten")
    schedule_lines.append(f"Instructie: {next_moment['instructions']}")

    return "\n".join(schedule_lines)