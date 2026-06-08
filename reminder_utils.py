import datetime
import json
import os

def check_reminder_notification():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    reminder_message = check_due_reminders()

    print(f"[REMINDER CHECK] {now} -> {reminder_message if reminder_message else 'geen reminder'}")

    if reminder_message:
        return reminder_message

    return f"Laatst gecontroleerd om {now}. Geen herinnering op dit moment."


def check_snoozed_reminders(snooze_path="snoozed_reminders.json"):
    if not os.path.exists(snooze_path):
        return ""

    with open(snooze_path, "r", encoding="utf-8") as file:
        snoozed_reminders = json.load(file)

    now = datetime.datetime.now()
    due_messages = []
    changed = False

    for reminder in snoozed_reminders:
        if not reminder.get("active", False):
            continue

        snooze_until = datetime.datetime.strptime(
            reminder["snooze_until"],
            "%Y-%m-%d %H:%M"
        )

        if now >= snooze_until:
            due_messages.append(reminder["reminder_message"])
            reminder["active"] = False
            changed = True

    if changed:
        with open(snooze_path, "w", encoding="utf-8") as file:
            json.dump(snoozed_reminders, file, indent=2, ensure_ascii=False)

    if due_messages:
        return "\n\n".join(due_messages)

    return ""



def check_due_reminders(
    schedule_path="medication_schedule.json",
    state_path="reminder_state.json"
):

    snoozed_message = check_snoozed_reminders()
    if snoozed_message:
        return snoozed_message

    if not os.path.exists(schedule_path):
        return ""

    with open(schedule_path, "r", encoding="utf-8") as file:
        schedule = json.load(file)

    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M")

    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as file:
            reminder_state = json.load(file)
    else:
        reminder_state = {}

    shown_today = reminder_state.get(current_date, [])
    due_messages = []

    for medication in schedule.get("medications", []):
        name = medication.get("name", "Onbekende medicatie")
        dose = medication.get("dose", "")
        times = medication.get("times", [])
        instructions = medication.get("instructions", "")
        reminder_enabled = medication.get("reminder_enabled", False)

        if not reminder_enabled:
            continue

        for time_str in times:
            reminder_id = f"{name}_{time_str}"

            if time_str == current_time and reminder_id not in shown_today:
                message = (
                    f"Herinnering: het is tijd om {name} in te nemen. "
                    f"Dosering volgens schema: {dose}. "
                    f"{instructions}"
                )

                due_messages.append(message)
                shown_today.append(reminder_id)

    if due_messages:
        reminder_state[current_date] = shown_today

        with open(state_path, "w", encoding="utf-8") as file:
            json.dump(reminder_state, file, indent=2, ensure_ascii=False)

        return "\n\n".join(due_messages)

    return ""


def save_medication_action(
    reminder_message,
    status,
    log_path="medication_log.json"
):
    now = datetime.datetime.now()

    log_entry = {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "reminder_message": reminder_message
    }

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as file:
            medication_log = json.load(file)
    else:
        medication_log = []

    medication_log.append(log_entry)

    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(medication_log, file, indent=2, ensure_ascii=False)

    return log_entry

def snooze_reminder(
    reminder_message,
    minutes=10,
    snooze_path="snoozed_reminders.json"
):
    now = datetime.datetime.now()
    snooze_time = now + datetime.timedelta(minutes=minutes)

    snooze_entry = {
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "snooze_until": snooze_time.strftime("%Y-%m-%d %H:%M"),
        "reminder_message": reminder_message,
        "active": True
    }

    if os.path.exists(snooze_path):
        with open(snooze_path, "r", encoding="utf-8") as file:
            snoozed_reminders = json.load(file)
    else:
        snoozed_reminders = []

    snoozed_reminders.append(snooze_entry)

    with open(snooze_path, "w", encoding="utf-8") as file:
        json.dump(snoozed_reminders, file, indent=2, ensure_ascii=False)

    return snooze_entry