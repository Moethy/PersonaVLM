import html
import re


def _extract_time(text: str):
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)

    if not match:
        return "Nu"

    hour = int(match.group(1))
    minute = int(match.group(2))

    return f"{hour}:{minute:02d}"


def _extract_medication_name(text: str):
    medication_names = [
        "paracetamol",
        "ibuprofen",
        "vitamine d",
        "amlodipine",
        "pantoprazol",
        "amoxicilline",
    ]

    lower_text = text.lower()

    for medication in medication_names:
        if medication in lower_text:
            return medication.title()

    return "Medicatie"


def _extract_dosage(text: str):
    match = re.search(
        r"\b(\d+(?:[,.]\d+)?\s?(?:mg|mcg|µg|g|ml|ie|tablet|tabletten))\b",
        text,
        re.IGNORECASE
    )

    if not match:
        return ""

    return match.group(1).replace(" ", "")


def _extract_instruction(text: str):
    match = re.search(r"(Neem .+?)(?:\.|$)", text, re.IGNORECASE)

    if match:
        return match.group(1)

    return "Neem uw medicatie zoals aangegeven in het schema."


def build_reminder_popup_html(reminder_message: str):
    safe_message = html.escape(reminder_message or "")

    reminder_time = html.escape(_extract_time(reminder_message))
    medication_name = html.escape(_extract_medication_name(reminder_message))
    dosage = html.escape(_extract_dosage(reminder_message))
    instruction = html.escape(_extract_instruction(reminder_message))

    return f"""
    <div class="reminder-card-content">
        <div class="reminder-time-badge">{reminder_time}</div>

        <div class="reminder-main-section">
            <div class="reminder-small-title">Het is tijd voor uw medicatie</div>
            <div class="reminder-medication-name">{medication_name}</div>
            <div class="reminder-dosage">{dosage}</div>
            <div class="reminder-instruction">{instruction}</div>
        </div>

        <div class="reminder-question-section">
            <div class="reminder-question">Heeft u de medicatie ingenomen?</div>
        </div>

    </div>
    """