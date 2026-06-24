import os
import re

from app_utils.schedule_utils import load_medication_schedule


def load_assistant_instructions():
    path = "prompt_instructions/medication_assistant.txt"

    if not os.path.exists(path):
        print("Prompt instructions file not found.")
        return ""

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    print(f"Loaded prompt instructions: {path} ({len(content)} characters)")
    return content


def load_medication_information(user_query: str):
    info_folder = "medication_info"

    if not os.path.exists(info_folder):
        print("Medication info folder not found.")
        return ""

    query = user_query.lower()

    medication_files = {
        "paracetamol": "paracetamol.txt",
        "ibuprofen": "ibuprofen.txt",
        "vitamine d": "vitamine_d.txt",
        "vitamined": "vitamine_d.txt",
        "amlodipine": "amlodipine.txt",
        "bloeddrukmedicatie": "amlodipine.txt",
        "pantoprazol": "pantoprazol.txt",
        "maagbeschermer": "pantoprazol.txt",
        "amoxicilline": "amoxicilline.txt",
        "antibiotica": "amoxicilline.txt",
        "antibioticakuur": "amoxicilline.txt",
    }

    selected_files = []

    for keyword, filename in medication_files.items():
        if keyword in query and filename not in selected_files:
            selected_files.append(filename)

    if not selected_files:
        print("Geen specifieke medicatie gevonden in de vraag.")
        return ""

    context_parts = []

    for filename in selected_files:
        path = os.path.join(info_folder, filename)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            print(f"Loaded medication file: {filename} ({len(content)} characters)")
            context_parts.append(f"--- BRONBESTAND: {filename} ---\n{content}")
        else:
            print(f"Medication file not found: {filename}")

    return "\n\n".join(context_parts)


def clean_assistant_response(text):
    if not text:
        return ""

    allowed_pattern = r"[^a-zA-ZÀ-ÿ0-9\s.,!?;:()\-/'\"%€+*=&\n]"
    text = re.sub(allowed_pattern, "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def build_agent_prompt(user_query: str):
    instructions = load_assistant_instructions()
    medication_context = load_medication_information(user_query)
    schedule_context = load_medication_schedule()

    return (
        f"{instructions}\n\n"
        f"Medicatie-informatie:\n{medication_context}\n\n"
        f"Medicatieschema en huidige planning:\n{schedule_context}\n\n"
        f"Vraag van gebruiker:\n{user_query}\n\n"
        "Antwoord nu volledig in het Nederlands:"
    )