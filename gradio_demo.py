import json
import os
import subprocess
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import pyaudio
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write

from PersonaVLM import PersonaVLMAgent
from inference import UserConfig
from app_utils.reminder_utils import check_due_reminders, save_medication_action, snooze_reminder
from app_utils.home_utils import (
    get_home_header,
    get_home_medication_card,
    update_home_view,
    refresh_schedule_and_home,
)

from app_utils.context_utils import (
    load_assistant_instructions,
    load_medication_information,
    clean_assistant_response,
    build_agent_prompt,
)


from app_utils.schedule_utils import load_medication_schedule, get_schedule_table

# =========================
# Configuratie
# =========================

STT_AUDIO_FILE = "stt_input.wav"
STT_MIC_DEVICE = None

STT_MAX_DURATION = 15
STT_CHUNK_DURATION = 0.2
STT_SILENCE_DURATION = 1.0
STT_SILENCE_THRESHOLD = 600

DEFAULT_INITIAL_PROFILE = (
    "name: Maria\n"
    "geslacht: vrouw\n"
    "taal: Nederlands\n"
    "voorkeuren: eenvoudige uitleg over medicatie"
)

if os.name == "nt":
    PIPER_EXE = r"piper\piper.exe"
    PIPER_VOICE_MODEL = r"piper\voices\nl\nl_NL\ronnie\medium\nl_NL-ronnie-medium.onnx"
else:
    PIPER_EXE = "piper/piper"
    PIPER_VOICE_MODEL = "piper/voices/nl/nl_NL/ronnie/medium/nl_NL-ronnie-medium.onnx"

PIPER_OUTPUT_WAV = "piper_response.wav"

stt_model = WhisperModel("base", device="cpu", compute_type="int8")


with open("assets/stylesheet.css", "r", encoding="utf-8") as file:
    stylesheet = file.read()


# =========================
# PersonaVLM agent wrapper
# =========================

class PersonaMedicationAgent:
    def __init__(self, initial_profile: str, model=None):
        self.initial_profile_on_creation = initial_profile
        self.config = self.create_user_config(initial_profile)
        self.agent = PersonaVLMAgent.PersonaVLM(self.config, agent_model=model)

    def create_user_config(self, initial_profile: str):
        profile = {}

        for line in initial_profile.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                profile[key.strip()] = value.strip()

        if "name" not in profile:
            raise ValueError("Initial profile must contain a 'name' field.")

        user_name = profile["name"]
        del profile["name"]

        config = UserConfig()
        config.USER_NAME = user_name
        config.PROFILE = profile
        config.DATA_STORAGE_PATH = Path(f"./output/{config.USER_NAME}")
        config.DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

        return config

    def get_latest_memory(self):
        data_json_path = os.path.join(self.config.DATA_STORAGE_PATH, "user_data.json")

        if not os.path.exists(data_json_path):
            return ""

        with open(data_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        sessions = data.get("sessions", [])
        if not sessions:
            return ""

        latest_session = sessions[-1]
        semantic_memory = latest_session.get("semantic_memory", [])
        interactions = latest_session.get("interactions", [])

        if not semantic_memory:
            return ""

        latest_index = len(semantic_memory) - 1
        latest_item = semantic_memory[latest_index]

        interaction_time = "N/A"
        if latest_index < len(interactions) and interactions[latest_index]:
            interaction_time = interactions[latest_index][0].get("time", "N/A")

        content = latest_item.get("content") or "N/A"
        keywords = latest_item.get("keywords") or "N/A"
        analysis = latest_item.get("reason") or "N/A"

        result = "\n".join([
            f"Time: {interaction_time}",
            f"Content: {content}",
            f"Keywords: {keywords}",
            f"Analysis: {analysis}"
        ])

        return result.replace("<img_fill>", "<image>")

    def parse_reasoning_chain(self, reasoning_chain):
        if not reasoning_chain:
            return ""

        steps = []

        for step in reasoning_chain:
            think = step.get("think", "")

            if step.get("action") == "retrieve":
                retrieve_info = step.get("retrieve_info", {})
                keywords = retrieve_info.get("keywords", "")
                start_time = retrieve_info.get("start_time", "None")
                end_time = retrieve_info.get("end_time", "None")

                steps.append(
                    f"<think>{think}</think>\n"
                    f"<retrieve>keywords: {keywords}, start_time: {start_time}, end_time: {end_time}</retrieve>"
                )
            else:
                steps.append(f"<think>{think}</think>")

        return "\n".join(steps)

    def send_message(self, current_time: str, query: str):
        message = {
            "query": query,
            "imgs": "",
            "time": current_time,
        }

        response, personality_info, reasoning_chain = self.agent.send_message(
            message,
            personality_input=None,
            return_response_only=False,
            update=True
        )

        memory_output = self.get_latest_memory()
        reasoning_output = self.parse_reasoning_chain(reasoning_chain)
        personality_output = "\n".join([f"{key}: {value}" for key, value in personality_info.items()])

        return response, memory_output, reasoning_output, personality_output


def get_or_create_agent(initial_profile: str, agent_state: PersonaMedicationAgent):
    if agent_state is None:
        return PersonaMedicationAgent(initial_profile)

    if agent_state.initial_profile_on_creation != initial_profile:
        return PersonaMedicationAgent(
            initial_profile=initial_profile,
            model=agent_state.agent.mllm
        )

    return agent_state


# =========================
# Spraak naar tekst
# =========================

def transcribe_microphone_audio():
    device_info = sd.query_devices(STT_MIC_DEVICE, "input")
    sample_rate = int(device_info["default_samplerate"])

    chunk_samples = int(STT_CHUNK_DURATION * sample_rate)
    max_chunks = int(STT_MAX_DURATION / STT_CHUNK_DURATION)
    silence_chunks_needed = int(STT_SILENCE_DURATION / STT_CHUNK_DURATION)

    frames = []
    has_started_speaking = False
    silent_chunks = 0

    print("Opname gestart. Spreek nu...")

    for _ in range(max_chunks):
        chunk = sd.rec(
            chunk_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=STT_MIC_DEVICE
        )
        sd.wait()

        volume = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        print(f"Volume: {volume:.2f}")

        if volume > STT_SILENCE_THRESHOLD:
            has_started_speaking = True
            silent_chunks = 0
            frames.append(chunk)
        else:
            if has_started_speaking:
                silent_chunks += 1
                frames.append(chunk)

                if silent_chunks >= silence_chunks_needed:
                    print("Stilte gedetecteerd. Opname gestopt.")
                    break

    if not frames:
        return ""

    audio = np.concatenate(frames, axis=0)
    write(STT_AUDIO_FILE, sample_rate, audio)

    segments, info = stt_model.transcribe(STT_AUDIO_FILE, language="nl")
    text = " ".join(segment.text.strip() for segment in segments)

    return text.strip()


def record_voice_input():
    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value="", interactive=False),
        ""
    )

    text = transcribe_microphone_audio()

    if not text:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", interactive=True),
            create_popup("Geen spraak gehoord, probeer opnieuw.", "error")
        )
    else:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=text, interactive=True),
            ""
        )


# =========================
# Tekst naar spraak
# =========================

def speak_response(text):
    if not text or not text.strip():
        return

    subprocess.run(
        [PIPER_EXE, "--model", PIPER_VOICE_MODEL, "--output_file", PIPER_OUTPUT_WAV],
        input=text,
        text=True,
        check=True
    )

    wave_file = wave.open(PIPER_OUTPUT_WAV, "rb")
    audio_player = pyaudio.PyAudio()

    stream = audio_player.open(
        format=audio_player.get_format_from_width(wave_file.getsampwidth()),
        channels=wave_file.getnchannels(),
        rate=wave_file.getframerate(),
        output=True
    )

    data = wave_file.readframes(1024)
    while data:
        stream.write(data)
        data = wave_file.readframes(1024)

    stream.stop_stream()
    stream.close()
    audio_player.terminate()
    wave_file.close()


# =========================
# Context laden
# =========================






# =========================
# Popupmeldingen
# =========================

def create_popup(message, popup_type="error"):
    if not message:
        return ""

    safe_message = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return f'<div class="center_popup popup_{popup_type}">{safe_message}</div>'


# =========================
# Reminderlogica
# =========================

def check_reminders():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    reminder_message = check_due_reminders()

    print(f"[REMINDER CHECK] {now} -> {reminder_message if reminder_message else 'geen reminder'}")

    if reminder_message:
        return (
            gr.update(visible=True),
            reminder_message,
            reminder_message
        )

    return (
        gr.update(),
        gr.update(),
        gr.update()
    )


def mark_reminder_taken(reminder_message):
    save_medication_action(
        reminder_message=reminder_message,
        status="ingenomen"
    )

    return (
        gr.update(visible=False),
        create_popup("Medicatie gemarkeerd als ingenomen.", "success")
    )


def mark_reminder_not_taken(reminder_message):
    save_medication_action(
        reminder_message=reminder_message,
        status="niet_ingenomen"
    )

    snooze_reminder(
        reminder_message=reminder_message,
        minutes=10
    )

    return (
        gr.update(visible=False),
        create_popup("Ik zal de herinnering over 10 minuten opnieuw tonen.", "info")
    )


def handle_reminder_uncertain(reminder_message, initial_profile, current_time, agent_state):
    save_medication_action(
        reminder_message=reminder_message,
        status="twijfel"
    )

    agent = get_or_create_agent(initial_profile, agent_state)

    if not current_time:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    instructions = load_assistant_instructions()
    medication_context = load_medication_information(reminder_message)
    schedule_context = load_medication_schedule()

    uncertainty_prompt = (
        f"{instructions}\n\n"
        f"Medicatie-informatie:\n{medication_context}\n\n"
        f"Medicatieschema en huidige planning:\n{schedule_context}\n\n"
        "Situatie:\n"
        "De gebruiker heeft bij een medicatieherinnering op 'Ik twijfel' gedrukt.\n\n"
        f"Herinnering:\n{reminder_message}\n\n"
        "Geef korte, rustige uitleg over wat de gebruiker nu kan doen. "
        "Leg uit dat de gebruiker het medicatieschema, de verpakking of bijsluiter kan controleren. "
        "Geef geen persoonlijke medische beslissing. "
        "Verwijs bij twijfel naar arts, apotheker of mantelzorger. "
        "Antwoord volledig in het Nederlands."
    )

    response, memory_output, reasoning_output, personality_output = agent.send_message(
        current_time=current_time,
        query=uncertainty_prompt
    )

    response = clean_assistant_response(response)

    return (
        gr.update(visible=False),
        create_popup("Twijfel genoteerd. De medicatie-assistent geeft extra uitleg.", "info"),
        response,
        memory_output,
        reasoning_output,
        personality_output,
        agent
    )


# =========================
# Berichten naar agent
# =========================

def send_user_message(initial_profile, current_time, user_query, agent_state):
    if not user_query or not user_query.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            agent_state,
            gr.update(value="", interactive=True)
        )

    agent = get_or_create_agent(initial_profile, agent_state)

    if not current_time:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    full_prompt = build_agent_prompt(user_query)

    response, memory_output, reasoning_output, personality_output = agent.send_message(
        current_time=current_time,
        query=full_prompt
    )

    response = clean_assistant_response(response)
    speak_response(response)

    return (
        response,
        memory_output,
        reasoning_output,
        personality_output,
        agent,
        gr.update(value="", interactive=True)
    )


def send_voice_message_if_available(initial_profile, current_time, user_query, agent_state):
    if not user_query or not user_query.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            agent_state,
            gr.update(value="", interactive=True)
        )

    return send_user_message(
        initial_profile=initial_profile,
        current_time=current_time,
        user_query=user_query,
        agent_state=agent_state
    )


def go_to_assistant_tab():
    return gr.update(selected="assistant_tab")


def go_to_schedule_tab():
    return gr.update(selected="schedule_tab")


# =========================
# Interface
# =========================

with gr.Blocks(theme=gr.themes.Soft(), css=stylesheet) as demo:
    agent_state = gr.State(value=None)
    current_reminder_state = gr.State(value="")

    with gr.Tabs(selected="home_tab") as main_tabs:

        with gr.Tab("Home", id="home_tab"):
            with gr.Column(elem_id="home_center_column"):
                home_header_output = gr.HTML(
                    value=get_home_header(DEFAULT_INITIAL_PROFILE)
                )

                home_medication_card = gr.HTML(
                    value=get_home_medication_card()
                )

                with gr.Row(equal_height=True, elem_id="home_button_row"):
                    home_question_btn = gr.Button(
                        "❔ Stel een vraag",
                        elem_id="home_question_button"
                    )

                    home_schedule_btn = gr.Button(
                        "📋 Medicatieschema",
                        elem_id="home_schedule_button"
                    )

        with gr.Tab("Assistent", id="assistant_tab"):
            with gr.Column(elem_id="assistant_center_column"):

                with gr.Group(visible=False) as reminder_popup:
                    reminder_output = gr.Textbox(
                        label="Medicatieherinnering",
                        lines=3,
                        interactive=False
                    )

                    with gr.Row():
                        taken_btn = gr.Button(
                            "Ingenomen",
                            variant="primary",
                            elem_id="taken_button"
                        )

                        not_taken_btn = gr.Button(
                            "Niet ingenomen",
                            elem_id="not_taken_button"
                        )

                        unsure_btn = gr.Button(
                            "Ik twijfel",
                            elem_id="unsure_button"
                        )

                action_status_output = gr.HTML(
                    value="",
                    elem_id="popup_output"
                )

                model_response_output = gr.Textbox(
                    label="",
                    show_label=False,
                    lines=8,
                    interactive=False,
                    container=True,
                    elem_id="model_response_box"
                )

                query_input = gr.Textbox(
                    label="",
                    show_label=False,
                    lines=1,
                    placeholder="Stel hier je vraag",
                    container=True,
                    elem_id="query_input_box"
                )

                with gr.Row(equal_height=True, elem_id="input_button_row"):
                    with gr.Column(scale=0, min_width=80):
                        speech_to_text_btn = gr.Button(
                            value="",
                            icon="assets/microphone.svg",
                            elem_id="speech_button",
                            min_width=0,
                            visible=True
                        )

                        speech_listening_btn = gr.Button(
                            value="",
                            icon="assets/microphone.svg",
                            elem_id="speech_button_listening",
                            min_width=0,
                            visible=False,
                            interactive=True
                        )

                    with gr.Column(scale=0, min_width=80):
                        submit_btn = gr.Button(
                            value="",
                            icon="assets/send.svg",
                            variant="primary",
                            elem_id="send_button",
                            min_width=0
                        )

                current_time_input = gr.Textbox(
                    label="Timestamp (Optional)",
                    info="Format: YYYY-MM-DD HH:MM. If empty, the current system time will be used.",
                    lines=3,
                    value=None,
                    visible=False
                )

                memory_output = gr.Textbox(
                    label="Remembering",
                    lines=3,
                    interactive=False,
                    visible=False
                )

                reasoning_output = gr.Textbox(
                    label="Reasoning",
                    lines=3,
                    max_lines=3,
                    interactive=False,
                    visible=False
                )

                personality_output = gr.Textbox(
                    label="Alignment",
                    lines=3,
                    interactive=False,
                    visible=False
                )

        with gr.Tab("Gebruikersprofiel", id="profile_tab"):
            gr.Markdown("## Gebruikersprofiel")

            gr.Markdown(
                "Hier kan het basisprofiel van de gebruiker of persona worden aangepast. "
                "Dit profiel wordt gebruikt door de medicatie-assistent om de toon en uitleg beter af te stemmen."
            )

            initial_profile_input = gr.Textbox(
                label="Initial User Profile",
                lines=8,
                value=DEFAULT_INITIAL_PROFILE,
                interactive=True
            )

        with gr.Tab("Medicatieschema", id="schedule_tab"):
            gr.Markdown("## Medicatieschema")

            schedule_table = gr.Dataframe(
                headers=[
                    "Medicatie",
                    "Dosering",
                    "Innamemomenten",
                    "Instructie",
                    "Herinnering actief"
                ],
                value=get_schedule_table(),
                interactive=False
            )

            refresh_schedule_btn = gr.Button("Schema verversen")

            refresh_schedule_btn.click(
                fn=refresh_schedule_and_home,
                inputs=[],
                outputs=[
                    schedule_table,
                    home_medication_card
                ]
            )

    demo.load(
        fn=update_home_view,
        inputs=[
            initial_profile_input
        ],
        outputs=[
            home_header_output,
            home_medication_card
        ]
    )

    initial_profile_input.change(
        fn=update_home_view,
        inputs=[
            initial_profile_input
        ],
        outputs=[
            home_header_output,
            home_medication_card
        ]
    )

    home_question_btn.click(
        fn=go_to_assistant_tab,
        inputs=[],
        outputs=main_tabs
    )

    home_schedule_btn.click(
        fn=go_to_schedule_tab,
        inputs=[],
        outputs=main_tabs
    )

    voice_event = speech_to_text_btn.click(
        fn=record_voice_input,
        inputs=[],
        outputs=[
            speech_to_text_btn,
            speech_listening_btn,
            query_input,
            action_status_output
        ]
    )

    voice_event.then(
        fn=send_voice_message_if_available,
        inputs=[
            initial_profile_input,
            current_time_input,
            query_input,
            agent_state
        ],
        outputs=[
            model_response_output,
            memory_output,
            reasoning_output,
            personality_output,
            agent_state,
            query_input
        ]
    )

    reminder_timer = gr.Timer(value=30, active=True)

    reminder_timer.tick(
        fn=check_reminders,
        inputs=[],
        outputs=[
            reminder_popup,
            reminder_output,
            current_reminder_state
        ]
    )

    taken_btn.click(
        fn=mark_reminder_taken,
        inputs=[
            current_reminder_state
        ],
        outputs=[
            reminder_popup,
            action_status_output
        ]
    )

    not_taken_btn.click(
        fn=mark_reminder_not_taken,
        inputs=[
            current_reminder_state
        ],
        outputs=[
            reminder_popup,
            action_status_output
        ]
    )

    unsure_btn.click(
        fn=handle_reminder_uncertain,
        inputs=[
            current_reminder_state,
            initial_profile_input,
            current_time_input,
            agent_state
        ],
        outputs=[
            reminder_popup,
            action_status_output,
            model_response_output,
            memory_output,
            reasoning_output,
            personality_output,
            agent_state
        ]
    )

    submit_btn.click(
        fn=send_user_message,
        inputs=[
            initial_profile_input,
            current_time_input,
            query_input,
            agent_state
        ],
        outputs=[
            model_response_output,
            memory_output,
            reasoning_output,
            personality_output,
            agent_state,
            query_input
        ]
    )

    query_input.submit(
        fn=send_user_message,
        inputs=[
            initial_profile_input,
            current_time_input,
            query_input,
            agent_state
        ],
        outputs=[
            model_response_output,
            memory_output,
            reasoning_output,
            personality_output,
            agent_state,
            query_input
        ]
    )


# Start app

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        share=True,
        server_port=7861,
    )