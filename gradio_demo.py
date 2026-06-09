import datetime
import json
import os
import base64
from pathlib import Path
import re

from schedule_utils import load_medication_schedule, get_schedule_table
from reminder_utils import check_due_reminders, save_medication_action, snooze_reminder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr
from PIL import Image

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

import subprocess
import wave
import pyaudio

from PersonaVLM import PersonaVLMAgent
from inference import UserConfig


STT_AUDIO_FILE = "stt_input.wav"
STT_MIC_DEVICE = None
STT_MAX_DURATION = 15          # maximale opnametijd
STT_CHUNK_DURATION = 0.2       # audio per stukje
STT_SILENCE_DURATION = 1.0     # stop na 1 sec stilte
STT_SILENCE_THRESHOLD = 600    # gevoeligheid microfoon

if os.name == "nt":  # Windows
    PIPER_EXE = r"piper\piper.exe"
    PIPER_VOICE_MODEL = r"piper\voices\nl\nl_NL\ronnie\medium\nl_NL-ronnie-medium.onnx"
else:  # Linux / macOS
    PIPER_EXE = "piper/piper"
    PIPER_VOICE_MODEL = "piper/voices/nl/nl_NL/ronnie/medium/nl_NL-ronnie-medium.onnx"

PIPER_OUTPUT_WAV = "piper_response.wav"

stt_model = WhisperModel("base", device="cpu", compute_type="int8")


class MockAgent:
    def __init__(self, initial_profile: str, model=None):
        self.initial_profile_on_creation = initial_profile 
        self.init_user_config(initial_profile)
        self.agent = PersonaVLMAgent.PersonaVLM(self.config, agent_model=model)
    
    def init_user_config(self, initial_profile):
        profile = {}
        for line in initial_profile.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                profile[key.strip()] = value.strip()
        if "name" not in profile:
            raise ValueError("Initial profile must contain a 'name' field.")
        USER_NAME = profile["name"]
        del profile["name"]
        self.config = UserConfig()
        self.config.USER_NAME = USER_NAME
        self.config.PROFILE = profile
        self.config.DATA_STORAGE_PATH = Path(f"./output/{self.config.USER_NAME}")
        self.config.DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    def get_latest_semantic_memory(self):
        data_json_path = os.path.join(self.config.DATA_STORAGE_PATH, "user_data.json")
        if not os.path.exists(data_json_path):
            return ""
        with open(data_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        session = data.get("sessions", [{}])[-1]
        semantic_memory = session.get("semantic_memory", [])
        for idx, item in enumerate(semantic_memory):
            time = session["interactions"][idx][0]["time"]
            content = item['content'] if item['content'] else "N/A"
            keywords = item['keywords'] if item['keywords'] else "N/A"
            anslysis = item['reason']
        res = '\n'.join([f"Time: {time}",
                         f"Content: {content}",
                         f"Keywords: {keywords}",
                         f"Analysis: {anslysis}"])

        return res.replace("<img_fill>", "<image>")

    def get_profile(self):
        data_json_path = os.path.join(self.config.DATA_STORAGE_PATH, "user_data.json")
        if not os.path.exists(data_json_path):
            return json.dumps(self.config.PROFILE, indent=2)
        with open(data_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sessions = data.get("sessions", [{}])
        profile = sessions[-1].get("profile", {})
        return '\n'.join([f"{k}: {v}" for k, v in profile.items()])

    def parse_reasoning_chain(self, reasoning_chain):
        res = []
        for step in reasoning_chain:
            think = step.get("think", "")
            if step.get("action") == "retrieve":
                keywords = step.get("retrieve_info", {}).get("keywords", "")
                start_time = step.get("retrieve_info", {}).get("start_time", "None")
                end_time = step.get("retrieve_info", {}).get("end_time", "None")
                res.append(f"<think>{think}</think>\n<retrieve>keywords: {keywords}, start_time: {start_time}, end_time: {end_time}</retrieve>")
            else:
                res.append(f"<think>{think}</think>")
        return '\n'.join(res)

    def draw_personality_evolution(self, personality_info):
        current_personality = [
            personality_info.get('openness', 3),
            personality_info.get('conscientiousness', 3),
            personality_info.get('extraversion', 3),
            personality_info.get('agreeableness', 3),
            personality_info.get('neuroticism', 3),
        ]
        data_json_path = os.path.join(self.config.DATA_STORAGE_PATH, "user_data.json")
        with open(data_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        evolved_personality = [
            float(str(data['big_five_personality'].get('openness', 3.0))[:3]),
            float(str(data['big_five_personality'].get('conscientiousness', 3.0))[:3]),
            float(str(data['big_five_personality'].get('extraversion', 3.0))[:3]),
            float(str(data['big_five_personality'].get('agreeableness', 3.0))[:3]),
            float(str(data['big_five_personality'].get('neuroticism', 3.0))[:3])
        ]
        save_path = os.path.join(self.config.DATA_STORAGE_PATH, "personality_evolution.png")
        plot_personaity_evolving(current_personality, evolved_personality, save_path)


    def send_message(self, current_time: str, query: str, img_path: str = None, personality_input: str = None):
        message = {
            "query": query,
            "imgs": img_path,
            "time": current_time,
        }
        response, personality_info, reasoning_chain = self.agent.send_message(
                                                    message,
                                                    personality_input=personality_input,
                                                    return_response_only=False,
                                                    update=True)
        model_response_output = response
        memory_output = self.get_latest_semantic_memory()
        reasoning_output = self.parse_reasoning_chain(reasoning_chain)
        personality_output = '\n'.join([f"{k}: {v}" for k, v in personality_info.items()])
        self.draw_personality_evolution(personality_info)
        
        return model_response_output, memory_output, reasoning_output, personality_output

def plot_personaity_evolving(current, evolving, save_path):
    labels = np.array(['O', 'C', 'E', 'A', 'N'])
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(figsize=(12, 6), ncols=2, subplot_kw=dict(polar=True))
    def _draw_radar(ax, values, color, title):
        plot_values = values + values[:1]
        ax.fill(angles, plot_values, color=color, alpha=0.4, hatch='//')
        ax.plot(angles, plot_values, color=color, linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=20)
        ax.tick_params(axis='x', pad=15)
        ax.set_ylim(0, 5)
        ax.spines['polar'].set_visible(False)
        ax.set_title(title, size=24, pad=20)
        ax.plot(0, 0, 'o', markersize=12, color=color, mfc='white', zorder=10)
        outer_values = [5] * (num_vars + 1)
        ax.plot(angles, outer_values, color='gray', linewidth=1.5, zorder=5)

    _draw_radar(axes[0], current, 'steelblue', 'Current Personality')
    _draw_radar(axes[1], evolving, 'firebrick', 'Evolved Personality')
    arrow_start = (0.47, 0.5)
    arrow_end = (0.53, 0.5)

    arrow = patches.FancyArrowPatch(
        arrow_start,
        arrow_end,
        transform=fig.transFigure,
        color='red',
        arrowstyle='->',
        mutation_scale=30,
        linewidth=3
    )
    fig.add_artist(arrow)
    plt.subplots_adjust(top=0.85) 
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    def add_top_padding(image_path, padding_pixels=500):
        with Image.open(image_path) as img:
            # 创建一个更高的新画布，背景为白色
            new_size = (img.width + padding_pixels, img.height + padding_pixels)
            new_img = Image.new("RGB", new_size, "white")
            # 将原始图片粘贴到新画布的下半部分
            new_img.paste(img, (int(padding_pixels/2), padding_pixels))
            # 覆盖保存原始文件
            new_img.save(image_path)
    add_top_padding(save_path)

def get_image_base64_src(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

# STT Functie
def record_and_transcribe():
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

    return text

# Piper TTS
def speak_with_piper(text):
    if not text or not text.strip():
        return

    subprocess.run(
        [PIPER_EXE, "--model", PIPER_VOICE_MODEL, "--output_file", PIPER_OUTPUT_WAV],
        input=text,
        text=True,
        check=True
    )

    wf = wave.open(PIPER_OUTPUT_WAV, "rb")
    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf.close()

    return text

    # Hier laadt ie de prompt instructies in
def load_prompt_instructions():
    path = "prompt_instructions/medication_assistant.txt"

    if not os.path.exists(path):
        print("Prompt instructions file not found.")
        return ""

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
        print(f"Loaded prompt instructions: {path} ({len(content)} characters)")
        print("Prompt instructions preview:")
        print(content[:500])
        return content

    # Medicatie informatie laden
def load_medication_context():
    info_folder = "medication_info"
    context_parts = []

    if not os.path.exists(info_folder):
        print("Medication info folder not found.")
        return ""

    for filename in os.listdir(info_folder):
        if filename.endswith(".txt"):
            path = os.path.join(info_folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                print(f"Loaded medication file: {filename} ({len(content)} characters)")
                context_parts.append(f"--- BRONBESTAND: {filename} ---\n{content}")

    medication_context = "\n\n".join(context_parts)
    print("Medication context preview:")
    print(medication_context[:500])

    return medication_context

def clean_model_response(text):
    if not text:
        return ""

    # Verwijder tekens die niet passen bij normale cijfers of leestekens
    # Dit haalt onverwachte symbolen weg
    allowed_pattern = r"[^a-zA-ZÀ-ÿ0-9\s.,!?;:()\-/'\"%€+*=&\n]"
    text = re.sub(allowed_pattern, "", text)

    # Ruim dubbele spaties op
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()



def check_reminder_notification():
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
def handle_taken(reminder_message):
    save_medication_action(
        reminder_message=reminder_message,
        status="ingenomen"
    )
    return (
        gr.update(visible=False),
        "Medicatie gemarkeerd als ingenomen."
    )


def handle_not_taken(reminder_message):
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
        "Ik zal de herinnering over 10 minuten opnieuw tonen."
    )


def handle_unsure(reminder_message, initial_profile, current_time, agent_state):
    save_medication_action(
        reminder_message=reminder_message,
        status="twijfel"
    )

    if agent_state is None or agent_state.initial_profile_on_creation != initial_profile:
        agent = MockAgent(
            initial_profile=initial_profile,
            model=agent_state.agent.mllm if agent_state else None
        )
    else:
        agent = agent_state

    if not current_time:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt_instructions = load_prompt_instructions()
    medication_context = load_medication_context()
    schedule_context = load_medication_schedule()

    twijfel_query = (
        f"{prompt_instructions}\n\n"
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
        current_time,
        twijfel_query,
        "",
        personality_input=None
    )

    response = clean_model_response(response)

    personality_image_path = os.path.join(agent.config.DATA_STORAGE_PATH, "personality_evolution.png")
    if not os.path.exists(personality_image_path):
        personality_image_path = None

    return (
        gr.update(visible=False),
        "Twijfel genoteerd. De medicatie-assistent geeft extra uitleg.",
        response,
        memory_output,
        reasoning_output,
        personality_output,
        agent,
        gr.update(value=personality_image_path, visible=True)
    )
    
def process_interaction(
    initial_profile: str,
    current_time: str,
    user_query: str,
    image_input,
    personality_input: str,
    agent_state: MockAgent
):
    if agent_state is None or agent_state.initial_profile_on_creation != initial_profile:
        agent = MockAgent(initial_profile=initial_profile, model=agent_state.agent.mllm if agent_state else None)
    else:
        agent = agent_state

    if not current_time:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    manual_personality = None
    default_personality_str = (
                "openness: 3.0\n"
                "conscientiousness: 3.0\n"
                "extraversion: 3.0\n"
                "agreeableness: 3.0\n"
                "neuroticism: 3.0"
            )
    if personality_input.strip() != default_personality_str and personality_input.strip():
        try:
            manual_personality = {}
            for line in personality_input.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    try:
                        value = float(value.strip()[:3])
                        value = max(1.0, min(5.0, value))
                        manual_personality[key.strip()] = value
                    except ValueError:
                        manual_personality[key.strip()] = value.strip()
            if not manual_personality: # 如果解析后为空字典
                manual_personality = None
        except Exception as e:
            print(f"Error parsing personality input: {e}")
            manual_personality = None
    # print('manual_personality:', manual_personality)
    Img = image_input if image_input else None
    if Img:
        image_path = os.path.join(agent.config.DATA_STORAGE_PATH, "image.png")
        Img.save(image_path)
    else:
        image_path = ''

     # Hier vertellen we wat het model moet doen
    prompt_instructions = load_prompt_instructions()
    medication_context = load_medication_context()
    schedule_context = load_medication_schedule()

    user_query = (
    f"{prompt_instructions}\n\n"
    f"Medicatie-informatie:\n{medication_context}\n\n"
    f"Medicatieschema en huidige planning:\n{schedule_context}\n\n"
    f"Vraag van gebruiker:\n{user_query}\n\n"
    "Antwoord nu volledig in het Nederlands:"
    )
    response, memory_output, reasoning_output, personality_output = agent.send_message(
        current_time, user_query, image_path, personality_input=manual_personality
    )

    

    response = clean_model_response(response)
    speak_with_piper(response)

    personality_image_path = os.path.join(agent.config.DATA_STORAGE_PATH, "personality_evolution.png")
    if not os.path.exists(personality_image_path):
        personality_image_path = None
    return response, memory_output, reasoning_output, personality_output, agent, gr.update(value=None, visible=False)


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@400&display=swap');
* {
    font-family: 'Lato', sans-serif !important;
}
.gradio-container {
    font-family: 'Lato', sans-serif !important;
}

#speech_button,
#send_button {
    min-height: 70px !important;
    height: 70px !important;
    max-width: 70px !important;
    padding: 12px !important;
}

#speech_button img,
#send_button img {
    width: 34px !important;
    height: 34px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown('---')

    agent_state = gr.State(value=None)
    current_reminder_state = gr.State(value="")

    with gr.Tab("Assistent"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Input")

                query_input = gr.Textbox(
                    label="Uw vraag",
                    lines=3,
                    placeholder="Stel hier je vraag",
                )

                with gr.Row(equal_height=True):
                    speech_to_text_btn = gr.Button(
                        value="",
                        icon="assets/microphone.svg",
                        elem_id="speech_button",
                        scale=1,
                        min_width=0
                    )

                    submit_btn = gr.Button(
                        value="",
                        icon="assets/send.svg",
                        variant="primary",
                        elem_id="send_button",
                        scale=1,
                        min_width=0
                    )

                image_upload_input = gr.Image(
                    label="Image Upload (Optional)",
                    type="pil",
                    visible=False
                )

                with gr.Row():
                    initial_profile_input = gr.Textbox(
                        label="Initial User Profile",
                        lines=3,
                        value="name: Henk\ngeslacht: man\ntaal: Nederlands\nvoorkeuren: eenvoudige uitleg over medicatie",
                        scale=1
                    )

                    current_time_input = gr.Textbox(
                        label="Timestamp (Optional)",
                        info="Format: YYYY-MM-DD HH:MM. If empty, the current system time will be used.",
                        lines=3,
                        value=None,
                        scale=1,
                        visible=False
                    )

                default_personality_str = (
                    "openness: 3.0\n"
                    "conscientiousness: 3.0\n"
                    "extraversion: 3.0\n"
                    "agreeableness: 3.0\n"
                    "neuroticism: 3.0"
                )

                personality_input = gr.Textbox(
                    label="Set Personality (Optional)",
                    info="Override the user's personality for this response.",
                    lines=5,
                    value=default_personality_str,
                    interactive=True,
                    visible=False
                )

            with gr.Column(scale=2):
                gr.Markdown("## Antwoord")

                with gr.Group(visible=False) as reminder_popup:
                    reminder_output = gr.Textbox(
                        label="Medicatieherinnering",
                        lines=3,
                        interactive=False
                    )

                    with gr.Row():
                        taken_btn = gr.Button("Ingenomen", variant="primary")
                        not_taken_btn = gr.Button("Niet ingenomen")
                        unsure_btn = gr.Button("Ik twijfel")

                action_status_output = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False,
                    visible=True
                )

                model_response_output = gr.Textbox(
                    label="Antwoord van de medicatie-assistent",
                    lines=5,
                    interactive=False
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

                personality_evolution_output = gr.Image(
                    label="Personality Evolution",
                    visible=False,
                )

    with gr.Tab("Medicatieschema"):
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
            fn=get_schedule_table,
            inputs=[],
            outputs=schedule_table
        )

    speech_to_text_btn.click(
        fn=record_and_transcribe,
        inputs=[],
        outputs=query_input
    )

    reminder_timer = gr.Timer(value=30, active=True)

    reminder_timer.tick(
        fn=check_reminder_notification,
        inputs=[],
        outputs=[reminder_popup, reminder_output, current_reminder_state]
    )

    taken_btn.click(
        fn=handle_taken,
        inputs=[current_reminder_state],
        outputs=[reminder_popup, action_status_output]
    )

    not_taken_btn.click(
        fn=handle_not_taken,
        inputs=[current_reminder_state],
        outputs=[reminder_popup, action_status_output]
    )

    unsure_btn.click(
        fn=handle_unsure,
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
            agent_state,
            personality_evolution_output
        ]
    )

    submit_btn.click(
        fn=process_interaction,
        inputs=[
            initial_profile_input,
            current_time_input,
            query_input,
            image_upload_input,
            personality_input,
            agent_state
        ],
        outputs=[
            model_response_output,
            memory_output,
            reasoning_output,
            personality_output,
            agent_state,
            personality_evolution_output
        ]
    )

if __name__ == "__main__":
    APP_USER = "admin"
    APP_PASSWORD = "personavlmdemo"

    demo.launch(
        server_name="0.0.0.0",
        share=True,
        server_port=7861,
    )