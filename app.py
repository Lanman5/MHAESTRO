import streamlit as st
from openai import OpenAI
from io import StringIO, BytesIO
import csv
import os
import re
import json
import logging 
import streamlit.components.v1 as components
from html import escape
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from datetime import datetime
import hashlib

#configures the logging feature 
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

#Some variables from streamlit secrets
AUTHORIZED_PASSWORDS = st.secrets.get("AUTHORIZED_PASSWORDS")

def check_password():
    """Simple password protection."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        password = st.text_input("Enter the app password:", type="password")
        if password in AUTHORIZED_PASSWORDS:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.warning("Incorrect password")
            st.stop()

#check_password() #APP IS NOW UNLOCKED -> Remove ONLY the first # to lock it again

st.set_page_config(page_title="Spirit Engine 2.0", page_icon="🧠", layout="centered")

st.title("🎙️Interviewer and Storyteller📖")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fetch all available voices dynamically
voice_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
voices_response = voice_client.voices.search()
voice_options = {voice.name: voice.voice_id for voice in voices_response.voices}

if not voice_options:
    st.error("No voices found in your account. Add a voice in ElevenLabs first.")

@st.cache_data
def get_model_list():
    try:
        models = client.models.list()
        return sorted([model.id for model in models.data if "gpt" in model.id])
    except Exception as e:
        st.error(f"Failed to fetch chat models - check your OpenAI API key and network connection:")
        logging.error(f"Error fetching models: {e}")
        return []
    
def get_elevenlabs_model_list():
    try:
        models = voice_client.models.list()
        return sorted([model.model_id for model in models])
    except Exception as e:
        st.error(f"Failed to fetch ElevenLabs models - check your ElevenLabs API key and network connection:")
        logging.error(f"Error fetching ElevenLabs models: {e}")
        return []

chat_models = get_model_list()
default_model = "gpt-4o"

if default_model in chat_models:
    default_index = chat_models.index(default_model)
else:
    default_index = 0 if chat_models else -1

if not chat_models:
    st.error("No models available. Please check your OpenAI API credentials or network connection.")
    st.stop()


#functions:
class SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"
    
def read_file(input_file):
    if input_file is None:
        return "ERROR: No file provided."
    if isinstance(input_file, str):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {input_file}: {e}")
            return f"ERROR reading file from path"
    try:
        return StringIO(input_file.getvalue().decode("utf-8")).read()
    except Exception as e:
        logging.error(f"Error reading uploaded file: {e}")
        return f"ERROR reading uploaded file"
    
def generate_transcript(messages, user_name="User"):
    transcript = ""
    for msg in messages[1:]:
        if msg["role"] == "system":
            continue
            #role = "prompt"
        elif msg["role"] == "assistant":
            role = "Interviewer"
        else:
            role = user_name

        content = msg["content"]
        transcript += f"{role}: {content}\n\n"
    return transcript

def check_story_quality(story_text):
    """Check quality of generated story using GPT."""
    try:
        response = client.chat.completions.create(
            model=st.session_state["story_model_select"],  # or a cheaper model if desired
            messages=[
                {"role": "system", "content": f"{st.session_state['framework_prompt']}"},
                {"role": "user", "content": f"Use the following framework to check the quality of this story and determine whether or not it's ready for release.\n\nStory:\n{story_text}\n\nReturn JSON strictly in the form {{\"adequate\": true}} or {{\"adequate\": false, \"reason\": \"...\"}}"}
            ],
            response_format={"type": "json_object"} 
        )
        eval_text = response.choices[0].message.content.strip()
        result = json.loads(eval_text)  # will error if GPT doesn't return valid JSON
        logging.debug("Story quality evaluation result: %s", result)
        return result
    except Exception as e:
        logging.debug("Story quality evaluation error: %s", e)
        return {"adequate": True}  # fallback → assume adequate to avoid infinite loops
    
def json_check(text):
#attempts to get rid of any trailing context text returned by AI before json.
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        logging.warning("No JSON object found in model output.")
        return {}
    
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decoding failed: {e}")
        return {}
    
def analyze_story_stages(messages, analysis_model, analysis_prompt):
    transcript = generate_transcript(messages)

    response = client.chat.completions.create(
        model=analysis_model,
        messages=[
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": "Carry out the analysis as specified in the framework above using this transcript:" +transcript}])

    result = response.choices[0].message.content
    logging.debug("Raw analysis result: %s", result)
    return json_check(result)

def read_csv():
    DEFAULT_FILE_PATH = "default_prompts.csv"
    string_data = None  
    if os.path.exists(DEFAULT_FILE_PATH):
        with open(DEFAULT_FILE_PATH, 'r', encoding='utf-8') as f:
            string_data = StringIO(f.read())
    else:
        st.error(f"default prompts contained in '{DEFAULT_FILE_PATH}' was not found.")
        return

    if string_data:
        try:
            reader = csv.reader(string_data, delimiter=';', quotechar='"')
            prompt_list = list(reader)
            return prompt_list
        except csv.Error as e:
            logging.error("Error reading CSV data: %s", e)
            st.error(f"Error reading CSV data - check the formatting carefully")
            return

def play_sound(text, key, voice_id):
    with st.spinner("Generating audio..."):
        try:
            tts_model = st.session_state.get("TTS_model", "eleven_multilingual_v2")
            response = voice_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=tts_model,
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.1,
                    speed=1.1,
                    use_speaker_boost=True
                )
            )

            # Combine chunks into bytes
            audio_bytes = b"".join(response)
            
            st.session_state[f"{key}_audio"] = audio_bytes

        except Exception as e:
            st.error(f"Error occurred while playing sound - check your elevenlabs key")
            logging.error(f"Text-to-speech error: {e}")

def transcribe_with_elevenlabs_sdk(audio_bytes: bytes, timeout: int = 600) -> str:
    """
    Convert audio bytes into text using ElevenLabs SDK.
    """
    try:
        bio = BytesIO(audio_bytes)
        resp = voice_client.speech_to_text.convert(
            file=bio,
            model_id="scribe_v1",
            diarize=False,
            tag_audio_events=False,
            request_options={"timeout_in_seconds": timeout}
        )
        return getattr(resp, "text", "") or resp.get("text", "")
    except Exception as e:
        st.error(f"Transcription failed - check your ElevenLabs API key and network connection:")
        logging.error(f"Transcription failed: {e}")
        return ""
    
def autoplay_audio(audio_bytes: bytes):
    import base64
    b64 = base64.b64encode(audio_bytes).decode()
    html_code = f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        window.currentAudio?.pause();
        window.currentAudio?.currentTime = 0;
        window.currentAudio = document.currentElementsByTagName('audio')[document.getElementsByTagName('audio').length-1];
    </script>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def clear_audio_keys():
    """Remove all stored audio blobs from session_state."""
    audio_keys = [k for k in st.session_state.keys() if k.endswith("_audio")]
    for k in audio_keys:
        st.session_state.pop(k, None)
# ---------------- Hybrid Chat Input ----------------
def hybrid_chat_input(label="Reply to interviewer...", disabled=False):
    """
    Provides both text and audio input.
    Ensures session_state is not wiped when using st.audio_input.
    Now supports a `disabled` flag to disable input when chat is locked.
    """
    col1, col2 = st.columns([3, 1])
    user_input = None

    with col1:
        # If disabled, render a disabled text input instead of st.chat_input
        if disabled:
            st.text_input(label, value="", placeholder="Chat is disabled.", disabled=True, label_visibility="collapsed")
        else:
            typed = st.chat_input(label)
            if typed:
                user_input = typed

    with col2:
        if disabled:
            # Render a disabled button-like placeholder instead of the audio input
            st.text_input("🎙️ Speak", value="", disabled=True, label_visibility="collapsed")
        else:
            audio = st.audio_input("🎙️ Speak", label_visibility="collapsed")
            if "auto_speak" in st.session_state:
                if st.session_state.auto_speak:
                    if st.button("🙊 Mute"):
                        st.markdown("""
                        <script>
                            if (window.currentAudio) {
                                window.currentAudio.pause();
                                window.currentAudio.currentTime = 0;
                            }
                        </script>
                        """, unsafe_allow_html=True)
            if audio:
                audio_bytes = audio.read()
                # compute hash to uniquely identify this audio blob
                fingerprint = hashlib.sha256(audio_bytes).hexdigest()
                processed_ids = st.session_state.setdefault("processed_audio_ids", set())

                if fingerprint not in processed_ids:
                    transcript = transcribe_with_elevenlabs_sdk(audio_bytes)
                    if transcript:
                        user_input = transcript
                        processed_ids.add(fingerprint)

    return user_input


def generate_testing_report():
    with st.spinner("Generating testing report - please do not leave this page"):
        return_text = "===================================================================================================\n"
        if 'transcript' in st.session_state:
            return_text += f"****Testing report for {st.session_state.get('interview_name', 'UNKNOWN')}****\n"
            return_text += "===================================================================================================\n"
            return_text += f"------------------------------------------##Interview transcript##--------------------------------------: \n {st.session_state.get('transcript', 'NO TRANSCRIPT AVAILABLE')}\n\n"
            return_text += "####################################################################################################\n"
            return_text += f"-------------------------------##Analysis##--------------------------------: \n {st.session_state.get('analysis', 'NO ANALYSIS AVAILABLE')}\n\n"
            return_text += "####################################################################################################\n"
            return_text += f"----------------------------##Generated Narrative##---------------------------------------: \n {st.session_state.get('narrative', 'NO NARRATIVE AVAILABLE')}"
            return_text += "####################################################################################################\n"

            if 'narrative' in st.session_state: #for each type of story, if it exists, add it in, else generate what would've been made
                if 'child_story' in st.session_state:
                    return_text += f"----------------------------##Child Story##-----------------------------: \n {st.session_state.get('child_story', 'ERROR FETCHING CHILD STORY')}\n\n"
                else:
                    child_story_context = ""
                    for selected_title in st.session_state["child_story_files"]:
                        for file_obj in st.session_state["titled_prereq_files"]:
                            if file_obj["title"] == selected_title:
                                child_story_context += f"\n\n----------{file_obj['title']}----------\n"
                                child_story_context += file_obj["content"]
                    try:
                        response = client.chat.completions.create(
                                model=st.session_state["story_model_select"],
                                messages=[
                                    {"role": "system", "content": st.session_state["child_system_prompt"] + child_story_context},
                                    {"role": "user", "content": st.session_state["child_init_prompt"] + "\n-------Analysis-------\n"+ st.session_state.get('analysis', 'NO ANALYSIS AVAILABLE')+ "\n-------Narrative-------\n"+ st.session_state.get('narrative', 'NO NARRATIVE AVAILABLE')}
                                ]
                            )
                        return_text += "-------------------------------##Child Story [NOT GENERATED BY USER]##------------------------------:\n" + response.choices[0].message.content
                        return_text += "####################################################################################################\n"
                    except Exception as e:
                        st.error(f"Error generating child story")
                        logging.error(f"Error generating child story: {e}")
            else:
                return_text += "NO STORIES WERE GENERATED IN THIS SESSION"
        else:
            st.error("TESTING REPORT COULD NOT BE GENERATED")
            st.stop()
            return "<TESTING REPORT ERROR>"

        return return_text
# Main app logic
prompt_list = read_csv()
if not prompt_list:
    st.error("Failed to read prompts.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config_initialized" not in st.session_state:
    if prompt_list and len(prompt_list) < 17: #when adding more default items, change this number accordingly 

        st.error("Insufficient prompts in Default CSV configuration. Please ensure at least 17 entries.")
        st.stop()

    #DEFAULT FILES -> WHEN ADDING MORE PRE-REQ FILES, ADD THEM HERE. 
    init_file_data = []
    init_file_data.append({"title": prompt_list[10][0], "content": read_file(prompt_list[10][1])})
    init_file_data.append({"title": prompt_list[11][0], "content": read_file(prompt_list[11][1])})
    init_file_data.append({"title": prompt_list[8][0], "content": read_file(prompt_list[8][1])})
    init_file_data.append({"title": prompt_list[9][0], "content": read_file(prompt_list[9][1])})
    init_file_data.append({"title": prompt_list[16][0], "content": read_file(prompt_list[16][1])})

    st.session_state.update({

        #Format here is "title the program will access information from": location in CSV file 
        #files 
        "titled_prereq_files": init_file_data,
        # Interviewer
        "interview_prompt": prompt_list[0][1],
        "first_question": prompt_list[1][1],
        "interviewer_model": default_model,
        "interview_selected_files": [prompt_list[8][0],prompt_list[9][0],prompt_list[10][0]],

        # Analysis
        "analysis_system_prompt": prompt_list[2][1],
        "analysis_init_prompt": prompt_list[3][1],
        "analysis_model": default_model,
        "analysis_selected_files": [],

        #First person narrative
        "narrative_system_prompt": prompt_list[14][1],
        "narrative_init_prompt": prompt_list[15][1],

        # Adult Stories
        "adult_system_prompt": prompt_list[4][1],
        "adult_init_prompt": prompt_list[5][1],
        "adult_story_files": [prompt_list[10][0],prompt_list[9][0],prompt_list[8][0]],

        # Child Stories
        "child_system_prompt": prompt_list[6][1],
        "child_init_prompt": prompt_list[7][1],
        "child_story_files":[prompt_list[10][0],prompt_list[11][0],prompt_list[8][0],prompt_list[9][0], prompt_list[16][0]],

        #assistant bots
        "steering_prompt": prompt_list[12][1],
        "safeguarding_prompt": prompt_list[13][1],

        "story_model_select": default_model,
        "steering_model": default_model,
        "safeguarding_model": default_model,

        "story_stages":{
        "Moment": False,
        "Details": False,
        "Realisation": False,
        "Change": False,
        "Resolution": False
        },

        "safeguarding_flag" : False,

        "config_initialized": True,

        #tts config
        "TTS_model": "eleven_v3",#set a default
        "interviewer_voiceid":  list(voice_options.values())[0],

# story quality framework prompt 
        "framework_prompt": """You are a strict story quality evaluator for children’s stories (ages 2–5). 
Your task is to evaluate a story against the following framework and return ONLY valid JSON.

Story Quality Evaluation Framework (Ages 2–5)

Category 1: Style & Inclusivity 
Goal: Story is simple, playful, inclusive, and suitable for ages 2–5.
Checks:
1. Language: Vocabulary is simple and concrete, with short sentences and rhythmic/engaging style.
2. Structure: Clear beginning–middle–end following a recognised pattern.
3. Tone: Warm, playful, reassuring, with light conflict resolved quickly.
4. Inclusivity: Differences shown as natural and positive, with no stereotypes or deficit framing.
5. Ending: Ends with a soft, reflective, child-like question.

Category 2: Learning Alignment (EYFS Distilled) 
Goal: Story supports 1–2 EYFS learning areas and developmental anchors.
Checks:
6. EYFS Coverage: Clearly reflects 1–2 EYFS areas (e.g. Communication, PSED, Literacy).
7. Developmental Anchors: Supports comprehension, social-emotional awareness, early literacy, curiosity, or imagination.
8. Effective Learning Traits: Models play, exploration, persistence, or problem-solving.
9. Avoids Didacticism: Values shown through characters’ actions and play, not adult teaching.

Category 3: Pedagogical Fit (Best Practices) 
Goal: Story can be used by educators, supporting play and scaffolding.
Checks:
10. Play-Based: Feels playful and imaginative, not didactic.
11. Child-Centred: Child/animal characters drive the action, adults in supporting roles.
12. Language-Rich: Introduces new vocabulary in context, with repetition or dialogue.
13. Scaffolding Potential: Leaves space for educators to extend (refrains, questions, props).
14. Holistic: Touches multiple domains (emotional, social, language, imaginative) without overload.

Category 4: Story Quality (Intangibles) 
Goal: Story is engaging, resonant, imaginative, and memorable.
Checks:
15. Engagement: Flows naturally and enjoyable to read aloud.
16. Emotional Resonance: Emotions clear and relatable, with satisfying resolution.
17. Imaginative Spark: Creates vivid, playful, imaginative images.
18. Memorability: Has a refrain, rhythm, or image that is likely to stick.

Pre-Release Standards 
- Category 1: All 5 must be Yes.
- Category 2: At least 3 of 4 must be Yes.
- Category 3: All 5 must be Yes or Partial.
- Category 4: All 4 must be Yes or Partial, with at least 1 Yes.

Final Holistic Check 
Question: “If you were reading this to a group of 3–5-year-olds, would they enjoy it? Would they ask to hear it again?”
Response: Yes/No + one-sentence reason.

Output Rules:
- Return ONLY JSON in the following format:
{
  "adequate": true/false,
  "reason": "If false, give a very short, practical reason for improvement."
}
- Never include explanations outside of JSON.
- If the story fully meets the Pre-Release Standards and passes the Holistic Check → {"adequate": true}.
- Otherwise → {"adequate": false, "reason": "..."}.
"""
    })

tab1, tab2, tab3 = st.tabs(["🗨️Interview", "📚Storytelling","⚙️ Configuration"])

with tab1:
    st.title("🗣️Interviewer")
    name = st.text_input("Enter your Name")
    auto_speak = st.checkbox("🔊 Automatically speak interviewer responses", value=True, key="auto_speak")
    if st.button("🎤 Begin Interview"):
        if not name:
            st.warning("Please enter your name to start the interview.")
        else:
            # log interview start time
            st.session_state["interview_start_time"] = datetime.now()
            st.session_state["interview_name"] = name
            clear_audio_keys()

            # Clear only relevant keys
            for key in ["messages", "transcript", "analysis", "view_analysis", "interview_ended", "chat_locked"]:
                st.session_state.pop(key, None)

            interview_context = ""
            for selected_title in st.session_state["interview_selected_files"]:
                for file_obj in st.session_state["titled_prereq_files"]:
                    if file_obj["title"] == selected_title:
                        interview_context += f"\n\n----------{file_obj['title']}----------\n"
                        interview_context += file_obj["content"]

            interview_prompt_template = st.session_state["interview_prompt"]
            interview_prompt = interview_prompt_template.format_map(SafeDict(name=name))

            interview_question_template = st.session_state["first_question"]
            interview_question = interview_question_template.format_map(SafeDict(name=name))
            st.session_state.messages = [
                {"role": "system", "content": interview_prompt + interview_context},
                {"role": "assistant", "content": interview_question}
            ]
            
            # 🎤 Auto-speak the first question if enabled
            if auto_speak:
                st.session_state["last_reply_to_speak"] = interview_question

    # ------------------- Chat Display -------------------
    if st.session_state.get("messages"):
        if "story_stages" in st.session_state:
            stages = st.session_state["story_stages"]
            total = len(stages)
            covered = sum(1 for v in stages.values() if v)
            progress = covered / total

            st.markdown(f"##### __***Interview Progress: {int(progress * 100)}%***__")
            st.progress(progress)

        inner = ""
        for msg in st.session_state.messages[1:]:
            if msg["role"] == "system":
                continue
            role = "🧑‍💼 Interviewer" if msg["role"] == "assistant" else f"🙋 {st.session_state['interview_name']}"
            content = escape(msg["content"]).replace("\n", "<br>")
            inner += f"<p><strong>{role}:</strong><br>{content}</p><hr>"

        chat_html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <div id="chat-container" style="
            height:400px; 
            overflow-y:auto; 
            padding:10px; 
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            line-height: 1.5;
            background-color: #f9f9f9;
            border-radius: 8px;
        ">
            {inner}
        </div>
        <script>
            const el = document.getElementById('chat-container');
            if (el) {{
                el.scrollTo({{ top: el.scrollHeight, behavior: 'smooth' }});
            }}
        </script>
        """
        components.html(chat_html, height=420, scrolling=False)

        # ------------------- Autoplay pending interviewer reply ------------------
        if auto_speak and "last_reply_to_speak" in st.session_state:
            reply_text = st.session_state.pop("last_reply_to_speak")

            # Generate audio
            play_sound(reply_text, "last_reply", st.session_state.get("interviewer_voiceid", list(voice_options.values())[0]))

            # Only autoplay if audio was successfully created
            if "last_reply_audio" in st.session_state:
                autoplay_audio(st.session_state["last_reply_audio"])

    # ------------------- Chat Input -------------------
    if st.session_state.get("messages"):
        if user_input := hybrid_chat_input("Type or speak your reply...", disabled=st.session_state.get("chat_locked", False)):
            st.session_state.messages.append({"role": "user", "content": user_input})

            all_vars_covered = True
            steering_parts = []

            with st.spinner("Thinking..."):
                new_analysis = analyze_story_stages(
                    st.session_state.messages,
                    st.session_state['steering_model'],
                    st.session_state['steering_prompt']
                )
                if new_analysis:
                    st.session_state['story_stages'].update(new_analysis)
                    missing = [s for s, covered in st.session_state['story_stages'].items() if not covered]
                    if missing:
                        all_vars_covered = False
                        steering_parts.append(
                            f"The following stages have not been meaningfully covered: {', '.join(missing)}."
                        )

                safeguarding_analysis = analyze_story_stages(
                    st.session_state.messages,
                    st.session_state['safeguarding_model'],
                    st.session_state['safeguarding_prompt']
                )
                if safeguarding_analysis:
                    st.session_state['safeguarding_flag'] = safeguarding_analysis.get('safeguarding_flag')

            # 3. combine all steering instructions
            if st.session_state['safeguarding_flag'] is True:
                steering_instruction = (
                    "The interviewee has indicated that either themselves or somebody else is at risk of harm. "
                    "Please end the interview immediately and advise them to seek help ensuring you don't ask any follow up questions."
                )
                st.session_state["chat_locked"] = True
            elif all_vars_covered:
                steering_instruction = (
                    "All criteria have been covered. Please thank the interviewee and ask them if there's "
                    "anything they'd like to add before ending the interview."
                )
            else:
                steering_instruction = (
                    " ".join(steering_parts)
                    + " Focus your next question to guide the participant toward one of these missing stages, "
                    "while still following the interview framework and maintaining empathy and depth."
                )

            # 4. Send steering instruction to main interviewer
            temp_messages = st.session_state.messages.copy()
            if steering_instruction:
                temp_messages.append({"role": "system", "content": steering_instruction})
            logging.debug(temp_messages)

            # 5. main interviewer generates the next question
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=st.session_state["interviewer_model"],
                        messages=temp_messages,
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = "Sorry, there was an issue generating a response."
                    st.error(f"Interviwer response generation failed - check your OpenAI API key and network connection:")
                    logging.error(f"Interviewer response generation error: {e}")

            # 6. Append interviewer message
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # 7. Auto-speak interviewer replies if enabled but push to next run due to rerun
            if auto_speak:
                st.session_state["last_reply_to_speak"] = reply

            # 8. Refresh UI
            st.rerun()

        if 'interview_ended' not in st.session_state:
            st.session_state.interview_ended = False

        if st.button("🛑 End Interview"):
            st.session_state.interview_ended = True
            clear_audio_keys()
            st.session_state.chat_locked = True  

        if st.session_state.interview_ended:
            st.session_state["interview_end_time"] = datetime.now()
            interview_length = st.session_state["interview_end_time"] - st.session_state["interview_start_time"]
            total_seconds = interview_length.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            if 'transcript' not in st.session_state:
                transcript_text = generate_transcript(st.session_state.messages, user_name=name)
                st.session_state['transcript'] = transcript_text + f"Interview length: {minutes} minutes, {seconds} seconds"
                st.session_state.messages = [
                    {"role": "assistant", "content": f"Thank you {name} for sharing your story. This concludes our interview."}
                ]
                st.chat_message("assistant").markdown(st.session_state.messages[-1]["content"])
                st.success("Interview ended. You can download your transcript below.")
                st.info("Please wait until analysis has finished before generating a story...")

            st.download_button(
                label="📥 Download Transcript",
                data=st.session_state['transcript'],
                file_name=f"{name}_interview_transcript.txt",
                mime="text/plain"
            )

            if 'analysis' not in st.session_state:
                with st.spinner("Analysing your interview..."):
                    analysis_context = ""
                    for selected_title in st.session_state["analysis_selected_files"]:
                        for file_obj in st.session_state["titled_prereq_files"]:
                            if file_obj["title"] == selected_title:
                                analysis_context += f"\n\n----------{file_obj['title']}----------\n"
                                analysis_context += file_obj["content"]
                    try:
                        response = client.chat.completions.create(
                            model=st.session_state["analysis_model"],
                            messages=[
                                {"role": "system", "content": st.session_state["analysis_system_prompt"] + analysis_context},  
                                {"role": "user", "content": st.session_state["analysis_init_prompt"] + "\n-------Transcript-------\n"+ st.session_state["transcript"]}
                            ]
                        )
                        st.session_state['analysis'] = response.choices[0].message.content
                        st.success("Analysis Complete! - You can now generate your own story by pressing on the 'Storytelling' tab!")
                    except Exception as e:
                        st.error("Analysis failed - please re-interview")

            if 'analysis' in st.session_state:
                if st.button("📄 View Analysis"):
                    st.session_state['view_analysis'] = True

                if st.session_state.get('view_analysis'):
                    with st.spinner("Loading analysis..."):
                        st.text_area("Analysis:", value=st.session_state['analysis'], height=300)
                
                st.download_button(
                label="💾 Download Analysis",
                data=st.session_state['analysis'],
                file_name=f"{name}_interview_analysis.txt",
                mime="text/plain"
            )

with tab2:
    if 'analysis' not in st.session_state:
        st.subheader("⚠️INTERVIEW NOT FOUND!", divider = "red")
        st.markdown("#### *Please complete an interview in the previous tab so it can be analysed for storytelling or upload your own transcript below!*")

        user_transcript = st.file_uploader(
        "Upload your own transcript:", 
        type=["txt"], 
        accept_multiple_files=False
        )
        if user_transcript:
           user_transcript_text = user_transcript.read().decode("utf-8")
           st.session_state['transcript'] = user_transcript_text
           with st.spinner("Analysing your interview..."):
                analysis_context = ""
                for selected_title in st.session_state["analysis_selected_files"]:
                    for file_obj in st.session_state["titled_prereq_files"]:
                        if file_obj["title"] == selected_title:
                            analysis_context += f"\n\n----------{file_obj['title']}----------\n"
                            analysis_context += file_obj["content"]
                try:
                    response = client.chat.completions.create(
                        model=st.session_state["analysis_model"],
                        messages=[
                            {"role": "system", "content": st.session_state["analysis_system_prompt"] + analysis_context},  
                            {"role": "user", "content": st.session_state["analysis_init_prompt"] + "\n-------Transcript-------\n"+ st.session_state["transcript"]}
                        ]
                    )
                    st.session_state['analysis'] = response.choices[0].message.content
                    st.success("Analysis Complete! - You can now generate your own story!")
                    if 'analysis' in st.session_state:
                        if st.button("➡️ Proceed to story generation"):
                            st.session_state['view_analysis'] = True

                        if st.session_state.get('view_analysis'):
                            with st.spinner("Loading analysis..."):
                                st.text_area("Analysis:", value=st.session_state['analysis'], height=300)
                except Exception as e:
                    st.error("Analysis failed - please re-interview")

    else:
        st.title("📔Story Generation")
        if 'narrative' not in st.session_state:
            with st.spinner("Generating narrative..."):
                try:
                    response = client.chat.completions.create(
                            model=st.session_state["story_model_select"],
                            messages=[
                                {"role": "system", "content": st.session_state["narrative_system_prompt"]},
                                {"role": "user", "content": st.session_state["narrative_init_prompt"] + "\n-------Analysis-------\n"+ st.session_state["analysis"] + "\n-------Transcript-------\n" + st.session_state["transcript"]}
                            ]
                        ) 
                    st.session_state['narrative'] = response.choices[0].message.content
                    st.success("Narrative Created! - You can now generate more stories!")
                except Exception as e:
                    st.error("Story Generation Failed - Please try again...")
                    logging.error(f"Story generation error: {e}")

        if 'narrative' in st.session_state:
            with st.expander("View your Narrative", expanded=True):
                st.text_area("Your Narrative Story", value=st.session_state['narrative'], height=500)
            # story_option = st.selectbox(
            #     "What type of Story would you like to generate?",
            #     ("Adult's Story", "Children's Story (age 3-5)"),
            # )
            if voice_options:
                selected_voice_name = st.selectbox("Select Voice:", list(voice_options.keys()))
                voice_id = voice_options[selected_voice_name]
            if st.button("Generate Children's Story (age 3-5)"):   
                st.session_state['generate_child_story'] = True
                # if story_option == "Adult's Story":
                #     st.session_state['generate_adult_story'] = True

                # if story_option == "Children's Story (age 3-5)":
                #     st.session_state['generate_child_story'] = True


                #FOR WHEN WE HAVE AN ADULT STORY FRAMEWORK THIS WILL DO THE WHOLE REGEN 3 TIMES THING
                # if st.session_state.get('generate_adult_story'):
                #     with st.spinner("Writing your story..."):
                #         adult_story_context = ""
                #         for selected_title in st.session_state["adult_story_files"]:
                #             for file_obj in st.session_state["titled_prereq_files"]:
                #                 if file_obj["title"] == selected_title:
                #                     adult_story_context += f"\n\n----------{file_obj['title']}----------\n"
                #                     adult_story_context += file_obj["content"]

                #         attempts = 0
                #         story_ok = False
                #         story_text = None
                #         failure_reason = None  
                #         while attempts < 3 and not story_ok:
                #             try:
                #                 system_prompt = st.session_state["adult_system_prompt"] + adult_story_context
                #                 if failure_reason:  # inject failure feedback
                #                     system_prompt += f"\n\nThe last attempt was rejected for this reason: {failure_reason}\nPlease improve the story accordingly."
                #                 # Step 1 - Generate story
                #                 response = client.chat.completions.create(
                #                     model=st.session_state["story_model_select"],
                #                     messages=[
                #                         {"role": "system", "content": system_prompt},
                #                         {"role": "user", "content": st.session_state["adult_init_prompt"] +
                #                         "\n-------Analysis-------\n" + st.session_state["analysis"] +
                #                         "\n-------Narrative-------\n" + st.session_state["narrative"]}
                #                     ]
                #                 )
                #                 story_text = response.choices[0].message.content

                #                 # Step 2 - Evaluate story
                #                 eval_result = check_story_quality(story_text)
                #                 if eval_result.get("adequate", True):
                #                     story_ok = True
                #                 else:
                #                     attempts += 1
                #                     failure_reason = eval_result.get("reason", "No reason given.")
                #                     if attempts < 3:
                #                         continue  # try regenerating
                #             except Exception as e:
                #                 st.error("Story Generation Failed - Please try again...")
                #                 break

                #         # Step 3 - Return final story (either passed or last attempt)
                #         if story_text:
                #             st.session_state['adult_story'] = story_text
                #             play_sound(st.session_state['adult_story'], key="adult_voice", voice_id=voice_id)
                #             st.success(f"Story Created! on attempt {attempts} - Enjoy!")
                #         st.session_state['generate_adult_story'] = False

                # if st.session_state.get('generate_adult_story'):
                #     with st.spinner("Writing your story..."):
                #         adult_story_context = ""
                #         for selected_title in st.session_state["adult_story_files"]:
                #             for file_obj in st.session_state["titled_prereq_files"]:
                #                 if file_obj["title"] == selected_title:
                #                     adult_story_context += f"\n\n----------{file_obj['title']}----------\n"
                #                     adult_story_context += file_obj["content"]
                #         try:
                #             response = client.chat.completions.create(
                #                     model=st.session_state["story_model_select"],
                #                     messages=[
                #                         {"role": "system", "content": st.session_state["adult_system_prompt"] + adult_story_context},
                #                         {"role": "user", "content": st.session_state["adult_init_prompt"] + "\n-------Analysis-------\n"+ st.session_state["analysis"]+ "\n-------Narrative-------\n"+ st.session_state["narrative"]}
                #                     ]
                #                 )
                #             st.session_state['adult_story'] = response.choices[0].message.content
                #             play_sound(st.session_state['adult_story'], key="adult_voice", voice_id=voice_id)
                #             st.success("Story Created! - Enjoy!")
                #             st.session_state['generate_adult_story'] = False
                #         except Exception as e:
                #             st.error("Story Generation Failed - Please try again...")

                if st.session_state.get('generate_child_story'):
                    with st.spinner("Writing your story..."):
                        child_story_context = ""
                        for selected_title in st.session_state["child_story_files"]:
                            for file_obj in st.session_state["titled_prereq_files"]:
                                if file_obj["title"] == selected_title:
                                    child_story_context += f"\n\n----------{file_obj['title']}----------\n"
                                    child_story_context += file_obj["content"]

                        attempts = 0
                        story_ok = False
                        story_text = None
                        failure_reason = None
                        while attempts < 3 and not story_ok:
                            try:
                                system_prompt = st.session_state["child_system_prompt"] + child_story_context
                                if failure_reason:  # inject failure feedback
                                    system_prompt += f"\n\nThe last attempt was rejected for this reason: {failure_reason}\nPlease improve the story accordingly."
                                # Step 1 - Generate story
                                response = client.chat.completions.create(
                                    model=st.session_state["story_model_select"],
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": st.session_state["child_init_prompt"] +
                                        "\n-------Analysis-------\n" + st.session_state["analysis"] +
                                        "\n-------Narrative-------\n" + st.session_state["narrative"]}
                                    ]
                                )
                                story_text = response.choices[0].message.content

                                # Step 2 - Evaluate story
                                eval_result = check_story_quality(story_text)
                                if eval_result.get("adequate", True):
                                    story_ok = True
                                else:
                                    attempts += 1
                                    failure_reason = eval_result.get("reason", "No reason given.")
                                    if attempts < 3:
                                        continue  # try regenerating
                            except Exception as e:
                                st.error("Story Generation Failed - Please try again...")
                                break

                        # Step 3 - Return final story (either passed or last attempt)
                        if story_text:
                            st.session_state['child_story'] = story_text
                            play_sound(st.session_state['child_story'], key="child_voice", voice_id=voice_id)
                            st.success(f"Story Created! on attempt {attempts} - Enjoy!")
                        st.session_state['generate_child_story'] = False


    # Show stories if they exist
    # if 'adult_story' in st.session_state:
    #     with st.container():
    #         st.text_area("Your Adult Story", value=st.session_state['adult_story'], height=500)
    #         if "adult_voice_audio" in st.session_state:
    #             st.audio(st.session_state["adult_voice_audio"], format="audio/mp3")

    if 'child_story' in st.session_state:
        with st.container():
            st.text_area("Your Children's Story", value=st.session_state['child_story'], height=500)
            if "child_voice_audio" in st.session_state:
                st.audio(st.session_state["child_voice_audio"], format="audio/mp3")

    #test report
    if 'narrative' in st.session_state:
        st.download_button(
                label="📑 Download Narrative",
                data=st.session_state['narrative'],
                file_name=f"{name}_interview_narrative.txt",
                mime="text/plain"
            )
        if "child_story" in st.session_state:
            st.download_button(
                label="📙 Download Children's Story",
                data=st.session_state['child_story'],
                file_name=f"{name}_interview_child_story.txt",
                mime="text/plain"
            )
        
        if st.button("📑Generate and Download your testing report"):
            # Step 1: Generate text
            text_result = generate_testing_report()

            # Step 2: Prepare file in memory for download
            file_name = f"{st.session_state.get('interview_name', 'JohnDoe')}_testing_report.txt"
            file_bytes = BytesIO(text_result.encode("utf-8"))

            # Step 3: Show download button
            st.download_button(
                label="💾 Download your testing report",
                data=file_bytes,
                file_name=file_name,
                mime="text/plain"
            )
    
with tab3:
    st.title("Settings")    
    st.subheader("Pre-Requisite Files")

    uploaded_prereq_files = st.file_uploader(
        "Upload prerequisite context text files for prompts - multiple files allowed!", 
        type=["txt"], 
        accept_multiple_files=True
    )

    if uploaded_prereq_files:
        st.session_state["user_uploaded_prereq_files"] = True  

        if "file_titles" not in st.session_state:
            st.session_state.file_titles = {}

        st.markdown("#### Title each uploaded file")

        current_uploaded_names = set(file.name for file in uploaded_prereq_files)

        # Remove stale keys for deleted files
        st.session_state.file_titles = {
            k: v for k, v in st.session_state.file_titles.items()
            if any(k.startswith(f"title_{name}_") for name in current_uploaded_names)
        }

        titled_files = []
        for i, file in enumerate(uploaded_prereq_files):
            title_key = f"title_{file.name}_{i}"
            st.markdown(f"**File:** {file.name}")
            title = st.text_input("Enter a title for this file", key=title_key)
            st.session_state.file_titles[title_key] = title

            if title.strip():
                file.seek(0)
                content = file.read().decode("utf-8")
                titled_files.append({"title": title.strip(), "content": content})

        if titled_files:
            st.session_state["titled_prereq_files"] = titled_files

    #Only clear if files were uploaded previously and now none remain
    elif st.session_state.get("user_uploaded_prereq_files") and not uploaded_prereq_files:
        st.session_state.pop("file_titles", None)
        st.session_state.pop("titled_prereq_files", None)
        st.session_state.pop("user_uploaded_prereq_files", None)


    st.markdown("### Text-to-Speech Settings")
    if voice_options:
        selected_voice_name = st.selectbox("Select Interviewer Voice:", list(voice_options.keys()))
        st.session_state.interviewer_voiceid = voice_options[selected_voice_name]
    model_list = get_elevenlabs_model_list()
    if model_list:
        selected_model_id = st.selectbox(
            "Select TTS Model:",
            model_list,
            key="TTS_model"
        )
    tab1s, tab2s, tab3s, tab4s = st.tabs(["🎤Interviewer Settings","🤖Assistant Bot Settings", "📈Analysis Settings"," 📑Storyteller Settings"])
    if "titled_prereq_files" in st.session_state:
         prereq_titles = [f["title"] for f in st.session_state["titled_prereq_files"]]
    else:
        prereq_titles = []
    with tab1s:
        st.subheader("Interviewer Settings")
        with st.expander("Expand to edit Interview prompts:"):
            
            st.text_area("Interviewer Prompt:", key = "interview_prompt", height = 350)
            
            st.text_area("Interviewer's First Question:",key = "first_question")
        st.selectbox("Select a model", chat_models, key="interviewer_model")

        st.multiselect(
            "Select prerequisite files for Interviewer",
            options=prereq_titles,
            key="interview_selected_files"
        )

    with tab2s:
        st.subheader("Assistant Bot Settings")
        with st.expander("Expand to view steering prompts:"):
            steering_model = st.selectbox("Select a steering model", chat_models, key = "steering_model")
            steering_prompt = st.text_area("Interview Steering Prompt:",key = "steering_prompt",height = 350)
        with st.expander("Expand to view safeguarding prompt"):
            steering_model = st.selectbox("Select a safeguarding model", chat_models, key = "safeguarding_model")
            steering_prompt = st.text_area("Interview Safeguarding Prompt:",key = "safeguarding_prompt",height = 350)

    with tab3s:
        st.subheader("Analysis Settings")
        with st.expander("Expand to edit Analysis prompts:"):
            analysis_system_prompt = st.text_area("Analysis System Configuration Prompt:", key = "analysis_system_prompt",height = 350 )
            analysis_init_prompt = st.text_area("Analysis Initiation Prompt:", key ="analysis_init_prompt")
        st.selectbox("Select a model", chat_models, key="analysis_model")
        st.multiselect(
            "Select prerequisite files for Analysis",
            options=prereq_titles,
            key="analysis_selected_files"
        )

    with tab4s:
        st.subheader("Story Generation Settings")
        st.selectbox("Select a model for storytelling", chat_models, key="story_model_select")

        tab1ns, tab3cs = st.tabs(["📃Narrative Settings", "🧒Children's Stories"])
        with tab1ns:
            st.text_area("Narrative System Prompt:", key = "narrative_system_prompt",height = 350)
            st.text_area("Narrative Initialisation Prompt",key = "narrative_init_prompt")

        # with tab2as:
        #     st.text_area("Adult Story System Prompt:", key = "adult_system_prompt",height = 350)
        #     st.text_area("Adult Story Initialisation Prompt",key = "adult_init_prompt")

        #     st.multiselect(
        #     "Select prerequisite files for Adult Story Generation",
        #     options=prereq_titles,
        #     key="adult_story_files"
        # )

        with tab3cs:
            st.text_area("Children's Story System Prompt:", key = "child_system_prompt",height = 350)
            st.text_area("Children's Story Initialisation Prompt",key = "child_init_prompt")

            st.multiselect(
            "Select prerequisite files for Child Story Generation",
            options=prereq_titles,
            key="child_story_files"
        )
           