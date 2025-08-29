import streamlit as st
from openai import OpenAI
from io import StringIO
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
from io import BytesIO
import base64

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

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

check_password()

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
        st.error(f"Failed to fetch models: {e}")
        return []

def get_elevenlabs_model_list():
    try:
        models = voice_client.models.list()
        return sorted([model.model_id for model in models])
    except Exception as e:
        st.error(f"Failed to fetch ElevenLabs models: {e}")
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
            return f"ERROR reading file from path: {e}"
    try:
        return StringIO(input_file.getvalue().decode("utf-8")).read()
    except Exception as e:
        return f"ERROR reading uploaded file: {e}"

def generate_transcript(messages, user_name="User"):
    transcript = ""
    for msg in messages[1:]:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "assistant":
            role = "Interviewer"
        else:
            role = user_name

        content = msg["content"]
        transcript += f"{role}: {content}\n\n"
    return transcript

def json_check(text):
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
            st.error(f"Error reading CSV data: {e}")
            return

def autoplay_html_audio(audio_obj, mime_type="audio/mp3"):
    if hasattr(audio_obj, "__iter__") and not isinstance(audio_obj, (bytes, bytearray)):
        audio_bytes = b"".join(audio_obj)
    elif hasattr(audio_obj, "read"):
        audio_bytes = audio_obj.read()
    elif hasattr(audio_obj, "content"):
        audio_bytes = audio_obj.content
    else:
        audio_bytes = audio_obj

    b64 = base64.b64encode(audio_bytes).decode()
    html_audio = f"""
    <audio autoplay="true" style="display:none;">
        <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
    </audio>
    """
    components.html(html_audio, height=0)

def play_sound(text, voice_id):
    try:
        response = voice_client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=st.session_state["TTS_model"],
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.1,
                use_speaker_boost=True
            )
        )
        audio_bytes = b"".join(response)
        autoplay_html_audio(audio_bytes)
    except Exception as e:
        st.error(f"Error occurred while playing sound: {e}")

prompt_list = read_csv()
if not prompt_list:
    st.error("Failed to read prompts.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config_initialized" not in st.session_state:
    if prompt_list and len(prompt_list) < 17:
        st.error("Insufficient prompts in Default CSV configuration. Please ensure at least 17 entries.")
        st.stop()

    init_file_data = []
    init_file_data.append({"title": prompt_list[10][0], "content": read_file(prompt_list[10][1])})
    init_file_data.append({"title": prompt_list[11][0], "content": read_file(prompt_list[11][1])})
    init_file_data.append({"title": prompt_list[12][0], "content": read_file(prompt_list[12][1])})

    st.session_state.update({
        "titled_prereq_files": init_file_data,
        "interview_prompt": prompt_list[0][1],
        "first_question": prompt_list[1][1],
        "interviewer_model": default_model,
        "interview_selected_files": [prompt_list[10][0], prompt_list[11][0], prompt_list[12][0]],
        "analysis_system_prompt": prompt_list[2][1],
        "analysis_init_prompt": prompt_list[3][1],
        "analysis_model": default_model,
        "analysis_selected_files": [],
        "narrative_system_prompt": prompt_list[15][1],
        "narrative_init_prompt": prompt_list[16][1],
        "adult_system_prompt": prompt_list[4][1],
        "adult_init_prompt": prompt_list[5][1],
        "adult_story_files": [prompt_list[10][0], prompt_list[11][0], prompt_list[12][0]],
        "child_system_prompt": prompt_list[6][1],
        "child_init_prompt": prompt_list[7][1],
        "child_story_files": [prompt_list[10][0], prompt_list[11][0], prompt_list[12][0]],
        "eyfs_system_prompt": prompt_list[8][1],
        "eyfs_init_prompt": prompt_list[9][1],
        "eyfs_story_files": [prompt_list[10][0], prompt_list[11][0], prompt_list[12][0]],
        "steering_prompt": prompt_list[13][1],
        "safeguarding_prompt": prompt_list[14][1],
        "story_model_select": default_model,
        "steering_model": default_model,
        "safeguarding_model": default_model,
        "story_stages": {
            "Moment": False,
            "Details": False,
            "Realisation": False,
            "Change": False,
            "Resolution": False
        },
        "safeguarding_flag": False,
        "config_initialized": True,
        "TTS_model": "eleven_multilingual_v2"
    })

tab1, tab2, tab3 = st.tabs(["🗨️Interview", "📚Storytelling", "⚙️ Configuration"])

with tab1:
    st.title("🗣️Interviewer")
    name = st.text_input("Enter your Name")
    
    speak_flag = False
    if voice_options:
        selected_voice_name = st.selectbox("Select Voice:", list(voice_options.keys()))
        st.session_state.interview_voice_id = voice_options[selected_voice_name]
        speak_flag = st.checkbox("🔊 Speak replies aloud", key="speak_flag", value=True)
    else:
        st.info("No voices found in your ElevenLabs account. Please add a voice in ElevenLabs to enable text-to-speech features.")

    if st.button("🎤 Begin Interview"):
        if not name:
            st.warning("Please enter your name to start the interview.")
        else:
            st.session_state["interview_name"] = name
            for key in ["messages", "transcript", "analysis", "view_analysis", "interview_ended", "story_stages"]:
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
            st.session_state.interview_ended = False
            st.session_state["interview_start_time"] = datetime.now()

            st.success("Interview started!")

            if speak_flag:
                with st.spinner("Generating voice..."):
                    play_sound(interview_question, st.session_state.interview_voice_id)

    # Display chat history with native elements
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        
        role = "🧑‍💼 Interviewer" if msg["role"] == "assistant" else f"🙋 {st.session_state.get('interview_name', 'User')}"
        
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "assistant" else "🙋"):
            st.markdown(msg["content"])
            
    # Handle chat input
    if st.session_state.get("messages") and not st.session_state.get("interview_ended", False):
        st.markdown("### Your Reply")

        col1, col2 = st.columns([2,1])
        with col1:
            text_prompt = st.chat_input("Type your reply...")
        with col2:
            audio_prompt = st.file_uploader("🎙️ Speak your reply", type=["mp3", "m4a", "wav"], key="audio_uploader")

        prompt = None
        if text_prompt:
            prompt = text_prompt
        elif audio_prompt:
            with st.spinner("Transcribing..."):
                transcription = voice_client.speech_to_text.convert(
                    file=audio_prompt,
                    model_id="eleven_multilingual_v2"
                )
                prompt = transcription.text

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    # Generate and handle assistant reply after user's prompt is appended
    if st.session_state.get("messages") and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Thinking..."):
            temp_messages = st.session_state.messages.copy()
            try:
                response = client.chat.completions.create(
                    model=st.session_state["interviewer_model"],
                    messages=temp_messages,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = "Sorry, there was an issue generating a response."
                st.error(f"Error: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
            if speak_flag:
                with st.spinner("Generating voice..."):
                    play_sound(reply, st.session_state.interview_voice_id)
            
            st.rerun()

    # End interview
    if st.session_state.get("messages") and not st.session_state.get("interview_ended"):
        if st.button("🛑 End Interview"):
            st.session_state.interview_ended = True
            st.rerun()

    if st.session_state.get("interview_ended"):
        if 'transcript' not in st.session_state:
            st.session_state["interview_end_time"] = datetime.now()
            interview_length = st.session_state["interview_end_time"] - st.session_state["interview_start_time"]
            total_seconds = interview_length.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            
            transcript_text = generate_transcript(st.session_state.messages, user_name=st.session_state.get("interview_name", "User"))
            st.session_state['transcript'] = transcript_text + f"Interview length: {minutes} minutes, {seconds} seconds"
            
            final_message = f"Thank you {st.session_state.get('interview_name', 'User')} for sharing your story. This concludes our interview."
            st.session_state.messages.append({"role": "assistant", "content": final_message})

        st.success("Interview ended. You can download your transcript below.")
        st.info("Please wait until analysis has finished before generating a story...")

        st.download_button(
            label="📥 Download Transcript",
            data=st.session_state['transcript'],
            file_name=f"{st.session_state.get('interview_name', 'User')}_interview_transcript.txt",
            mime="text/plain"
        )

        if 'analysis' not in st.session_state:
            with st.spinner("Analysing your interview..."):
                analysis_context = ""
                for selected_title in st.session_state.get("analysis_selected_files", []):
                    for file_obj in st.session_state.get("titled_prereq_files", []):
                        if file_obj["title"] == selected_title:
                            analysis_context += f"\n\n----------{file_obj['title']}----------\n"
                            analysis_context += file_obj["content"]
                try:
                    response = client.chat.completions.create(
                        model=st.session_state.get("analysis_model", default_model),
                        messages=[
                            {"role": "system", "content": st.session_state.get("analysis_system_prompt", '') + analysis_context},
                            {"role": "user", "content": st.session_state.get("analysis_init_prompt", '') + "\n-------Transcript-------\n" + st.session_state["transcript"]}
                        ]
                    )
                    st.session_state['analysis'] = response.choices[0].message.content
                    st.success("Analysis Complete! - You can now generate your own story!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if 'analysis' in st.session_state:
            if st.button("📄 View Analysis"):
                st.session_state['view_analysis'] = not st.session_state.get('view_analysis', False)
                st.rerun()

            if st.session_state.get('view_analysis'):
                st.text_area("Analysis:", value=st.session_state['analysis'], height=300, disabled=True)
            
            st.download_button(
                label="💾 Download Analysis",
                data=st.session_state['analysis'],
                file_name=f"{st.session_state.get('interview_name', 'User')}_interview_analysis.txt",
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
                    st.error(f"Error: {e}")

        if 'narrative' in st.session_state:
            with st.expander("View your Narrative", expanded=True):
                st.text_area("Your Narrative Story", value=st.session_state['narrative'], height=500)
            story_option = st.selectbox(
                "What type of Story would you like to generate?",
                ("Adult's Story", "Children's Story", "EYFS Story"),
            )
            if voice_options:
                selected_voice_name = st.selectbox("Select Voice:", list(voice_options.keys()))
                voice_id = voice_options[selected_voice_name]
            if st.button("Generate Story"):   
                if story_option == "Adult's Story":
                    st.session_state['generate_adult_story'] = True

                if story_option == "Children's Story":
                    st.session_state['generate_child_story'] = True
                        
                if story_option == "EYFS Story":
                    st.session_state['generate_eyfs_story'] = True 

                if st.session_state.get('generate_adult_story'):
                    with st.spinner("Writing your story..."):
                        adult_story_context = ""
                        for selected_title in st.session_state["adult_story_files"]:
                            for file_obj in st.session_state["titled_prereq_files"]:
                                if file_obj["title"] == selected_title:
                                    adult_story_context += f"\n\n----------{file_obj['title']}----------\n"
                                    adult_story_context += file_obj["content"]
                        try:
                            response = client.chat.completions.create(
                                    model=st.session_state["story_model_select"],
                                    messages=[
                                        {"role": "system", "content": st.session_state["adult_system_prompt"] + adult_story_context},
                                        {"role": "user", "content": st.session_state["adult_init_prompt"] + "\n-------Analysis-------\n"+ st.session_state["analysis"]+ "\n-------Narrative-------\n"+ st.session_state["narrative"]}
                                    ]
                                )
                            st.session_state['adult_story'] = response.choices[0].message.content
                            play_sound(st.session_state['adult_story'], key="adult_voice", voice_id=voice_id)
                            st.success("Story Created! - Enjoy!")
                            st.session_state['generate_adult_story'] = False
                        except Exception as e:
                            st.error("Story Generation Failed - Please try again...")

                if st.session_state.get('generate_child_story'):
                    with st.spinner("Writing your story..."):
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
                                        {"role": "user", "content": st.session_state["child_init_prompt"] + "\n-------Analysis-------\n"+ st.session_state["analysis"] + "\n-------Narrative-------\n"+ st.session_state["narrative"]}
                                    ]
                                )
                            st.session_state['child_story'] = response.choices[0].message.content
                            play_sound(st.session_state['child_story'], key="child_voice", voice_id=voice_id)
                            st.success("Story Created! - Enjoy!")
                            st.session_state['generate_child_story'] = False
                        except Exception as e:
                            st.error("Story Generation Failed - Please try again...")

                    

                if st.session_state.get('generate_eyfs_story'):
                    with st.spinner("Writing your story..."):
                        eyfs_story_context = ""
                        for selected_title in st.session_state["eyfs_story_files"]:
                            for file_obj in st.session_state["titled_prereq_files"]:
                                if file_obj["title"] == selected_title:
                                    eyfs_story_context += f"\n\n----------{file_obj['title']}----------\n"
                                    eyfs_story_context += file_obj["content"]
                        try:
                            response = client.chat.completions.create(
                                    model=st.session_state["story_model_select"],
                                    messages=[
                                        {"role": "system", "content": st.session_state["eyfs_system_prompt"] + eyfs_story_context},
                                        {"role": "user", "content": st.session_state["eyfs_init_prompt"] + "\n-------Analysis-------\n"+ st.session_state["analysis"]+ "\n-------Narrative-------\n"+ st.session_state["narrative"]}
                                    ]
                                )
                            st.session_state['eyfs_story'] = response.choices[0].message.content
                            play_sound(st.session_state['eyfs_story'], key="eyfs_voice", voice_id=voice_id)
                            st.success("Story Created! - Enjoy!")
                            st.session_state['generate_eyfs_story'] = False
                        except Exception as e:
                            st.error("Story Generation Failed - Please try again...")

    # Show stories if they exist
    if 'adult_story' in st.session_state:
        with st.container():
            st.text_area("Your Adult Story", value=st.session_state['adult_story'], height=500)
            if "adult_voice_audio" in st.session_state:
                st.audio(st.session_state["adult_voice_audio"], format="audio/mp3")

    if 'child_story' in st.session_state:
        with st.container():
            st.text_area("Your Children's Story", value=st.session_state['child_story'], height=500)
            if "child_voice_audio" in st.session_state:
                st.audio(st.session_state["child_voice_audio"], format="audio/mp3")


    if 'eyfs_story' in st.session_state:
        with st.container():
            st.text_area("Your EYFS Story", value=st.session_state['eyfs_story'], height=500)
            if "eyfs_voice_audio" in st.session_state:
                st.audio(st.session_state["eyfs_voice_audio"], format="audio/mp3")

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


    st.markdown("### Select Text-to-Speech model")
    model_list = get_elevenlabs_model_list()
    if model_list:
        selected_model_id = st.selectbox(
            "Select Model:",
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

        tab1ns, tab2as, tab3cs, tab4es = st.tabs(["📃Narrative Settings", "👨‍🦰Adult Stories", "🧒Children's Stories"," 👶EYFS Stories"])
        with tab1ns:
            st.text_area("Narrative System Prompt:", key = "narrative_system_prompt",height = 350)
            st.text_area("Narrative Initialisation Prompt",key = "narrative_init_prompt")

        with tab2as:
            st.text_area("Adult Story System Prompt:", key = "adult_system_prompt",height = 350)
            st.text_area("Adult Story Initialisation Prompt",key = "adult_init_prompt")

            st.multiselect(
            "Select prerequisite files for Adult Story Generation",
            options=prereq_titles,
            key="adult_story_files"
        )

        with tab3cs:
            st.text_area("Children's Story System Prompt:", key = "child_system_prompt",height = 350)
            st.text_area("Children's Story Initialisation Prompt",key = "child_init_prompt")

            st.multiselect(
            "Select prerequisite files for Child Story Generation",
            options=prereq_titles,
            key="child_story_files"
        )
           
        with tab4es:
            st.text_area("EYFS Story System Prompt:", key = "eyfs_system_prompt",height = 350)
            st.text_area("EYFS Story Initialisation Prompt",key = "eyfs_init_prompt")
            st.multiselect(
            "Select prerequisite files for EYFS Story Generation",
            options=prereq_titles,
            key="eyfs_story_files"
        )
           