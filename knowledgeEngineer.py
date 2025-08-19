import streamlit as st
from openai import OpenAI
import logging 
import streamlit.components.v1 as components
from html import escape
from io import StringIO
import csv
import os

# Configure logging level and format

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

st.set_page_config(page_title="Knowledge Engineering", page_icon="🧠", layout="centered")
st.title("Knowledge Engineering")


#AI Model Configuration
client = OpenAI()

def generate_transcript(messages):
    transcript = ""
    for msg in messages[1:]:  # skip initial system prompt
        if msg["role"] == "system":
            continue
            #role = "System"
        elif msg["role"] == "assistant":
            role = "Interviewer"
        else:
            role = st.session_state["interviewee"]
        transcript += f"{role}: {msg['content']}\n\n"
    return transcript

def read_csv():
    DEFAULT_FILE_PATH = "engineering_prompts.csv"
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

prompt_list = read_csv()
if not prompt_list:
    st.error("Failed to read prompts.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config_initialized" not in st.session_state:
    if prompt_list and len(prompt_list) < 3:

        st.error("Insufficient prompts in Default CSV configuration. Please ensure at least 2 entries.")
        st.stop()

    st.session_state.update({
        "interview_prompt": prompt_list[0][1],
        "first_question": prompt_list[1][1],
        "json_prompt": prompt_list[2][1],

        "config_initialized": True
    })

with st.expander("⚙️ Settings"):
    st.text_area("Interviewer Prompt:", key="interview_prompt", height=350)
    st.text_area("Interviewer's First Question:", key="first_question")
    st.text_area("JSON Analysis Prompt:", key="json_prompt", height=350)

st.subheader("Interview:")
st.session_state["interviewee"] = st.text_input("Enter your Name:")

if st.button("🎤 Begin Interview") and st.session_state["interviewee"]:

    st.session_state["interview_in_progress"] = True
    st.session_state["json_generated"] = False  
    
    name = st.session_state["interviewee"]
    formatted_system_prompt = st.session_state["interview_prompt"].format(name=name)
    formatted_first_question = st.session_state["first_question"].format(name=name)

    st.session_state.messages = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "assistant", "content": formatted_first_question}
    ]

if 'interview_in_progress' in st.session_state:
    #scrollable chatUI
    st.subheader("💬 Interview Chat")
    inner = ""
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "system":
            continue
        role = "🧑‍💼 Interviewer" if msg["role"] == "assistant" else f"🙋 {st.session_state['interviewee']}"
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

    # Render chat + auto-scroll
    components.html(chat_html, height=420, scrolling=False)


    if prompt := st.chat_input("Type your reply..."):
        # 1. Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model= "gpt-4o",
                    messages=st.session_state.messages,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = "Sorry, there was an issue generating a response."
                st.error(f"Error: {e}")

        # 6. Append interviewer message
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # 7. Refresh UI
        st.rerun()


    # End interview and download buttons
    if st.button("🛑 End Interview"):
        st.session_state["interview_in_progress"] = False
        st.session_state["transcript_text"] = generate_transcript(st.session_state.messages)
        st.session_state.messages = [
            {"role": "assistant", "content": f"Thank you {st.session_state['interviewee']}. This concludes our interview."}
        ]
        st.success("Interview ended. You can download your transcript below.")
        
        with st.spinner("Analysing your interview..."):
            try:
                response = client.chat.completions.create(
                        model="gpt-4o",
                        response_format={ "type": "json_object" },
                        messages=[
                            {"role": "system", "content": st.session_state['json_prompt']},
                            {"role": "user", "content": "Analyse the following transcript as specified using the framework above:\n" + st.session_state["transcript_text"]}
                        ]
                    )
                st.session_state['json_content'] = response.choices[0].message.content
                st.session_state['json_generated'] = True
            except Exception as e:
                reply = "Sorry, there was an issue generating a response."
                st.error(f"Error: {e}")
        
# Show download buttons only after interview ended
if not st.session_state.get("interview_in_progress", True):

    if "transcript_text" in st.session_state:
        st.download_button(
            label="📥 Download Transcript",
            data=st.session_state["transcript_text"],
            file_name=f"{st.session_state['interviewee']}_interview_transcript.txt",
            mime="text/plain"
        )

    if st.session_state.get('json_generated', False) and "json_content" in st.session_state:
        st.download_button(
            label="📥 Download JSON",
            data=st.session_state['json_content'],
            file_name=f"{st.session_state['interviewee']}_interview_analysis.json",
            mime="application/json"
        )




            

