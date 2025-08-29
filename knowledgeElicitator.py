import streamlit as st
from openai import OpenAI
import io
import csv
import re
import json
import logging 
import streamlit.components.v1 as components
from html import escape

# Configure logging level and format

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

st.set_page_config(page_title="Knowledge Elicitator", page_icon="🧠", layout="centered")
st.title("Knowledge Elicitation")


#AI Model Configuration
client = OpenAI()

#functions
def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

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

def analyze_story_stages(messages, stage_info):
    transcript = generate_transcript(messages)
    formatted_steering_prompt = st.session_state["steering_prompt"].format(stage = stage_info['question'])
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": formatted_steering_prompt},
            {"role": "user", "content": "Carry out the analysis as specified in the framework above using this transcript:" +transcript}])

    result = response.choices[0].message.content
    logging.debug("Raw analysis result: %s", result)
    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decoding failed: {e}")
        return {} 

def generate_csv(prompt, transcript):
    for attempt in range(5):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"""By following the exact framework specified above, return only the CSV file contents, nothing else.
                 Interview questions (ordered):  
                    {st.session_state.stage_path}  
                Transcript:  
                {transcript} """}])

        csv_text = response.choices[0].message.content.strip()
        csv_text = re.sub(r"^```(?:csv)?\s*|\s*```$", "", csv_text, flags=re.MULTILINE).strip()

        # Try parsing with csv reader
        try:
            reader = csv.reader(io.StringIO(csv_text))
            rows = list(reader)

            # Validate: at least 2 rows (header + 1 row) and 2+ columns
            if len(rows) < 2:
                raise ValueError("CSV contained no data rows")
            if len(rows[0]) < 2:
                raise ValueError("CSV did not contain at least 2 columns")

            return csv_text

        except Exception as e:
            logging.warning(f"Attempt {attempt+1}/5: Invalid CSV, regenerating... ({e})")

    return None

#tree navigation functions
def calculate_max_depth(node):
    if "children" not in node or not node["children"]:
        return 1
    max_child_depth = 0
    for child in node["children"].values():
        max_child_depth = max(max_child_depth, calculate_max_depth(child))
    return 1 + max_child_depth

def path_to_questions(decision_tree, path):
    questions = []
    for node_id in path:
        node = get_next_stage(decision_tree, [node_id])
        if node and "question" in node:
            questions.append(node["question"])
    return questions

def get_next_stage(decision_tree, path):
    """
    Traverse tree by a list of node IDs, return the corresponding node dict.
    """
    def find_by_id(node, target_id):
        if node.get("id") == target_id:
            return node
        for child in node.get("children", {}).values():
            found = find_by_id(child, target_id)
            if found:
                return found
        return None

    node = decision_tree
    for node_id in path[1:]:  # skip root (already node)
        node = find_by_id(decision_tree, node_id) or {}
    return node if isinstance(node, dict) else {}

def move_to_next_stage(decision_tree, stage_path):
    transcript = generate_transcript(st.session_state.messages) 
    current_node = get_next_stage(decision_tree, stage_path)
    children = current_node.get("children", {})
    options_text = "\n".join([f"- {k}: {v['question']}" for k, v in children.items()])

    # Case 1: No children (leaf node)
    if not children:
        return current_node, stage_path

    #Case 2: 1 Child, just pick child
    if len(children) == 1:
        next_node = list(children.values())[0]
        stage_path.append(next_node["id"])
        return next_node, stage_path

    # Case 3: If multiple children, GPT picks key
    formatted_choice_prompt = st.session_state["choice_prompt"].format(tree_path=path_to_questions(st.session_state["decision_tree"], st.session_state["stage_path"]), transcript=transcript, options_text=options_text)
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": formatted_choice_prompt},
        ]
    )

    result = response.choices[0].message.content
    logging.debug("Tree path decision result: %s", result)
    try:
        parsed = json.loads(result)
        next_key = parsed.get("next_key")
        next_node = children.get(next_key, list(children.values())[0])
        stage_path.append(next_node["id"])
        return next_node, stage_path
    except Exception:
        # fallback to first child if GPT fails
        first_child = list(children.values())[0]
        next_node = first_child
        stage_path.append(next_node["id"])
        return next_node, stage_path

#default configuration
if "config_initialized" not in st.session_state:

    st.session_state.update({
        "interview_in_progress": False,
        "interview_ended": False,
        "stage_path": [],
        "decision_tree": {},
        "current_node": None,
        # Interviewer
        "interview_prompt": """You are a knowledge elicitation expert conducting an in-depth and structured interview with {name}, who is a student returning from their placement. 
By asking perceptive questions and following the framework outlined by this expertly crafted interview JSON decision tree {tree} , you should be able to gain in depth and perceptive answers from the student required from that question.
When interviewing the student, make sure your questions are put forward in a natural and conservational way while making sure it aligns with the current stage, ensuring you don’t invent any unrelated questions outside that current stage.

Your approach:
• Ask only one question at a time.
• Base each new question on {name}’s previous answer, ensuring a natural and logical progression as outlined in the interview framework.
• Ensure the first question you ask is what's in the decision tree.
• Use British English spelling, grammar, and punctuation throughout (e.g., realise, organisation).
• Identify the first question {name} will ask candidates, and any mandatory questions all candidates must answer.
• When asking the first question, greet the user using their name and thank them for taking the time to interview and ensure the question is clear and concise, setting the tone for the interview
• Ensure you are properly following the interview decision tree
• Keep interviewees on track and ensure consistent coverage of essential topics.
• You will be reminded by assistant agents what stage/question of the interview you are currently on, ensure you stick to that.
Your goal is to follow the interview framework to be able to elicit enough information that is of a high and insightful level in order to understand that students experiences on placements and how we can use their experiences to improve the experiences and prospects of future placement students.""",

        #assistant bots
        "steering_prompt": """You are an analysis expert evaluating interview answers given in the transcript so far.
    The interviewer is currently at the stage in the interview decision tree: {stage}.
    Determine if the interviewee's last response sufficiently answers this stage with a perceptive level of detail in order to be able to move on in the interview.
    Return ONLY JSON strictly in this format:
    {{"adequate": true/false, "reason": "short explanation that will tell the interviewer how to probe further to collect adequate information."}}""",

        "choice_prompt": """You are an expert interview analyst, monitoring an ongoing interview transcript helping to navigate a structured interview decision tree.

    The interview transcript so far:
    {transcript}

    The path the interview has taken through the decision tree so far is {tree_path}.

    You must choose the most relevant next branch from these options that best suits the direction of the interview:
    {options_text}

    Return ONLY a JSON object like:
    {{"next_key": "<selected_child_key>", "reason": "<short reason dictating why you have chosen this path.>"}}""",

        "csv_prompt": """You are an expert transcript analyst.  
Your task is to convert an interview transcript into a structured **CSV file**.  

### Rules:
1. Output **ONLY raw CSV text** (no explanations, no markdown, no code blocks).  
2. The CSV must have **exactly these headers**:  
   `question,answer`  
3. For each interview stage in the question set extract the matching information from the transcript.  
   - Column `question` = the interview question text.  
   - Column `answer`   = the interviewee’s response, based only on the interviewee's response in the transcript.  
4. If a question was not answered or not covered in the transcript, write `N/A` in the `answer` field.  
5. Do **not invent, summarise, or expand** beyond what is in the transcript.  
6. Ensure the CSV is valid and parsable — one row per question.  
7. Use plain text commas as delimiters. Escape any quotes or commas inside fields properly.""",
        "config_initialized": True
    })

#MAIN PROGRAM
tab1, tab2 = st.tabs(["🗨️Interview","⚙️ Configuration"])
with tab1:
    st.subheader("🎙️Interview:")
    uploaded_file = st.file_uploader("Upload your JSON decision tree here:", type="json")
    if uploaded_file:
        st.session_state.decision_tree = json.load(uploaded_file)
    st.session_state["interviewee"] = st.text_input("Enter your Name:")

    if st.button("🎤 Begin Interview") :
        if st.session_state["interviewee"] and st.session_state["decision_tree"]:
            st.session_state.update({
                "interview_in_progress": True,
                "interview_ended": False,
                "stage_path": [st.session_state.decision_tree["id"]],
                "current_node": st.session_state.decision_tree,
                "messages": [],
                "transcript": None,
                "csv_output": None,
                "finalised": False,   
            })
            name = st.session_state.get("interviewee", "")
            formatted_system_prompt = st.session_state["interview_prompt"].format(name=name, tree=st.session_state["decision_tree"])
            st.session_state.messages = [
                {"role": "system", "content": formatted_system_prompt},
            ]

            with st.spinner("Beginning interview..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=st.session_state.messages,
                    )
                    first_reply = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": first_reply})
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please ensure you have entered your name and a valid interview decision tree before starting.")

    #chatUI display            
    if st.session_state.get("messages"):
        #progres bar
        if st.session_state.get("interview_in_progress") and not st.session_state.get("interview_ended"):
            current_depth = len(st.session_state.stage_path)
            progress_percentage = current_depth / st.session_state.max_depth
            st.markdown("##### __***Interview Progress:***__")
            st.progress(progress_percentage)
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

        #user input and handling
        #1. user reply
        if prompt := st.chat_input("Type your reply..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

        #2.story stage analysis
            with st.spinner("Thinking..."):
                new_analysis = analyze_story_stages(st.session_state.messages, st.session_state.current_node)
                if new_analysis.get("adequate"):
                    steering_instruction = f"The user has answered {st.session_state.current_node['question']} adequately. Please proceed to the next question."
                    next_node, new_path = move_to_next_stage(st.session_state.decision_tree,st.session_state.stage_path)
                    if next_node != st.session_state.current_node:
                        st.session_state.current_node = next_node
                        st.session_state.stage_path = new_path
                    else:
                        steering_instruction = "Leaf node of interview decision tree has been reached, thank the interviewee and wrap up the interview"
                        st.session_state.interview_ended = True
                else:
                    # Probe deeper on same node
                    probe_instruction = new_analysis.get("reason", "Please ask the interviewee to elaborate further.")
                    steering_instruction = f"The user hasn't fully answered the current question, {st.session_state.current_node['question']} due to " + probe_instruction + " please rephrase the question in order to get a comprehensive answer."

            temp_messages = st.session_state.messages.copy()
            temp_messages.append({"role": "system", "content": steering_instruction})
            logging.debug(temp_messages)

            if not st.session_state.interview_ended:
                # 5. main interviewer generates the next question
                with st.spinner("Thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=temp_messages,
                        )
                        reply = response.choices[0].message.content
                    except Exception as e:
                        reply = "Sorry, there was an issue generating a response."
                        st.error(f"Error: {e}")

                # 6. Append interviewer message
                st.session_state.messages.append({"role": "assistant", "content": reply})

                # 7. Refresh UI
                st.rerun()

        if 'interview_ended' not in st.session_state:
            st.session_state.interview_ended = False

        if st.button("🛑 End Interview"):
            st.session_state.interview_ended = True

        if st.session_state.interview_ended:
            # Only do transcript/CSV generation once
            if not st.session_state.get("finalised", False):
                transcript_text = generate_transcript(st.session_state.messages)
                st.session_state['transcript'] = transcript_text
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Thank you {st.session_state['interviewee']}, that's all the questions we have for today. Thank you for taking the time to share your placement experience. This concludes our interview."
                })
                st.success("Interview ended. You can download your transcript below.")
                st.info("Please wait until analysis has finished before downloading data...")

                csv_output = generate_csv(st.session_state["csv_prompt"], transcript_text)
                if csv_output:
                    st.session_state["csv_output"] = csv_output
                    st.success("✅ CSV generated successfully!")
                else:
                    st.error("❌ Could not generate valid CSV after 5 retries.")

                st.session_state["finalised"] = True

                st.rerun()

            # After finalization, just render downloads
            if 'transcript' in st.session_state:
                st.download_button(
                    label="📥 Download Transcript",
                    data=st.session_state['transcript'],
                    file_name=f"{st.session_state['interviewee']}_interview_transcript.txt",
                    mime="text/plain"
                )

            if "csv_output" in st.session_state and st.session_state["csv_output"]:
                st.download_button(
                    label="💾 Download CSV",
                    data=st.session_state["csv_output"],
                    file_name=f"{st.session_state['interviewee']}_interview.csv",
                    mime="text/csv"
                )
with tab2:
    st.title("⚙️ Settings")
    st.text_area("Interviewer Prompt:", key="interview_prompt", height=350)
    st.text_area("Steering Prompt:", key="steering_prompt", height=150)
    st.text_area("Decision Tree Prompt:", key="choice_prompt", height=350)
    st.text_area("CSV Generation Prompt:", key="csv_prompt", height=250)
