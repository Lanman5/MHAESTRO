import streamlit as st
from openai import OpenAI
import io
import csv
import re
import json
import logging 
import streamlit.components.v1 as components
from html import escape
import pandas as pd
from io import StringIO
import smtplib
from email.message import EmailMessage

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
# Configure logging level and format

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

st.set_page_config(page_title="Knowledge Elicitator", page_icon="🧠", layout="centered")
st.title("Knowledge Elicitation")

MY_APP_PASSWORD = st.secrets.get("MY_APP_PASSWORD")
MY_APP_PASSWORD = "test"
MY_EMAIL = "alannaky6@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

#AI Model Configuration
client = OpenAI()

#functions
def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    
def send_email(body, attachment_content, attachment_filename):
    msg = EmailMessage()
    msg["From"] = MY_EMAIL
    msg["To"] = "A.Naky@lboro.ac.uk"
    msg["Subject"] = f"Testing Report for {st.session_state.get('interviewee', 'Unknown')}"
    msg.set_content(body)
    msg.add_attachment(
        attachment_content,
        maintype="text",
        subtype="plain",
        filename=attachment_filename
    )

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(MY_EMAIL, MY_APP_PASSWORD)
        smtp.send_message(msg)

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

#tree navigation functions
def calculate_max_depth(node):
    if "children" not in node or not node["children"]:
        return 1
    max_child_depth = 0
    for child in node["children"].values():
        max_child_depth = max(max_child_depth, calculate_max_depth(child))
    return 1 + max_child_depth

# Helper: robust recursive search for a node by its id anywhere in the tree
def find_node_by_id(node, target_id):
    if not isinstance(node, dict):
        return None
    if node.get("id") == target_id:
        return node
    children = node.get("children") or {}
    # children might be a dict of {key: node} or a list of nodes
    if isinstance(children, dict):
        iterator = children.values()
    else:
        iterator = children
    for child in iterator:
        found = find_node_by_id(child, target_id)
        if found:
            return found
    return None

def get_next_stage(decision_tree, path):
    """
    Traverse the tree by a path of node IDs.
    Path should normally start with the root id.
    Returns the node dict for the last id in path or {} if not found.
    """
    logging.debug(">>> get_next_stage called with path: %s", path)

    def find_node_by_id(node, target_id):
        if not isinstance(node, dict):
            return None
        if node.get("id") == target_id:
            return node
        children = node.get("children") or {}
        if isinstance(children, dict):
            for child in children.values():
                found = find_node_by_id(child, target_id)
                if found:
                    return found
        else:
            for child in children:
                found = find_node_by_id(child, target_id)
                if found:
                    return found
        return None

    if not path:
        logging.debug("Empty path provided")
        return {}

    root_id = decision_tree.get("id")
    if path[0] != root_id:
        logging.debug("Path does not start with root, searching globally")
        found = find_node_by_id(decision_tree, path[-1]) or {}
        logging.debug("Found node (global search): %s", found)
        return found

    node = decision_tree
    for node_id in path[1:]:
        children = node.get("children") or {}
        if isinstance(children, dict):
            node = next((c for c in children.values() if c.get("id") == node_id), None)
        else:
            node = next((c for c in children if c.get("id") == node_id), None)
        if not node:
            logging.warning("Node id %s not found under current node, doing global search", node_id)
            node = find_node_by_id(decision_tree, node_id)
            if not node:
                logging.error("Node id %s not found in entire tree", node_id)
                return {}
    logging.debug("get_next_stage returning node: %s", node)
    return node


def path_to_questions(decision_tree, path):
    """
    Given a path of node IDs (root first), return the list of question strings
    for each node along that path.
    """
    logging.debug(">>> path_to_questions called with path: %s", path)
    questions = []
    for i in range(len(path)):
        subpath = path[: i + 1]
        node = get_next_stage(decision_tree, subpath)
        logging.debug("node at subpath %s: %s", subpath, node)
        if node and "question" in node:
            questions.append(node["question"])
            logging.debug("questions so far: %s", questions)
    return questions


def move_to_next_stage(decision_tree, stage_path):
    """
    Progress the interview to the next stage based on the current node and GPT's choice (if needed).

    Args:
        decision_tree (dict): The full decision tree.
        stage_path (list[str]): The list of node IDs visited so far (root first).

    Returns:
        (next_node: dict, new_stage_path: list[str])
    """
    logging.debug(">>> move_to_next_stage called")
    logging.debug("decision_tree root id: %s", decision_tree.get("id"))
    logging.debug("initial stage_path: %s", stage_path)

    # Generate transcript of the interview so far
    transcript = generate_transcript(st.session_state.messages)
    logging.debug("transcript:\n%s", transcript)

    # Get the current node
    current_node = get_next_stage(decision_tree, stage_path)
    logging.debug("current_node: %s", current_node)

    children = current_node.get("children") or {}
    logging.debug("children raw: %s", children)

    # Handle both dict and list structures for children
    if isinstance(children, list):
        children_dict = {str(i): child for i, child in enumerate(children)}
    else:
        children_dict = children
    logging.debug("children_dict (normalized): %s", children_dict)

    # ----- Case 1: No children (leaf node) -----
    if not children_dict:
        logging.debug("Leaf node reached, no children — returning current_node")
        return current_node, stage_path

    # ----- Case 2: Only one child → auto-progress -----
    if len(children_dict) == 1:
        next_node = list(children_dict.values())[0]
        logging.debug("Only one child found: %s", next_node)
        if next_node and "id" in next_node:
            stage_path.append(next_node["id"])
            logging.debug("stage_path updated (auto-progress): %s", stage_path)
        return next_node, stage_path

    # ----- Case 3: Multiple children → ask GPT to choose -----
    options_text = "\n".join([
        f"- key: {key} | id: {child.get('id')} | question: {child.get('question', 'N/A')}"
        for key, child in children_dict.items()
    ])
    logging.debug("options_text built:\n%s", options_text)

    formatted_choice_prompt = st.session_state["choice_prompt"].format(
        tree_path=path_to_questions(st.session_state["decision_tree"], st.session_state["stage_path"]),
        transcript=transcript,
        options_text=options_text
    )
    logging.debug("formatted_choice_prompt:\n%s", formatted_choice_prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": formatted_choice_prompt},
                {"role": "user", "content": transcript}
            ]
        )

        result = response.choices[0].message.content
        logging.debug("GPT raw response:\n%s", result)

        parsed = json.loads(result)
        logging.debug("parsed GPT response: %s", parsed)

        next_key = parsed.get("next_key")
        logging.debug("next_key chosen: %s", next_key)

        next_node = children_dict.get(next_key)
        logging.debug("next_node via key: %s", next_node)

        if not next_node:
            next_id = parsed.get("next_id")
            logging.debug("next_node not found by key, trying next_id: %s", next_id)
            if next_id:
                for child in children_dict.values():
                    if child.get("id") == next_id:
                        next_node = child
                        logging.debug("next_node found via id match: %s", next_node)
                        break

        if not next_node:
            logging.warning("GPT choice not valid; falling back to first child")
            next_node = list(children_dict.values())[0]
            logging.debug("fallback next_node: %s", next_node)

        if next_node and "id" in next_node:
            stage_path.append(next_node["id"])
            logging.debug("stage_path updated (GPT choice): %s", stage_path)
        else:
            logging.error("Selected next_node has no 'id'; cannot safely update stage_path")

        return next_node, stage_path

    except Exception as e:
        logging.error("Error choosing next stage via GPT: %s", e)
        next_node = list(children_dict.values())[0]
        logging.debug("fallback next_node (exception): %s", next_node)
        if next_node and "id" in next_node:
            stage_path.append(next_node["id"])
            logging.debug("stage_path updated (exception fallback): %s", stage_path)
        return next_node, stage_path


#default configuration
if "config_initialized" not in st.session_state:

    st.session_state.update({
        "current_decision_tree": "nanaBanana.json",
        "interview_in_progress": False,
        "interview_ended": False,
        "stage_path": [],
        "decision_tree": {},
        "current_node": None,
        # Interviewer
        "interview_prompt": """You are a knowledge elicitation expert conducting an in-depth and structured interview with {name}, who is a player who has just played a new board game about teaching people about the etymology of names from different regions. 
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
• Don't ask for too many specific details, try and obtain generalised answers while still allowing for depth, detail and insight.
Your goal is to follow the interview framework to be able to elicit enough information that is of a high and insightful level in order to understand that students experiences on placements and how we can use their experiences to improve the experiences and prospects of future placement students. You can where appropriate use closed questions to be able to extract some details from the user quickly.
""",

        #assistant bots
        "steering_prompt": """You are an analysis expert evaluating interview answers given in the transcript so far.
    The interviewer is currently at the stage in the interview decision tree: {stage}.
    Determine if the interviewee's last response sufficiently answers this stage with a perceptive level of detail in order to be able to move on in the interview.
    Return ONLY JSON strictly in this format:
    {{"adequate": true/false, "reason": "short explanation that will tell the interviewer how to probe further to collect adequate information."}}

    However, keep track of the transcript, and if you sense that you find that the interviewer has repeated themselves or are have probed on the same node/question twice with no sufficient progress, instruct the interviewer to move onto the next question by returning adequate as true even if the question may not be fully answered. Guage on interviewee mood and adjust your approach accordingly.
    """,

        "choice_prompt": """You are an expert interview analyst, monitoring an ongoing interview transcript helping to navigate a structured interview decision tree.

    The interview transcript so far:
    {transcript}

    The path the interview has taken through the decision tree so far is {tree_path}.

    You must choose the most relevant next branch from these options that best suits the direction of the interview:
    {options_text}

    Return ONLY a JSON object like:
    {{"next_key": "<selected_child_key>", "reason": "<short reason dictating why you have chosen this path.>"}}. """,


        "evaluation_prompt": f"""You are an expert qualitative analyst. You will receive:

1. An interview transcript.
2. A list of five evaluation questions.

Your task:

- Carefully analyze the interview transcript in depth.
- Use only the interviewee’s own responses (no external assumptions).
- For each evaluation question, write a summary answer in no more than 3 sentences.
- Each answer must use specific details from the interviewee's responses whenever possible.
- Do not repeat the transcript verbatim; paraphrase concisely while keeping the meaning faithful.
- If there's no sufficient data for that question, write "N/A" in the answer field.

Output format (JSON):

{{
  "answers": [
    {{
      "question": "<evaluation question 1>",
      "summary_answer": "<3-sentence answer based only on the interviewee's responses>"
    }},
    {{
      "question": "<evaluation question 2>",
      "summary_answer": "<3-sentence answer based only on the interviewee's responses>"
    }}
    // ... one object per question
  ]
}}
""",
        "config_initialized": True,
        "evaluation_questions": ["In a social situation, have you ever wondered or wanted to find out the meaning of somebody’s name?", "If yes, what triggered you to ask that question", "Do you believe it's important to know the meaning of other people's names?", "Does an understanding of name etymology make you a more diverse thinker and EDI aware?", "Does name etymology knowledge increase your curiosity about different cultures and does this knowledge empower you within a social circle?"],
                "ui_questions": [
    "Was the program easy to use?",
    "Is the interface visually appealing?",
    "Were the questions coherent?",
    "Were the questions repetitive or intrusive at any point?",
    "Do you think this tool is just as effective as a human led interview?"
            ]
    })

#MAIN PROGRAM
tab1, tab2 = st.tabs(["🗨️Interview","⚙️ Configuration"])
with tab1:
    st.subheader("🎙️Interview:")
    uploaded_file = st.session_state["current_decision_tree"] 

    if uploaded_file:
        if isinstance(uploaded_file, str):  # default path
            content = read_file(uploaded_file)
        else:  # user uploaded file
            content = uploaded_file.read().decode("utf-8")
        st.session_state.decision_tree = json.loads(content)
        st.session_state.max_depth = calculate_max_depth(st.session_state.decision_tree)
    else:
        st.error("No decision tree found - please upload your interview framework")
        st.stop()
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
    #st.write(f"Current Node: {st.session_state.current_node.get('id', 'N/A') if st.session_state.get('current_node') else 'N/A'}")
    if st.session_state.get("interview_in_progress") and not st.session_state.get("interview_ended"):
        stage_path = st.session_state.get("stage_path", [])
        max_depth = max(1, st.session_state.get("max_depth", 1) - 1)

        # Depth = number of nodes visited (excluding root)
        current_depth = max(0, len(stage_path) - 1)

        # Calculate percentage
        progress_percentage = min(1.0, current_depth / max_depth)

        st.markdown(f"##### __***Interview Progress: {int(progress_percentage * 100)}%***__")
        st.progress(progress_percentage)


        # st.write(f"Decision Tree: {st.session_state.get('decision_tree', {})}")
        # st.write(f"Current Node: {st.session_state.get('current_node', {})}")
        # st.write(f"Stage Path: {st.session_state.get('stage_path', [])}")
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
                "content": f"Thank you {st.session_state['interviewee']}, that's all the questions we have for today. Thank you for taking the time to share your thoughts playing Namely. This concludes our interview."
            })
            st.success("Interview ended. Please proceed with the evaluation below .")

            default_eval_questions = [
                "In a social situation, have you ever wondered or wanted to find out the meaning of somebody's name?",
                "If yes, what triggered you to ask that question",
                "Do you believe it's important to know the meaning of other people's names?",
                "Does an understanding of name etymology make you a more diverse thinker and EDI aware?",
                "Does name etymology knowledge increase your curiosity about different cultures and does this knowledge empower you within a social circle?"
            ]
            evaluation_questions = st.session_state.get('evaluation_questions', default_eval_questions)

            with st.spinner("Summarising your answers..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": st.session_state["evaluation_prompt"]},
                            {"role": "user", "content": (
                                f"By following the exact framework specified above using the following transcript: "
                                f"{st.session_state.get('transcript', 'TRANSCRIPT UNAVAILABLE')}, "
                                f"and evaluation questions: {evaluation_questions}, "
                                "return only the JSON file as specified, nothing else."
                            )}
                        ]
                    )

                    try:
                        st.session_state['evaluation_json'] = json.loads(response.choices[0].message.content)
                    except json.JSONDecodeError:
                        st.error("⚠️ Model output wasn't valid JSON.")
                        st.stop()
                    st.session_state['finalised'] = True
                except Exception as e:
                    st.error(f"Evaluation failed - please try again. Error: {e}")

            if "likert_scores" not in st.session_state:
                st.session_state.likert_scores = {}

        # From here on, use the parsed JSON object from session state
        if st.session_state.get("finalised", False):
            evaluation_json = st.session_state['evaluation_json']  # parsed dict always available

            for idx, item in enumerate(evaluation_json["answers"]):
                cols = st.columns([3, 4, 2])  # adjust ratios as you like

                with cols[0]:
                    st.markdown(f"**Q{idx+1}:** {item['question']}")
                with cols[1]:
                    st.markdown(f"*Summary:* {item['summary_answer']}")
                with cols[2]:
                    score = st.slider(
                        label=f"Score for Q{idx+1}",
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        key=f"likert_{idx}"
                    )
                    st.session_state.likert_scores[idx] = score

            st.markdown("### 💻 User Interface Feedback")
            if "ui_likert_scores" not in st.session_state:
                st.session_state.ui_likert_scores = {}

            # Define Likert labels if you want text on the pills
            likert_labels = ["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]

            for i, q in enumerate(st.session_state.ui_questions):
                st.markdown(f"**Q{i+1}:** {q}")
                score = st.pills(
                    label="",  # no label, we already show the question
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: likert_labels[x-1],  # optional pretty labels
                    default=3  # default selection (3 = Neutral)
                )
                st.session_state.ui_likert_scores[i] = score


            if st.button("📑 Generate and Submit your testing report"):
                with st.spinner("Submitting your results - please do not leave this page"):
                    rows = []
                    
                    # Add evaluation answers
                    for idx, item in enumerate(evaluation_json["answers"]):
                        rows.append({
                            "Category": "Interview Evaluation",
                            "Question": item["question"],
                            "Summary Answer": item["summary_answer"],
                            "Likert Score": st.session_state.likert_scores.get(idx, "")
                        })

                    # Add UI feedback
                    for idx, q in enumerate(st.session_state.ui_questions):
                        rows.append({
                            "Category": "UI Feedback",
                            "Question": q,
                            "Summary Answer": "",
                            "Likert Score": st.session_state.ui_likert_scores.get(idx, "")
                        })

                    df = pd.DataFrame(rows)

                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue().encode("utf-8")

                    file_name = f"{st.session_state.get('interviewee', 'JohnDoe')}_testing_report.csv"

                    st.download_button(
                        label="💾 Download your testing report",
                        data=csv_bytes,
                        file_name=file_name,
                        mime="text/csv"
                    )

                    try:
                        send_email(
                            body=f"Please find attached the testing report generated for the participant: {st.session_state.get('interviewee', 'JohnDoe')}.",
                            attachment_content=csv_bytes,
                            attachment_filename=file_name
                        )
                        st.success("✅ File successfully submitted - thank you so much for taking the time to test our projects!")
                    except Exception as e:
                        st.error(f"⚠️ Could not send email: Please email your testing file manually. Error: {e}")



with tab2:
    st.title("⚙️ Settings")
    new_upload = st.file_uploader("Upload your JSON decision tree here:", type="json")
    if new_upload is not None:
        st.session_state["current_decision_tree"] = new_upload
        st.success("New decision tree loaded for this session.")
    st.text_area("Interviewer Prompt:", key="interview_prompt", height=350)
    st.text_area("Steering Prompt:", key="steering_prompt", height=150)
    st.text_area("Decision Tree Prompt:", key="choice_prompt", height=350)
    st.text_area("Evaluation Prompt:", key="evaluation_prompt", height=250)

    with st.expander("Edit Evaluation Questions"):
        updated = []

        for i, q in enumerate(st.session_state.evaluation_questions):
            cols = st.columns([8, 2])
            new_q = cols[0].text_input(f"Question {i+1}", value=q, key=f"q_{i}")
            if not cols[1].button("Remove", key=f"remove_{i}"):
                updated.append(new_q)

        if st.button("Add New Question"):
            updated.append("New question text here...")

        if updated != st.session_state.evaluation_questions:
            st.session_state.evaluation_questions = updated
            st.rerun() 

        st.write("Current Questions in Session State:", st.session_state.evaluation_questions)

    
    with st.expander("Edit UI Questions"):
        updated_ui = []
        for i, q in enumerate(st.session_state.ui_questions):
            cols = st.columns([8, 2])
            new_q = cols[0].text_input(f"UI Question {i+1}", value=q, key=f"ui_q_{i}")
            if not cols[1].button("Remove", key=f"remove_ui_{i}"):
                updated_ui.append(new_q)

        if st.button("Add New UI Question"):
            updated_ui.append("New UI question text here...")

        if updated_ui != st.session_state.ui_questions:
            st.session_state.ui_questions = updated_ui
            st.rerun()

        st.write("Current UI Questions in Session State:", st.session_state.ui_questions)
