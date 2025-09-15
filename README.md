# __Spirit Engine 2.0__

Spirit Engine 2.0 is an **AI-powered interactive tool** that allows users to conduct interviews, analyse them, and generate narrative and story outputs for both adults and young children (ages 3–5).  

It combines **conversational AI, text-to-speech (TTS), and automated storytelling**, providing a seamless experience for collecting, analyzing, and creating stories from interviews.

---

## __Features__

- Conduct interactive interviews with AI as your interviewer with the option of typing or speaking to the interviewer.  
- Generate transcripts automatically.  
- Analyse interviews for key insights in line with the Spirit Engine.  
- Generate narratives, adult stories, and children's stories (ages 3–5).  
- Built-in **Text-to-Speech (TTS)** to automatically read interview questions and stories.  
- Automatically produces a **testing report** that can be downloaded or emailed.  
- Fully configurable via **CSV prompt files** and **Streamlit settings**.  

---

## __📖 Table of Contents__

1. [Installation](#-installation)  
2. [Setup and API Keys](#-setup-and-api-keys)  
3. [Running Locally](#-running-locally)  
4. [Using Github and Hosting on Streamlit Community Cloud](#-using-github-and-hosting-on-streamlit-community-cloud)  
4. [Github Repository Contents](#repository-contents)
5. [Using Spirit Engine 2.0](#-using-spirit-engine-20)  
   - [Conducting an Interview](#conducting-an-interview)  
   - [Generating Narratives and Stories](#generating-narratives-and-stories)  
   - [TTS Playback](#tts-playback)  
   - [Downloading and Sending Reports](#downloading-and-sending-reports)  
7. [Updating Prompts and Pre-Requesite Files](#updating-csv-prompts)  
6. [Dependencies](#-dependencies)  
7. [Maintainence and Relevant Changes](#-maintaining-the-app)  
8. [Troubleshooting](#-troubleshooting)  
9. [Tutorial Example](#-tutorial-example)  

---

## __💻 Installation__
You should only need to do these once.
1. Save your OpenAI and Elevenlabs API keys as an environmental variable on your computer (this means you can test using these keys when running your program locally rather than online) by executing the following instructions in Powershell. 

```
setx OPENAI_API_KEY "your_api_key_here"
setx ELEVENLABS_API_KEY "your_api_key_here"
```
Should these change at any time just re-run the relevant command in powershell.

If they are open already you need to shut and reopen Visual Studio Code and Powershell in order for these changes to take effect. 

2. Install Python if not done so already (at time of writing 15/09/2025 use python 3.13) from this link [here](https://www.python.org/downloads/). 
3. Download and open Visual Studio Code (the blue one).
4. Link your github account to Visual Studio Code and select clone a github repository in the main menu and open this one to access all program files. 
5. Install the following dependencies (imports seen at the top of the code) by running the following in the terminal:
```
pip install streamlit openai elevenlabs python-dotenv pandas
```

---

## __▶ Running Locally__

Start a new terminal (either accessed in the top ribbon or in the bottom box)

If the correct folder isn't already there by default navigate to the correct folder where your code is located by the following command, cd <directory_path> e.g.:
```
cd C:\Users\alann\OneDrive - Loughborough University\Internship
```
Start the app with Streamlit by running the following code in your terminal:

```
streamlit run app.py
```
The app will open in your default browser.  
Navigate through the tabs: **Interview, Storytelling, and Settings** as normal.

---

## __☁ Using Github and Hosting on Streamlit Community Cloud__

1. Push your repository to GitHub (this is also required when you've made a change to the program and it needs to be both saved to your github account and pushed to the online version). 

This can be done by first saving any changes you've made to the file you're working on by pressing `ctrl + s` and then acessing the Source Control menu from the third icon down on the leftmost ribbon on your screen. First you must stage any changes (press the little plus on any files with an M), committing (adding a message in the top text box and pressing commit) to get it ready to upload and then pushing to the cloud (pressing sync changes). 

The terminal commands to do this if you prefer are:
```
git add -A
git commit -m "your message here to say what this change does"
git push
```

You should only need to do this if you've made any changes. If not there should be no changes displayed for you to stage and commit. 

You can look back at your commit history to revert to any previous versions here too (this is why meaningful commit messages can be very useful)

2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and log in.  
3. Click **"New app" → Select your repository and branch**.  
4. Set environment variables in **Secrets** for OpenAI, ElevenLabs, and any other relevant credentials (e.g. email passwords for testing or access passwords) this can be accessed in settings.  

The secrets tab should look something like this:
```
OPENAI_API_KEY = "insert_key_here"
AUTHORIZED_PASSWORDS = ["password1", "password2"]
ELEVENLABS_API_KEY="insert_key_here"
```
5. In **settings > sharing** set this app is public and searchable to allow anyone with the link to use the app.
5. Click **Deploy**. Your app will be live and accessible online.  

---

## **Repository Contents**
The github repository contains:
- app.py (the main code)
- chat_components.html (used to load the chat interface)
- default_prompts.csv (contains all default prompts and references to default pre-requesite files)
- any pre-requesite files you wish to be used on default
- README.md (this file)
-.devcontainer (ignore this -> something to do with streamlit I think...)

These should all be on the same level and not contained within any subfolders. 

## __🛠 Using Spirit Engine 2.0__

### __Conducting an Interview__

- Go to the **Interview** tab.  
- Enter your name and enable **Auto-speak** if you want the AI to read questions aloud.  
- Click **Begin Interview**.  
- Respond either by typing or speaking (audio input is transcribed automatically).  
- The AI will guide the interview, track story stages, and flag safeguarding concerns.  

---

### __Generating Narratives and Stories__

- After ending an interview, go to **Storytelling** tab.  
- Click **Generate Narrative** to produce a first-person narrative.  
- Select **Adult Story** or **Children’s Story (ages 3–5)**.  
- Click **Generate Story**.  
- For children’s stories, the AI retries up to 3 times to ensure quality.  

---

### __TTS Playback__

- Voices can be selected in **Settings → Text-to-Speech**.  
- Stories and interviewer questions can be read aloud.  
- Use the **Mute** button to pause playback so you can speak to the interviewer wihtout being spoken over.  

---

### __Downloading Reports__

- Download transcripts, analysis, and testing reports with one click.  

---

### __Updating CSV Prompts__

Spirit Engine 2.0 uses **`default_prompts.csv`** for system prompts and default pre-requesite file configuration.

To add or edit prompts:  

1. Open **default_prompts.csv** in visual studio code.  
2. Add, edit, or delete prompts.  

- Column 1: Prompt/File name/title  
- Column 2: Prompt content / File Name

It needs to follow the following format `name;"content"`
You must make sure that semi colons are used as the separator and there is maximum one semicolon per line. 
Normally CSV files work by having one entry per line. The use of speech marks allows the prompts to span over multiple lines. 

**IF YOU'VE ONLY EDITED THE CSV FILE:**
- Ensure you've followed the correct format outlined above by ensuring you've used a semi colon to seperate and enclosed any multiline prompts in speech marks. 
- You shouldn't use any speech marks or semi-colons in the prompt just to be safe. If you really need to, search up the relevant escape characters to ensure it works with this CSV format. 

**IF YOU'VE ADDED OR REMOVED ANY ENTRIES:**

Find this line: `if prompt_list and len(prompt_list) < 17:` and change the number to the number of entries in the CSV document (by counging how many titles you have left of a semi-colon). __THE PROGRAM WILL NOT RUN IF YOU HAVEN'T DONE THIS CORRECTLY__

Update the session state in **`app.py`** if changing order/number of prompts by finding this code and making the following changes:

**If you've added a new prompt:**
- On the left of the colon contains how the program will reference and access this entry. 
- This is index notation 
    - Index notation always starts from 0
    - The [number] on the left represents the entry in the CSV file so 2 will mean the third entry as we start from 0
    - The [number] on the right represents the column of the CSV entry. We use one as it's the second entry (the content)

```
"analysis_system_prompt": prompt_list[2][1],
```

**If you've added a new file:**

Around line 410 find the similar sequences of code in the session state update and add the relevant entry for the new pre-requesite file.

```
init_file_data.append({"title": prompt_list[16][0], "content": read_file(prompt_list[16][1])})
```

You then also need to reference the relevant files into the correct section to set it as the default for that set of pre-requesite files.:
```
 "adult_story_files": [prompt_list[10][0],prompt_list[9][0],prompt_list[8][0]]
```

---

## __📦 Dependencies__

- `streamlit` – main web interface  
- `openai` – AI chat completions  
- `elevenlabs` – TTS audio  
- `python-dotenv` – environment variables  
- `pandas, csv, re, hashlib, logging` – internal usage  
- Standard Python libraries: `os, datetime, io, email, smtplib, base64`  

---

## __⚙ Maintaining the App and changes you may wish to make__ 
- **Making and Saving Changes** - to any part of the program whether it be code or not:
    - If hosting locally just press `ctrl + s` (there should be no bullet point next to your file anymore) and then follow [these Instructions](#-running-locally)  
    - If hosting online and/or you wish to save this changes to github as well as your local version follow [these Instructions](#-using-github-and-hosting-on-streamlit-community-cloud)  
- **Adding story stages:** Update `story_stages` in `st.session_state`. Ensure you update any relevant prompts for this (such as analysis)
- **Troubleshooting:** Check Streamlit logs & API keys validity. 
- **Adding or ammending Prompts and Files**: See [above section](#updating-csv-prompts) 
- **Password Protect the app**:
    - Locate this line of code: `check_password()` (around line 45)
        - Comment it out with a leading hashtag to leave the code unlocked (so it will go green and look like this) `#checkPassword()`
        - Leave it uncommented to lock the app so one of the AUTHORIZED_PASSWORDS are required to enter the program and it will look like this `check_password()`

 - **Removing Adult Stories**: Comment out any references to the adult stories interface (betweeen line 820 and 1000). Block commenting can be done by pressing `ctrl + /` (again it will all go green if it's commented out)
---

## __🐞 Troubleshooting__

- **No models available:**, **API ERROR** Verify OpenAI & ElevenLabs API keys and ensure there is enough credit.  
- **CSV issues:** Ensure `default_prompts.csv` exists with ≥ 17 entries.  
- **Audio not playing:** Confirm browser supports audio & valid voice is selected.  
- **Story generation fails:** Check prompts & update session state if CSV changed.  

---

## __🎯 Simpe Usage Tutorial __

1. Start the app. Enter **name**, enable **auto-speak**.  
2. Begin interview → respond to AI questions.  
3. End interview → download transcript.  
4. Generate narrative → view & listen to it.  
5. Generate children’s story → AI retries if quality is low and doesn't conform to storytelling criteria. .  
6. Download and email testing reports.  

---
## **Finally**
Thank you so much for the opporunity to let me work on this project, it's been amazing! Should you have any questions I'm always one (student) email away...