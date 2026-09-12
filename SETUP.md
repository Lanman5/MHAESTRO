# Setup and testing

Everything needed to get both apps running locally, test them properly, and
deploy them for the open day.

---

## 1. Local setup

### Python

**3.10 or newer is required.** Your default `python` is 3.9.12, and on 3.9 pip
silently resolves `anthropic` to the old 0.x SDK, which has no `output_config` —
the Anthropic structured-output path would then fail at run time rather than at
install time. `requirements.txt` pins `anthropic>=1.0` so that fails loudly
instead.

A virtual environment is already built for you at `.venv`, using your Anaconda
Python 3.12.4:

```
python 3.12.4 · streamlit 1.63.0 · openai 3.13.0 · anthropic 1.5.0 · google-genai
```

If you ever need to rebuild it:

```bash
"C:/Users/alann/anaconda3/python.exe" -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then open `.streamlit/secrets.toml` and fill it in. It is gitignored. You need:

| Key | Needed for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | one provider minimum | GPT-4-class is the published MHAESTRO configuration, so it is the default |
| `ANTHROPIC_API_KEY` | optional | enables the Claude models in the picker |
| `GEMINI_API_KEY` | optional | enables the Gemini models in the picker |
| `REPORT_FROM_EMAIL` | delivery | the Gmail account that sends session records |
| `REPORT_EMAIL_PASSWORD` | delivery | a Gmail **App Password**, not your account password |
| `REPORT_TO_EMAIL` | delivery | defaults to `A.Naky@lboro.ac.uk` |
| `RESEARCHER_PIN` | **required before the event** | gates the Elicitor's study controls |
| `EXPERT_PASSCODE` | **required before the event** | gates the whole K-Eng app |

The apps show only the providers whose key *and* SDK are actually present, so a
single-provider setup is fine.

### The Gmail App Password

A normal Google password will be rejected by SMTP.

1. Google Account → **Security** → turn on **2-Step Verification** (required).
2. Go to <https://myaccount.google.com/apppasswords>.
3. Create one named e.g. "MHAESTRO", copy the 16 characters.
4. Paste into `REPORT_EMAIL_PASSWORD`. Spaces are fine.

### Set a spend limit

Both apps are public once deployed and every turn costs money. Set a hard
monthly cap on each provider's billing page before the event.

---

## 2. Running locally

```bash
.venv/Scripts/python.exe -m streamlit run knowledge-elicitation/app.py
```

```bash
.venv/Scripts/python.exe -m streamlit run knowledge-engineer/app.py --server.port 8502
```

Run them on different ports if you want both open at once. There is also
`.claude/launch.json` with both configured, if you drive them from the IDE.

---

## 3. The automated tests

Run these after any change. No network, no framework, no fixtures.

```bash
.venv/Scripts/python.exe tests/test_core.py
```

```bash
.venv/Scripts/python.exe tests/test_screens.py
```

```bash
.venv/Scripts/python.exe tests/test_arms.py
```

- **test_core** — schema validation, the coverage guarantee, fault detection,
  JSON extraction, block randomisation, telemetry aggregation, prompt rendering.
- **test_screens** — drives both apps headlessly and asserts no screen raises:
  the information sheet, both age routes, the consent gate, the access gates, the
  expert's graph review, the outcome battery end to end.
- **test_arms** — complete sessions through all four arms against a scripted fake
  provider: the adequacy gate, the re-probe loop, the soft cap, labelled-edge
  traversal, traversal-error recovery, the skip control, coverage accounting.

All three pass. They use dummy API keys and make no real model call.

---

## 4. Manual testing checklist

The automated tests cannot check that the *model* behaves well — no real call is
made anywhere in them. Work through this with real keys before the event.

### K-Eng (the expert tool)

- [ ] Passcode gate refuses a wrong code, accepts the right one.
- [ ] Fill in the event and respondents, start the interview.
- [ ] Have a genuine 8–10 minute conversation as if you were the Open Day lead.
      Check the questions actually build on your answers rather than reading a list.
- [ ] Click **I've said enough — build the survey**.
- [ ] The graph compiles and passes validation. If it needed a repair pass, the
      session metadata records `compile_repairs_needed` — worth knowing how often.
- [ ] The **Flow** tab draws the graph legibly. Blue = every visitor is asked;
      grey = conditional; green = ends here.
- [ ] No "these must-ask questions can be skipped" warning. If there is one, the
      coverage guarantee is broken — tell it so in the verdict box and rebuild.
- [ ] The estimated longest run fits the time budget.
- [ ] Fill in the verdict, submit, confirm the email arrives with five attachments
      (four session files plus the policy JSON).
- [ ] Download the policy JSON and drop it into `knowledge-elicitation/trees/`.

### Elicitor (the participant tool)

Do this on a phone, not a laptop — that is what visitors will use.

- [ ] Information sheet reads correctly and the minute counts are right.
- [ ] **Under 18** route: warning about the external AI provider appears, the tool
      is usable, the practice banner stays visible, and **no email arrives**.
- [ ] **Over 18** route: consent form matches the approved wording, **Begin** is
      disabled until the box is ticked, "I do not wish to take part" ends cleanly.
- [ ] Enter the researcher PIN in the sidebar, then use **Force arm** to walk all
      four arms in turn:
      - **A** graph + adequacy — should re-probe at most once per question
      - **B** graph, no check — should never re-probe
      - **C** free-form + adequacy — no fixed order, but still gated
      - **D** free-form, no check — a plain interviewer
- [ ] Time each arm with a stopwatch. This is the number that decides whether the
      budget works.
- [ ] Try answering "I don't know" — it should accept it and move on, not press.
- [ ] Press **Skip this one** and **Why are you asking?**; both should work and
      both are logged.
- [ ] The summaries at the end genuinely reflect what you said.
- [ ] Submit; confirm the email arrives with four attachments.
- [ ] Deliberately break the email (wrong password) and confirm the failure is
      shown on screen with working download buttons.

### The data

- [ ] Save the emailed `*_session.csv` files into one folder and run:

```bash
.venv/Scripts/python.exe analysis/pool_sessions.py path/to/folder -o pooled.csv
```

- [ ] Check `meta_arm_id` is populated and the four arms appear.
- [ ] Check `fidelity_fluidity_gap`, `ux_non_intrusiveness_aligned` and
      `tlx_raw_mean` have values.
- [ ] Open an `*_events.csv` and confirm every row carries a node id, a timestamp,
      a model and a prompt-template hash.

---

## 5. Deploying to Streamlit Community Cloud

### Push the code

Nothing is committed yet. Your remote is already
`https://github.com/Lanman5/MHAESTRO.git`.

```bash
git add -A && git commit -m "MHAESTRO 2.0: validated policy graphs, four-arm design, full audit log"
```

```bash
git push origin main
```

`.gitignore` keeps `secrets.toml`, `.venv/` and any collected participant data out
of the repository. Check `git status` before pushing if you have already collected
anything.

### Deploy the Elicitor

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `Lanman5/MHAESTRO`, branch `main`,
   main file path **`knowledge-elicitation/app.py`**.
4. **Advanced settings** → Python version **3.12** (or 3.13 — *not* 3.9).
5. Paste your whole `secrets.toml` contents into the **Secrets** box.
6. Deploy. Note the URL — this is what the QR code points at.

### Deploy K-Eng

Same again, as a **second app from the same repository**, with main file path
**`knowledge-engineer/app.py`**. Same secrets. It needs `EXPERT_PASSCODE` set,
because its URL is public and every screen past the setup form spends credits.

### The QR code

Generate it from the Elicitor's URL. Test by scanning it on a phone that is not
signed into anything, on mobile data rather than campus wifi.

---

## 6. On the day

- [ ] Open the Elicitor once yourself first — Community Cloud sleeps idle apps and
      the first load takes ~30 seconds. Wake it before visitors arrive.
- [ ] Enter the researcher PIN and confirm **Force arm** is on **Randomise**.
- [ ] Check the arm counter in the sidebar every so often; it should stay roughly
      even. If the app restarts, the block randomiser restarts too — that is a
      re-seed, not a bias, and the allocation method is recorded per session.
- [ ] Watch your inbox. One email per completed participant. A gap means a
      delivery failure — the visitor's screen will have offered downloads.
- [ ] Keep a laptop handy so a failed submission can be rescued by download.

### Known limits

- **The app sleeps.** Community Cloud suspends apps after inactivity. Keep a tab
  open on a phone or laptop at the stand.
- **One app instance shares one randomiser.** If Community Cloud restarts the app
  mid-event, allocation restarts from a fresh block.
- **Email is the only durable sink.** Community Cloud has no persistent disk. If
  an email fails and nobody downloads the file, that session is gone.

---

## 7. Tuning before you commit to the design

| Want | Change |
|---|---|
| A shorter session | `target_minutes` in `knowledge-elicitation/trees/cs_open_day.json`. At 3 both optional probes get dropped under time pressure. |
| A harsher arm A, matching the published run | `max_reprobes: 2` on every node in the policy JSON. |
| A shorter/longer feedback battery | `mhaestro/feedback.py`: flip `include=` on any `EXPERIENCE_ITEMS` or `OPEN_ITEMS` entry; set `DEFAULT_TLX_KEYS = RTLX_KEYS` to restore the full six-subscale TLX. Cut items keep their CSV columns, so the table's shape never changes. |
| A gentler adequacy bar on a given question | `adequacy.require_reason: false` (or `require_stance: false`) on that node in the policy JSON. Softens the bar without removing the gate, so the arm contrast survives. |
| Different models per role | Elicitor: researcher sidebar. K-Eng: settings sidebar. Defaults come from `DEFAULT_PROVIDER`, `INTERVIEWER_MODEL` and `CONTROL_MODEL` in secrets. |
| Speech input for participants | `RESEARCHER_PIN` sidebar → **Offer speech input**. Needs an OpenAI key. Off by default because it adds latency. |
| A different survey | Run K-Eng, download the policy, drop it in `knowledge-elicitation/trees/`. |

> **The five-minute problem.** The battery is now 16 rated items and one free-text
> box, down from 20 and four. That is roughly 2 minutes rather than 3–4. Six core
> questions on top still put the whole thing near 5–6 minutes, so the information
> sheet promises 5 minutes of conversation plus 2 of feedback. Time the arms with a
> stopwatch in manual testing before deciding whether to cut further — the levers
> are all in the table above.
