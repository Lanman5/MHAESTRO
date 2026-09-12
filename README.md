# MHAESTRO

A hybrid, auditable conversational-surveying system, and the instrument for the
follow-up study to *Hybrid Multi-Agent Systems for Auditable AI Surveying*
(Naky, Saravi, Batmaz, Yang, Derakhshan, Nevisi & Storey, 2025).

Two Streamlit applications share one library:

| App | Who uses it | What it does |
|---|---|---|
| `knowledge-engineer/app.py` | The academic running the event | Interviews them about what they need to find out, then compiles it into a **validated decision graph**, and shows it to them for review. |
| `knowledge-elicitation/app.py` | Open day visitors, via a QR code | Runs the governed conversational survey, randomised between four arms, and collects the outcome battery. |

---

## What this revision is for

The published study (N=8) found the architecture works technically and costs the
participant dearly: extraction fidelity M=4.20, but aligned non-intrusiveness
M=1.75 and perceived human-equivalence M=2.75 — a **fidelity–fluidity gap of
2.45 Likert points**. Its own stated next step (§6.5, §6.7, §7) is an A/B
comparison of the governed flow against an agency-enhanced variant, holding
coverage as a fixed constraint.

This build generalises that to a **2×2 between-participants factorial** crossing
the two things MHAESTRO actually contributes, so the paper's confound is broken:

|  | adequacy check | no check |
|---|---|---|
| **decision graph** | **A** — MHAESTRO as published | **B** — single pass, never re-probes |
| **free-form** | **C** — checker over an unstructured interviewer | **D** — the naive LLM interviewer |

The published run cannot say whether its fidelity came from the graph, from the
checker, or from both — nor which produced the gap. Crossing them answers that.
Arm assignment uses permuted blocks of four shared across everyone hitting the
same app instance, so the arms stay balanced as visitors arrive.

### What changed, and why

| Change | Why |
|---|---|
| **Flat node map with labelled edges** replaces nested `children` dicts | §3.2.4 specifies `condition`/`target_id` transitions and permits joins (in-degree > 1). A nested tree cannot express a join, and the old traversal code guessed between dict- and list-shaped children. |
| **Generate → validate → repair** loop in K-Eng | The contract is checkable (targets exist, labels unique, acyclic, terminating, core nodes on every path, fits the time budget). The prototype discovered violations when a live session hit them. |
| **Soft cap on re-probes**, visible **skip** control, **"why are you asking?"** hint | §6.1's own proposed remedies for the intrusiveness finding. Present in every arm. |
| **Agent [b] rewritten** | It now treats "I don't know" / "let's move on" as adequate and is told to prefer advancing when in doubt. The published prompt's instruction to *push* the interviewee is what produced the 4.25 repetitiveness rating. |
| **Full per-turn log** — node id, timestamp, agent role, provider/model, prompt-template id **and hash**, latency, tokens, rationale | §4.5 and §6.2 name the missing log schema as the limitation that let Results verify coverage but not sequence. |
| **Post-hoc coverage coding on every arm** | Coverage is structurally guaranteed in the graph arms and not in the free-form arms. Coding all four the same way is what makes coverage usable as the fixed constraint. |
| **Repetitiveness and intrusiveness asked separately** | They were one item, and the four UX facets did not form a scale (α=0.54). The 2.45-point gap rests entirely on that single item. |
| **Battery trimmed to 16 rated items + 1 free-text box** | Burden is the thing under study; a four-minute questionnaire after a five-minute survey would manufacture the fatigue being measured. All four of the paper's baseline UX facets are kept. `tlx_variant` records that a reduced TLX was administered, so a write-up cannot claim RTLX by accident. |
| **NASA-TLX added** | §6.3 argues the interaction strain is an accessibility and equity concern for people with high cognitive fatigue. Perceived workload is the standard instrument for that claim. |
| **Native chat UI** replaces the fixed-height HTML iframe | The old chat was a 400px `components.html` block: no resize, no theme, unreachable by a screen reader. §6.3 treats accessibility as part of validity. |
| **Any model id, negotiated not assumed** | Providers disagree about request parameters and change between generations (reasoning models renamed `max_tokens`; Haiku 4.5 rejects `effort`; older Claude predates structured outputs). A rejected parameter is dropped and the call retried, the finding cached, and the drop recorded in `llm_degraded` -- so a model released after this was written still runs, and a silently weaker request never stays silent. |
| **Multi-provider** (Anthropic, OpenAI, Gemini), per-role | Model family is a confound the published study could not examine. Now it is a recorded variable. Defaults to Claude Haiku 4.5: latency is itself a study variable, since the session has a five-minute budget and time pressure is measured. `meta_models` records provider and model per role, so the departure from the published GPT-4 run is explicit in the data. |

> **Arm A is not the published experience verbatim.** It is the published
> architecture with the paper's own recommended fixes applied — the re-probe cap
> in particular. `meta_agency_controls` and each node's `max_reprobes` record the
> configuration, so the difference from the unbounded published run is explicit.
> Set `max_reprobes` to 2 throughout if you want the harsher baseline back.

---

## Running it

**[SETUP.md](SETUP.md) has the full setup, testing checklist and deployment steps.**
The short version:

```bash
pip install -r requirements.txt
```

```bash
streamlit run knowledge-engineer/app.py
```

```bash
streamlit run knowledge-elicitation/app.py
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it
in, or paste the same keys into each app's **Secrets** box on Streamlit Community
Cloud. At least one provider key is required; the apps offer only the providers
whose key and SDK are actually present.

**Set `RESEARCHER_PIN` and `EXPERT_PASSCODE` before the event.** Both gates fail
closed. Without `RESEARCHER_PIN` the Elicitor's study controls — policy selection,
model choice, arm forcing — are hidden entirely rather than left open to whoever
is holding the phone. Without `EXPERT_PASSCODE` the K-Eng app is ungated, and on
Community Cloud its URL is public and every screen past the setup form spends API
credits.

### Deploying for an open day

1. Deploy `knowledge-elicitation/app.py` to Community Cloud; point the QR code at it.
2. Set the secrets, including `RESEARCHER_PIN` and the email credentials.
3. Check the arm counter in the researcher sidebar between visitors.
4. Save every emailed `*_session.csv` into one folder and run the pooling script.

Community Cloud has **no durable filesystem**, so each completed session is
emailed out immediately. Delivery retries three times, and a failure is shown on
screen with download buttons rather than swallowed — a visibly failed participant
is recoverable, a silently lost one is not.

---

## Data collected

Four artefacts per session, emailed on submit:

| File | Shape | Use |
|---|---|---|
| `*_session.csv` | one row | the analysis table — concatenate across participants |
| `*_turns.csv` | one row per participant turn | per-turn measures |
| `*_events.csv` | one row per event | sequence and provenance audit trail |
| `*_bundle.json` | everything | lossless archival copy, including the transcript |

Session-level columns include the arm and its two factors, the policy hash, the
prompt-bundle hash, the model used per role, structural and coded coverage, the
path taken, duration, turn counts, re-probe rate, soft-cap hits, skips, why-hint
requests, token totals, reply-length mean and **slope** (a disengagement proxy),
agent latencies, the full outcome battery, and `fidelity_fluidity_gap` computed
per session.

```bash
python analysis/pool_sessions.py path/to/emailed/csvs -o pooled.csv
```

prints per-arm and marginal means for the primary outcomes against the published
baseline, plus the coverage check.

---

## Ethics

`mhaestro/consent.py` reproduces the approved participant information sheet and
consent form verbatim; only the two bracketed placeholders the approved text
marks for substitution are filled. Three outcomes:

- **Consented** — 18 or over, information read, box ticked. Full collection.
- **Practice** — under 18. The tool works so nobody is turned away at a public
  stand, but **nothing is recorded or transmitted by us**, and the visitor is told
  plainly that their words still reach a third-party AI provider, because that is
  true regardless of what we keep. No battery, no submission.
- **Declined** — nothing runs, nothing is recorded.

The age gate comes *before* the consent form, so an under-18 visitor is never
shown a form they cannot validly give. `ConsentRecord.may_collect_data` is the
single gate every write passes through, and `delivery.deliver()` checks it again
before touching the network.

Participants are identified only by a generated pseudonym (`P-XXXXXXXXXX`). The
app never asks for a name — the published version's "Enter your Name" field is
gone, since the information sheet promises no identifiable information is
collected.

---

## Layout

```
mhaestro/          shared library
  config.py        secrets and environment resolution
  llm.py           provider-agnostic completion (OpenAI, Anthropic, Gemini)
  prompts.py       versioned, content-hashed prompt templates
  schema.py        the policy artefact: normalisation, validation, hashing
  agents.py        Agents [a]-[d], summariser, coverage coder, analysis agent
  arms.py          the 2x2 design and its block randomiser
  telemetry.py     append-only event log and the session record
  consent.py       information sheet, age gate, informed consent
  feedback.py      fidelity, experience and workload instruments
  delivery.py      email transport with retries and a manual fallback
  viz.py           policy graph rendering for expert review
  speech.py        optional speech input
  ui.py            shared presentation layer
knowledge-engineer/app.py
knowledge-elicitation/app.py
knowledge-elicitation/trees/cs_open_day.json
analysis/pool_sessions.py
```

`knowledgeElicitator.py`, `knowledgeEngineer.py`, `engineering_prompts.csv` and
`nanaBanana.json` are the published prototype, kept for provenance. Nothing
imports them. `nanaBanana.json` still loads: `schema.normalise_policy` up-converts
the legacy nested format, so the published tree can be re-run against the new
engine.

---

## The policy artefact

```json
{
  "survey_id": "cs_open_day",
  "root_id": "ask_activities",
  "target_minutes": 5,
  "summary_questions": ["..."],
  "nodes": [
    {
      "id": "ask_overall",
      "question": "How was the day overall for you?",
      "objective": "Gives an overall evaluation and some indication of what shaped it.",
      "priority": "core",
      "topic": "Overall experience",
      "max_reprobes": 1,
      "children": [
        {"condition": "clear_view_with_reason", "target_id": "ask_liked_most", "description": "..."}
      ]
    }
  ]
}
```

**Blocking faults**, checked before any session may start: a `target_id` that does
not resolve, `condition` labels duplicated within a node, a cycle, no reachable
terminal, a node with no question text.

**Warnings**, surfaced to the expert but never used to halt a live event: an
unreachable node, a node with no objective for the checker to test against, and —
most consequentially — a `core` node that *some* routes bypass, which means
coverage is not actually guaranteed for every visitor.

`priority: "optional"` marks depth probes that may be dropped when a session runs
over its time budget. Core topics never are.
