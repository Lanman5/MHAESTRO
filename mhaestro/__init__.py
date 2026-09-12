"""MHAESTRO: a hybrid, auditable conversational-surveying system.

Shared library behind the two Streamlit applications:

    knowledge-engineer/app.py   Phase 1, K-Eng   -- compiles an expert's
                                requirements into a validated decision graph.
    knowledge-elicitation/app.py Phase 2, Elicitor -- runs the governed survey with
                                participants and collects the outcome battery.

Module map:

    config      secrets and environment resolution
    llm         provider-agnostic chat completion (OpenAI, Anthropic, Gemini)
    prompts     versioned, content-hashed prompt templates
    schema      the policy-graph artefact: normalisation, validation, hashing
    agents      Agents [a]-[d], the summariser, coverage coder and analysis agent
    arms        the 2x2 experimental design and its block randomiser
    telemetry   append-only event log and the per-session record
    consent     participant information, age gate and informed consent
    feedback    extraction fidelity, experience and workload instruments
    delivery    getting a finished session out of the browser
    ui          shared presentation layer
"""

__version__ = "2.0.0"
