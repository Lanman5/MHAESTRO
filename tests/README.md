# Tests

No test framework, no fixtures, no network. Run each directly.

```bash
python tests/test_core.py
```

Schema validation, the coverage guarantee, fault detection, JSON extraction,
block randomisation, telemetry aggregation and prompt rendering. Imports nothing
that needs Streamlit, so it runs anywhere.

```bash
python tests/test_screens.py
```

Drives both Streamlit apps headlessly via `AppTest` and asserts no screen raises:
the information sheet, both age routes, the consent gate, the interview screen,
the expert's graph review, and the outcome battery end to end.

```bash
python tests/test_arms.py
```

Runs complete sessions through all four arms against a scripted fake provider.
Covers the adequacy gate, the re-probe loop, the soft cap, labelled-edge
traversal, traversal-error recovery, the skip control and coverage accounting.

```bash
python tests/test_provider_requests.py
```

Validates the exact request built for each provider against that provider's real
rules: `messages` is never empty (the opening turn is system-only), `effort` is
only sent to models that accept it, `temperature` never reaches a current Claude
model, and a mid-conversation control instruction stays at the *end* of the
exchange rather than being hoisted into the system prompt. Uses fake clients, so
no key and no network.

```bash
python tests/test_model_compatibility.py
```

Builds fake models that reject each request parameter in turn -- and in
combination -- and asserts every one of them still completes: `max_tokens` vs
`max_completion_tokens`, `temperature`, `effort`, structured outputs, and output
ceilings. Also checks that a correct prior costs no wasted round trip, that a
learned limitation is cached rather than rediscovered every turn, that only the
offending parameter is dropped, and that genuine faults (unknown model, provider
outage) still fail loudly instead of silently degrading.

```bash
python tests/test_provider_switch.py
```

Drives the model pickers through real Streamlit reruns: switching provider
updates the model list, lands on a valid model for that provider, and remembers
each provider's own last pick.

`test_screens.py`, `test_arms.py` and `test_provider_switch.py` need `streamlit`, `openai`, `anthropic` and
`google-genai` installed. They set dummy API keys; no real call is made on any
path they exercise, and where one would be, the failure path is what is asserted.
