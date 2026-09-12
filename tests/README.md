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

`test_screens.py` and `test_arms.py` need `streamlit`, `openai`, `anthropic` and
`google-genai` installed. They set dummy API keys; no real call is made on any
path they exercise, and where one would be, the failure path is what is asserted.
