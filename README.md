# Multi-language AI customer assistant (multi-tenant SaaS)

Businesses (tenants) talk to their customers in the customer's own language over **WhatsApp**
(phone calls are stubbed for phase 2). Launch languages: **Telugu, Hindi, English** (adding one
is config only). First industry: **clinics** (book / reschedule / cancel, FAQs, report status);
also a **customer-service** industry (order status, returns, complaints, human handoff).

Priorities, in order: **low cost per conversation, accuracy (never invent), low latency, new
industry = config not code.**

## How a message is handled

```
WhatsApp -> /webhooks/whatsapp (verify signature, dedupe, rate-limit) -> Arq queue -> worker
worker: lock (tenant, phone) -> load state from Postgres -> Router -> reply -> save -> send
```

Router, cheapest path first (each step only runs if the previous could not decide):

1. **Rules** (zero cost): button taps, yes/no, keyword intents, greetings, business hours.
2. **Approved FAQ** answers: exact match, then semantic, then lexical (verbatim, no generation).
3. **Semantic cache** per tenant + language.
4. **Small LLM, JSON only**: intent + slots (`router/extractor.py`); the LLM only fills gaps.
5. **Small LLM + RAG**: top 3 short chunks, then deterministic checks (every number in the
   answer must appear in the retrieved text).
6. **Stronger LLM** only when the small one is unsure; **human handoff** with an auto summary.

The **state machine** (`orchestrator/engine.py`) owns the flow: greet, identify, collect slots,
confirm, execute tool, close. Every critical slot is shown back for confirmation; time-sensitive
choices (e.g. a slot) are **re-validated against live data** before booking and again at the
moment of booking. State lives only in Postgres, so a customer can reply hours later.

## What's where

| Folder | Purpose |
|---|---|
| `api/` | FastAPI: WhatsApp webhook, health, admin (handoff inbox, cost report) |
| `channels/` | WhatsApp adapter (+ voice and web-chat stubs); all produce one `InboundMessage` |
| `orchestrator/` | state machine, flows (`industries/*.yaml`), language detection, handoff, service |
| `router/` | rules, hours, LLM extractor, semantic cache |
| `llm/` | provider-agnostic client (LiteLLM), model profiles + runtime switching, circuit breaker + fallback, usage/trace |
| `rag/` | chunker, retriever (pgvector), grounding guards, knowledge answerer |
| `tools/` | booking (Postgres) + Google Calendar mirror, order/report/returns/CRM, send template |
| `tenants/` | config schema, YAML import, CLI, sample tenants |
| `jobs/` | Arq worker: inbound processing, reminders, expiry, send retries, calendar sync, handoff alerts |
| `evals/` | YAML conversation suite + metrics (accuracy, tokens, cost) |
| `tests/` | ~320 tests (unit + real Postgres/pgvector integration) |

## Run it

```bash
cp .env.example .env            # edit secrets; secrets only ever come from the environment
docker compose up --build       # postgres(+pgvector), redis, migrate, api, worker
# set API_PORT=... if port 8000 is taken on your machine

# create/update a tenant (validates the YAML first; embeds knowledge if EMBEDDING_MODEL is set)
docker compose run --rm worker python -m assistant.tenants.cli import \
    assistant/tenants/templates/clinic_sample.yaml
```

Then point Meta's webhook at `https://<host>/webhooks/whatsapp` using your
`WHATSAPP_VERIFY_TOKEN`; set `whatsapp_phone_number_id` in the tenant YAML to route the number.

Local dev: `python -m venv .venv && pip install -e ".[dev]"`, then `pytest`, `ruff check .`,
`mypy assistant`. DB tests use `TEST_DATABASE_URL` (default `.../assistant_test`) and skip
cleanly if Postgres is not reachable.

## Costs and models (switch any time, no redeploy)

Nothing paid is chosen for you. Models live in **named profiles** in the database; exactly one
is **active**. Provider API keys are read from the environment by LiteLLM
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, ...) and are **never stored**.

```bash
python -m assistant.llm.cli list
python -m assistant.llm.cli set fast-cheap --fast provider/small-model --strong provider/big-model         --fallback provider/other-model --embedding provider/embedding-model   # 1024-dim
python -m assistant.llm.cli test fast-cheap       # tiny real call per model (a few tokens)
python -m assistant.llm.cli activate fast-cheap   # verify, switch, re-embed if needed
python -m assistant.llm.cli activate rules-only   # back to zero cost, instantly
```

The same is available over HTTP (`X-Admin-Key`): `GET /admin/models`,
`PUT /admin/models/profiles/{name}`, `POST /admin/models/test`, `POST /admin/models/activate`.

* **Safe by default:** `activate` first probes every model; if any call fails (wrong id, missing
  key) **nothing changes**. `--no-verify` / `"verify": false` overrides.
* **Instant:** LLM models switch everywhere within ~5 s (per-process cache). The first boot
  seeds a `default` profile from `LLM_FAST_MODEL` / `LLM_STRONG_MODEL` / `LLM_FALLBACK_MODEL` /
  `EMBEDDING_MODEL`; after that the database is the source of truth.
* **`rules-only` profile** (built in): no LLM, no embeddings, zero API cost; FAQ answers and the
  whole button-driven booking journey still work.
* **Per tenant:** a tenant may set `models.profile` (use another profile's LLM models) or pin
  `models.fast` / `models.strong` in its YAML.
* **Embedding model** is platform-wide (one vector space). Switching it re-embeds the knowledge
  base in the background into a staging column while the OLD vectors keep serving; when all
  chunks are ready it swaps them in **atomically** and clears the cache. A failure midway changes
  nothing user-visible and the migration resumes when re-run. The model must output exactly
  1024 dimensions (the probe checks this).
* **Compare providers** without touching production:
  `python -m assistant.evals.run --live --fast provider/a --strong provider/b`.
* `GET /admin/tenants/{slug}/cost-report` shows cost and tokens per model, so you can see the
  effect of a switch. Each turn also logs a Langfuse-shaped `turn_trace` line (optional
  `LANGFUSE_*` export).

## Evaluate

```bash
python -m assistant.evals.run              # offline, deterministic: replays scripted LLM output
python -m assistant.evals.run --live       # uses your configured real LLM extractor
python -m assistant.evals.run --json --language te
```

54 sample conversations (Telugu, Hindi, English, code-mixed, romanized, a second industry) are
also a pytest gate (`tests/test_evals.py`). A new case is ~10 lines of YAML.

## Extend

* **New language:** add it to the tenant YAML (`languages:` with greetings / yes / no /
  keywords) and add its text to the industry message tables. No code.
* **New industry:** copy `orchestrator/industries/clinic.yaml`, change flows and texts, set
  `industry:` in the tenant YAML. (`support.yaml` was added this way.)
* **New tool:** a small typed class in `tools/`, registered in `tools/registry.py`; tenants
  opt in via `tools_enabled`.
* **Phone calls (phase 2):** implement `SpeechToText`, `TextToSpeech`, `TelephonyProvider` in
  `channels/voice/interfaces.py` (checklist at the top of that file). Everything else is reused.

## Security notes

Webhook signature verified over the raw body (fail closed); secrets only from env; phone numbers
and emails masked in logs; per-tenant keys on every query and cache key; SSRF guard on
tenant-configured integration URLs; admin API disabled unless `ADMIN_API_KEY` is set; bookings
and cancellations check ownership on every access.

## Known limitations / follow-ups

* Romanized Telugu/Hindi ("nenu doctor ni kalavali") is routed by the LLM, not the free rules.
* Admin key is platform-wide; per-tenant staff credentials are a follow-up.
* WhatsApp access token is global; per-tenant tokens (encrypted) are a follow-up.
* Delivery/read receipts from WhatsApp are ignored (no per-message delivery tracking yet).
* No STT/TTS/telephony vendor is chosen: voice notes get a polite "please type" reply until one is.
* Google Calendar is a one-way mirror and its service-account auth is not covered by tests (it
  needs real credentials); availability is always decided by Postgres.
* RAG/semantic quality in Telugu/Hindi depends on the embedding model you choose and has not
  been measured here (tests use a non-semantic stand-in).
