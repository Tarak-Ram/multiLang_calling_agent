"""Conversation orchestrator: a deterministic state machine per flow.

States: greet -> identify_intent -> collect_slots -> confirm -> (execute tool) -> close,
plus handoff. The LLM extractor (optional) only fills gaps the rules couldn't; every state
transition, validation and tool call here is plain code.

`Orchestrator.handle` is pure with respect to storage: it takes the persisted ConvState in and
returns the new one, so it is trivially testable and nothing lives in memory between webhooks.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from assistant.channels.base import Option
from assistant.core.models import ConversationStatus
from assistant.orchestrator.flows import FlowDef, IndustryDef, SlotDef
from assistant.orchestrator.language import detect_language, resolve_language
from assistant.orchestrator.types import (
    Answerer,
    ConvState,
    ExtractHints,
    Extraction,
    Extractor,
    Reply,
    TurnInput,
    TurnResult,
)
from assistant.router.hours import is_open
from assistant.router.rules import RuleHit, apply_rules, normalize
from assistant.tenants.schema import TenantConfig
from assistant.tools.base import ToolContext, ToolEnv, ToolNotEnabledError, ToolRegistry

log = logging.getLogger("assistant.orchestrator")

MAX_RETRIES = 2  # re-asks per slot before we give up and hand off
MAX_OPTIONS = 10  # WhatsApp list limit
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_/]{2,39}$")


class _Handoff(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Orchestrator:
    """Shared, stateless; build once per process."""

    def __init__(
        self,
        industries: dict[str, IndustryDef],
        tools: ToolRegistry,
        extractor: Extractor | None = None,
        answerer: Answerer | None = None,
    ) -> None:
        self.industries = industries
        self.tools = tools
        self.extractor = extractor
        self.answerer = answerer

    async def handle(
        self, cfg: TenantConfig, industry: str, state: ConvState, inp: TurnInput, now: datetime,
        env: ToolEnv | None = None,
    ) -> TurnResult:
        """Process one customer message and return replies + the new state."""
        turn = _Turn(self, cfg, self.industries[industry], state.model_copy(deep=True), inp, now,
                     env or ToolEnv())
        return await turn.run()


class _Turn:
    """One message's worth of work. Mutates its own copy of the state."""

    def __init__(
        self, orch: Orchestrator, cfg: TenantConfig, defn: IndustryDef, state: ConvState,
        inp: TurnInput, now: datetime, env: ToolEnv,
    ) -> None:
        self.o, self.cfg, self.defn, self.st, self.inp, self.now = orch, cfg, defn, state, inp, now
        self.env = env
        self.replies: list[Reply] = []
        self.route = "rule"
        self.tool_calls: list[dict[str, Any]] = []
        self.needs_human = False
        self.handoff_reason: str | None = None
        self.extraction: Extraction | None = None
        self.lang = state.language or cfg.default_language

    # ---------------------------------------------------------------- helpers
    @property
    def flow(self) -> FlowDef:
        assert self.st.flow
        return self.defn.flows[self.st.flow]

    @property
    def _labels(self) -> dict[str, str]:
        labels: dict[str, str] = self.st.slots.setdefault("_labels", {})
        return labels

    @property
    def _retries(self) -> dict[str, int]:
        retries: dict[str, int] = self.st.slots.setdefault("_retries", {})
        return retries

    def text(self, key: str, **extra: Any) -> str:
        template = self.defn.text(self.cfg, key, self.lang)
        values: dict[str, Any] = {"phone": self.st.customer_phone, **extra}
        labels = self._labels  # fetched first: this property may insert "_labels" into slots
        for k, v in list(self.st.slots.items()):
            if not k.startswith("_"):
                values[k] = labels.get(k, v)
        return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), template)

    def say(self, key: str, options: list[Option] | None = None, **extra: Any) -> None:
        opts = (options or [])[:MAX_OPTIONS]
        self.replies.append(
            Reply(text=self.text(key, **extra), options=opts,
                  list_button=self.text("choose") if len(opts) > 3 else "")
        )

    def result(self) -> TurnResult:
        return TurnResult(
            replies=self.replies, state=self.st, route=self.route, needs_human=self.needs_human,
            handoff_reason=self.handoff_reason, tool_calls=self.tool_calls,
        )

    def _tool_ctx(self) -> ToolContext:
        return ToolContext(
            tenant_id=self.st.tenant_id, customer_phone=self.st.customer_phone,
            conversation_id=self.st.conversation_id, config=self.cfg, language=self.lang,
            session=self.env.session, account_id=self.env.account_id,
        )

    async def _call(self, name: str, args: dict[str, Any]) -> Any:
        try:
            res = await self.o.tools.run(name, self._tool_ctx(), args)
        except ToolNotEnabledError:
            raise _Handoff(f"tool_unavailable:{name}") from None
        except Exception:
            log.exception("tool_crashed", extra={"tool": name})
            raise _Handoff(f"tool_crashed:{name}") from None
        self.tool_calls.append({"tool": name, "ok": res.ok, "error": res.error_code})
        return res

    def _public_slots(self) -> dict[str, Any]:
        return {k: v for k, v in self.st.slots.items() if not k.startswith("_")}

    async def _options(self, slot: SlotDef) -> list[Option]:
        assert slot.options_tool
        res = await self._call(slot.options_tool, {"slots": self._public_slots()})
        return [Option.model_validate(o) for o in res.data.get("options", [])]

    def _hints(self, slot: SlotDef | None = None, options: list[Option] | None = None
               ) -> ExtractHints:
        local = self.now.astimezone(ZoneInfo(self.cfg.timezone))
        missing = ([s.name for s in self.flow.slots if s.name not in self.st.slots]
                   if self.st.flow else [])
        intents = [*self._available_flows(), "faq", "human", "medical_advice", "other"]
        return ExtractHints(
            intents=intents, today=local.strftime("%Y-%m-%d %a"), slot=slot.name if slot else None,
            options=options or [], slot_names=missing,
        )

    async def _extract(self, hints: ExtractHints) -> Extraction | None:
        """Call the LLM extractor (only reached when rules failed). Applies the tenant's
        confidence threshold and a language correction for romanized Telugu/Hindi."""
        if not self.o.extractor or not (self.inp.text or "").strip():
            return None
        ext = await self.o.extractor.extract(self.cfg, self.st, self.inp.text or "", hints)
        if ext is None:
            return None
        if ext.needs_human:
            raise _Handoff("extractor_needs_human")
        if ext.confidence < self.cfg.thresholds.intent_confidence:
            return None
        enabled = {lang.code for lang in self.cfg.languages}
        if (ext.language in enabled and ext.language != self.lang
                and detect_language(self.inp.text or "", list(enabled)) == "en"):
            # Latin-script text that is really romanized Telugu/Hindi: switch reply language.
            assert ext.language is not None
            self.lang = self.st.language = ext.language
        self.route = "extractor"
        return ext

    async def _lookup_faq(self, exact_only: bool = False) -> bool:
        """Free/cheap path tried BEFORE any LLM call: approved FAQ or semantic-cache hit."""
        if not self.o.answerer or not (self.inp.text or "").strip():
            return False
        ans = await self.o.answerer.lookup(self.cfg, self.st, self.inp.text or "", self.lang,
                                           exact_only=exact_only)
        if not ans:
            return False
        self.replies.append(Reply(text=ans.text))
        self.route = ans.source
        return True

    async def _answer_question(self) -> bool:
        """Knowledge question (extractor said 'faq'): RAG answer or admit we don't know."""
        if not self.o.answerer:
            return False
        ans = await self.o.answerer.generate(self.cfg, self.st, self.inp.text or "", self.lang)
        if ans:
            self.replies.append(Reply(text=ans.text))
            self.route = ans.source
        else:
            self.say("dont_know", options=self._menu_options())  # say so + offer a human
            self.route = "unknown"
        return True

    # ------------------------------------------------------------------- main
    async def run(self) -> TurnResult:
        st = self.st
        if st.status == ConversationStatus.HANDOFF:  # a human owns this conversation
            self.route = "human_owned"
            return self.result()
        enabled = [lang.code for lang in self.cfg.languages]
        self.lang = st.language = resolve_language(
            st.language, self.inp.text, enabled, self.cfg.default_language
        )
        recap = st.slots.pop("_recap", None)
        if isinstance(recap, dict) and recap.get("flow"):
            # Returning after the conversation expired: one line of context, then proceed
            # normally with what the customer just wrote (fresh state, nothing stale).
            self.say("recap", flow_name=self.text(f"flow_{recap['flow']}"))
        try:
            await self._dispatch()
        except _Handoff as h:
            self._handoff(h.reason)
        return self.result()

    async def _dispatch(self) -> None:
        st, inp = self.st, self.inp
        if not (inp.text or "").strip() and not inp.payload_id:
            self.say("unsupported_media")  # only reached if no STT transcribed the voice note first (see channels/voice/voicenote.py)
            if st.state in ("collect_slots", "confirm"):
                await self._advance()
            return

        hit = apply_rules(self.cfg, inp.text, inp.payload_id)
        if hit and hit.kind == "payload" and hit.value.startswith("menu:"):
            hit = RuleHit("intent", hit.value[5:])  # menu taps behave like keyword intents

        if (hit and hit.kind == "intent" and hit.value not in ("human", "medical_advice", "menu")
                and await self._lookup_faq(exact_only=True)):
            # An approved FAQ asked verbatim beats keyword intents ("What is your return
            # policy?" must not start a return flow). Mid-flow, re-ask the pending slot.
            if st.state == "collect_slots":
                await self._advance()
            else:
                st.state = "identify_intent"
            return

        if hit and hit.kind == "intent":  # global intents work in any state
            if hit.value == "human":
                raise _Handoff("user_requested")
            if hit.value == "medical_advice":
                self.say("medical_refusal")  # never advise; staff takes over
                raise _Handoff("medical_question")
            if hit.value == "menu":
                self._reset_flow()
                self._show_menu("greeting")
                return

        if st.state == "collect_slots":
            await self._collect(hit)
        elif st.state == "confirm":
            await self._confirm(hit)
        else:  # greet / identify_intent / close
            await self._identify(hit)

    # --------------------------------------------------------------- identify
    def _available_flows(self) -> list[str]:
        return [f for f in self.defn.flows if f in self.cfg.flows_enabled]

    def _menu_options(self) -> list[Option]:
        return [
            Option(id=f"menu:{m.id}", title=self.text(f"menu_{m.id}"))
            for m in self.defn.menu
            if m.action or (m.flow and m.flow in self._available_flows())
        ]

    def _show_menu(self, intro_key: str) -> None:
        self.st.state = "identify_intent"
        self.say(intro_key, options=self._menu_options())

    def _reset_flow(self) -> None:
        self.st.flow, self.st.state, self.st.slots = None, "identify_intent", {}

    async def _identify(self, hit: RuleHit | None) -> None:
        action: str | None = hit.value if hit and hit.kind == "intent" else None
        if hit and hit.kind == "greeting":
            self._show_menu("greeting")
            return
        if action is None and await self._lookup_faq():
            self.st.state = "identify_intent"
            return
        if action is None:
            ext = await self._extract(self._hints())
            if ext and ext.intent:
                action, self.extraction = ext.intent, ext
        if action == "medical_advice":  # extractor-detected (rules catch the obvious ones)
            self.say("medical_refusal")
            raise _Handoff("medical_question")
        if action == "human":
            raise _Handoff("user_requested")
        if action == "faq" and await self._answer_question():
            self.st.state = "identify_intent"
            return
        if action in self._available_flows():
            self.st.flow, self.st.state, self.st.slots = action, "collect_slots", {}
            if self.extraction:
                await self._fill_from_extraction()
            await self._advance()
        elif action and action not in ("faq", "other"):
            # The intent is real but this tenant has no (enabled) flow for it: a person takes over.
            # New flows are added in the industry YAML, not here.
            raise _Handoff(f"flow_unavailable:{action}")
        else:
            self._bump("_menu")
            self._show_menu("didnt_understand")

    # ---------------------------------------------------------------- collect
    def _next_slot(self) -> SlotDef | None:
        return next((s for s in self.flow.slots if s.name not in self.st.slots), None)

    def _bump(self, key: str) -> None:
        self._retries[key] = self._retries.get(key, 0) + 1
        if self._retries[key] > MAX_RETRIES:
            raise _Handoff(f"unclear:{key}")

    def _match_option(self, text: str | None, payload: str | None, opts: list[Option]) -> Option | None:
        if payload:
            return next((o for o in opts if o.id == payload), None)
        t = normalize(text or "")
        if not t:
            return None
        if t.isdigit() and 1 <= int(t) <= len(opts):  # "2" picks the 2nd option
            return opts[int(t) - 1]
        return next((o for o in opts if t in (normalize(o.title), normalize(o.id))), None)

    @staticmethod
    def _valid_name(text: str | None) -> str | None:
        r"""A person's name in ANY script. Indic names contain combining vowel signs (Unicode
        category M*) that `\w`-style regexes reject, so we validate by Unicode category:
        letters + marks + space . ' - only, no digits, 2-60 chars, at most 5 words."""
        t = " ".join((text or "").split())
        if not (2 <= len(t) <= 60) or len(t.split()) > 5:
            return None
        if not all(unicodedata.category(c)[0] in "LM" or c in " .'-" for c in t):
            return None
        return t if sum(unicodedata.category(c)[0] == "L" for c in t) >= 2 else None

    @staticmethod
    def _valid_text(text: str | None) -> str | None:
        t = " ".join((text or "").split())
        return t if 2 <= len(t) <= 300 else None

    @staticmethod
    def _valid_token(text: str | None) -> str | None:
        t = (text or "").strip()
        return t if _TOKEN_RE.match(t) else None

    async def _try_fill(self, slot: SlotDef, value_text: str | None, payload: str | None,
                        opts: list[Option] | None = None) -> bool:
        if slot.validate_as in ("name", "text", "token"):
            check = {"name": self._valid_name, "text": self._valid_text,
                     "token": self._valid_token}[slot.validate_as]
            value = None if payload else check(value_text)
            if value:
                self.st.slots[slot.name] = value
            return value is not None
        if opts is None:
            opts = await self._options(slot) if slot.options_tool else []
        chosen = self._match_option(value_text, payload, opts)
        if chosen:
            self.st.slots[slot.name] = chosen.id
            self._labels[slot.name] = chosen.title
        return chosen is not None

    async def _fill_from_extraction(self) -> None:
        """Apply LLM-extracted slots in flow order, each validated like user input."""
        assert self.extraction
        for slot in self.flow.slots:
            if slot.name in self.st.slots:
                continue
            value = self.extraction.slots.get(slot.name)
            if not value or not await self._try_fill(slot, value, None):
                break  # later slots may depend on this one (e.g. time depends on date)

    async def _collect(self, hit: RuleHit | None) -> None:
        slot = self._next_slot()
        if slot is None:
            await self._advance()
            return
        payload = hit.value if hit and hit.kind == "payload" else None
        opts = await self._options(slot) if slot.options_tool else []
        filled = await self._try_fill(slot, self.inp.text, payload, opts)
        if not filled and slot.validate_as == "option" and not payload:
            if await self._lookup_faq():  # e.g. "what are your timings?" mid-booking
                await self._advance()  # answered for free; re-ask the pending slot
                return
            ext = await self._extract(self._hints(slot, opts))
            if ext:
                self.extraction = ext
                if ext.intent == "faq" and await self._answer_question():
                    await self._advance()  # answered; now re-ask the pending slot
                    return
                if ext.intent == "medical_advice":
                    self.say("medical_refusal")
                    raise _Handoff("medical_question")
                if ext.intent == "human":
                    raise _Handoff("user_requested")
                await self._fill_from_extraction()
                filled = slot.name in self.st.slots
        if filled:
            self._retries.pop(slot.name, None)
            await self._advance()
        else:
            self._bump(slot.name)
            self.say("didnt_understand")
            await self._advance()  # re-ask the same slot with its options

    async def _revalidate(self) -> bool:
        """Re-check time-sensitive choices (slot may have been taken since we asked).
        Clears the stale slot and any later option slots. Returns True if all still valid."""
        for i, slot in enumerate(self.flow.slots):
            if not (slot.revalidate and slot.options_tool and slot.name in self.st.slots):
                continue
            if self.st.slots[slot.name] in {o.id for o in await self._options(slot)}:
                continue
            for later in self.flow.slots[i:]:
                if later.options_tool:
                    self.st.slots.pop(later.name, None)
                    self._labels.pop(later.name, None)
            return False
        return True

    async def _advance(self) -> None:
        """Ask for the next missing slot, or move to confirmation when all are filled."""
        if not await self._revalidate():
            self.say("slot_unavailable")
        slot = self._next_slot()
        self.st.state = "collect_slots"
        if slot is None and self.flow.confirm_prompt is None:
            await self._execute()  # read-only flow (e.g. order status): no confirmation needed
            return
        if slot is None:
            assert self.flow.confirm_prompt
            self.st.state = "confirm"
            self.say(
                self.flow.confirm_prompt,
                options=[Option(id="confirm_yes", title=self.text("btn_yes")),
                         Option(id="confirm_no", title=self.text("btn_no"))],
            )
            return
        opts = await self._options(slot) if slot.options_tool else []
        if slot.options_tool and not opts:
            if slot.on_empty:  # e.g. "you have no upcoming appointments": end flow, show menu
                self.say(slot.on_empty)
                self._reset_flow()
                self._show_menu("greeting")
                return
            raise _Handoff(f"no_options:{slot.name}")
        self.say(slot.prompt, options=opts)

    # ---------------------------------------------------------------- confirm
    async def _confirm(self, hit: RuleHit | None) -> None:
        payload = hit.value if hit and hit.kind == "payload" else None
        if (hit and hit.kind == "yes") or payload == "confirm_yes":
            if not await self._revalidate():  # slot may have been taken while we waited
                self.say("slot_unavailable")
                await self._advance()
                return
            await self._execute()
        elif (hit and hit.kind == "no") or payload == "confirm_no":
            self.st.slots = {}
            self.say("restart")
            await self._advance()
        else:
            self._bump("_confirm")
            self.say("didnt_understand")
            await self._advance()

    async def _execute(self) -> None:
        ex = self.flow.execute
        slots = self._public_slots()
        key_src = f"{self.st.conversation_id}:{sorted(slots.items())}"
        args = {**slots, "phone": self.st.customer_phone,
                "idempotency_key": hashlib.sha256(key_src.encode()).hexdigest()[:32]}
        res = await self._call(ex.tool, args)
        if res.ok:
            self.say(ex.success, **res.data)
            self.st.state, self.st.status = "close", ConversationStatus.CLOSED
            return
        action = ex.on_error.get(res.error_code or "")
        if action is None:
            raise _Handoff(f"tool_failed:{res.error_code}")
        for name in action.clear:
            self.st.slots.pop(name, None)
            self._labels.pop(name, None)
        self.say(action.message)
        if action.then == "handoff":
            raise _Handoff(f"tool_error:{res.error_code}")
        if action.then == "menu":
            self._reset_flow()
            self._show_menu("greeting")
            return
        await self._advance()

    # ---------------------------------------------------------------- handoff
    def _handoff(self, reason: str) -> None:
        """Hand the conversation to staff. The summary is built by the service; staff are alerted
        by jobs.handoff.notify_handoff."""
        self.st.status, self.st.state = ConversationStatus.HANDOFF, "handoff"
        self.needs_human, self.handoff_reason = True, reason
        if is_open(self.cfg, self.now):
            custom = self.cfg.escalation.message.get(self.lang)
            if custom:
                self.replies.append(Reply(text=custom))
            else:
                self.say("handoff")
        else:
            self.say("handoff_closed")
        log.info("handoff", extra={"reason": reason})
