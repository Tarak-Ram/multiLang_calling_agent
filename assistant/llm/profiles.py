"""Model profiles: switch LLM provider / embedding model at runtime, no redeploy.

A profile is a named set of LiteLLM model ids (fast / strong / fallback / embedding). One profile
is ACTIVE platform-wide; a tenant may pick a different profile for its LLM models, and may still
pin individual models in its own config. API keys are NEVER stored here: they stay in the
environment, where LiteLLM reads them (switching to a provider whose key is missing is caught by
the verification probe before the switch happens).

Propagation: every process caches the snapshot for `ttl_s` (default 5 s), so a switch takes
effect everywhere within seconds with no restart and no extra infrastructure.

The EMBEDDING model is platform-wide on purpose (one vector space). It changes in two phases
(see llm/switch.py): `embedding_effective` keeps pointing at the old model until every chunk has
been re-embedded with the new one, then flips atomically, so retrieval never sees mixed spaces.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.config import Settings
from assistant.core.models import EMBEDDING_DIM, ModelProfile, PlatformSetting
from assistant.tenants.schema import TenantConfig

log = logging.getLogger("assistant.llm.profiles")

RULES_ONLY = "rules-only"  # built-in: no LLM, no embeddings (zero API cost)
DEFAULT = "default"  # seeded once from the LLM_*/EMBEDDING_* environment variables
ACTIVE_KEY, EFFECTIVE_KEY, MIGRATION_KEY = "active_profile", "embedding_effective", "embedding_migration"


class ProfileSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,62}$")
    fast: str = ""
    strong: str = ""
    fallback: str = ""
    embedding_model: str = ""
    embedding_dim: int = EMBEDDING_DIM
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedModels:
    """What a request should actually use right now. Empty string = that capability is off."""

    fast: str = ""
    strong: str = ""
    fallback: str = ""
    embedding_model: str = ""
    embedding_dim: int = EMBEDDING_DIM
    profile: str = ""


class ModelResolver(Protocol):
    async def resolve(self, cfg: TenantConfig) -> ResolvedModels: ...


class StaticModels:
    """Fixed models (tests, evals, simple deployments). Still honours tenant model overrides."""

    def __init__(self, fast: str = "", strong: str = "", fallback: str = "",
                 embedding_model: str = "", embedding_dim: int = EMBEDDING_DIM) -> None:
        self._m = ResolvedModels(fast, strong, fallback, embedding_model, embedding_dim, "static")

    async def resolve(self, cfg: TenantConfig) -> ResolvedModels:
        return apply_tenant_overrides(self._m, cfg)


def apply_tenant_overrides(base: ResolvedModels, cfg: TenantConfig) -> ResolvedModels:
    """Tenant-pinned models win over the profile's (embedding stays platform-wide)."""
    fast, strong = cfg.models.fast or base.fast, cfg.models.strong or base.strong
    if (fast, strong) == (base.fast, base.strong):
        return base
    return ResolvedModels(fast, strong, base.fallback, base.embedding_model, base.embedding_dim,
                          base.profile)


@dataclass(slots=True)
class Snapshot:
    profiles: dict[str, ProfileSpec]
    active: str
    embedding_model: str  # effective: what retrieval uses right now
    embedding_dim: int
    migration: dict[str, Any]


class ProfileStore:
    """DB-backed profiles + settings with a short in-process cache. Implements ModelResolver."""

    def __init__(self, factory: async_sessionmaker[AsyncSession],
                 env_defaults: ProfileSpec | None = None, ttl_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._f, self._env, self._ttl, self._clock = factory, env_defaults, ttl_s, clock
        self._snap: Snapshot | None = None
        self._at = 0.0
        self._seeded = False

    # ------------------------------------------------------------------ reads
    async def snapshot(self, fresh: bool = False) -> Snapshot:
        if not fresh and self._snap and self._clock() - self._at < self._ttl:
            return self._snap
        async with self._f() as s:
            profiles = {p.name: ProfileSpec(
                name=p.name, fast=p.fast, strong=p.strong, fallback=p.fallback,
                embedding_model=p.embedding_model, embedding_dim=p.embedding_dim, notes=p.notes)
                for p in (await s.execute(select(ModelProfile))).scalars().all()}
            settings = {r.key: r.value for r in (await s.execute(select(PlatformSetting))
                                                 ).scalars().all()}
        active = str(settings.get(ACTIVE_KEY, {}).get("name", RULES_ONLY))
        eff = settings.get(EFFECTIVE_KEY, {})
        self._snap = Snapshot(
            profiles=profiles, active=active if active in profiles else RULES_ONLY,
            embedding_model=str(eff.get("model", "")), embedding_dim=int(eff.get("dim", EMBEDDING_DIM)),
            migration=dict(settings.get(MIGRATION_KEY, {"state": "idle"})),
        )
        self._at = self._clock()
        return self._snap

    def invalidate(self) -> None:
        """Drop this process's cache (the writer calls it; others catch up within ttl_s)."""
        self._snap = None

    async def resolve(self, cfg: TenantConfig) -> ResolvedModels:
        snap = await self.snapshot()
        chosen = snap.profiles.get(cfg.models.profile or "") or snap.profiles.get(snap.active)
        chosen = chosen or ProfileSpec(name=RULES_ONLY)
        base = ResolvedModels(chosen.fast, chosen.strong, chosen.fallback,
                              snap.embedding_model, snap.embedding_dim, chosen.name)
        return apply_tenant_overrides(base, cfg)

    # ----------------------------------------------------------------- writes
    async def ensure_seeded(self) -> None:
        """First boot: create the built-in profiles and activate `default` (from env)."""
        if self._seeded:
            return
        async with self._f() as s, s.begin():
            have = {p for p in (await s.execute(select(ModelProfile.name))).scalars().all()}
            if RULES_ONLY not in have:
                s.add(ModelProfile(name=RULES_ONLY, notes="No LLM, no embeddings: zero API cost"))
            if DEFAULT not in have:
                env = self._env or ProfileSpec(name=DEFAULT)
                s.add(ModelProfile(name=DEFAULT, fast=env.fast, strong=env.strong,
                                   fallback=env.fallback, embedding_model=env.embedding_model,
                                   embedding_dim=env.embedding_dim, notes="Seeded from environment"))
            existing = {r.key for r in (await s.execute(select(PlatformSetting))).scalars().all()}
            if ACTIVE_KEY not in existing:
                env = self._env or ProfileSpec(name=DEFAULT)
                s.add(PlatformSetting(key=ACTIVE_KEY, value={"name": DEFAULT}))
                s.add(PlatformSetting(key=EFFECTIVE_KEY, value={
                    "model": env.embedding_model, "dim": env.embedding_dim}))
        self._seeded = True
        self.invalidate()

    async def upsert_profile(self, spec: ProfileSpec) -> None:
        if spec.embedding_dim != EMBEDDING_DIM and spec.embedding_model:
            raise ValueError(f"embedding_dim must be {EMBEDDING_DIM} (the database column size); "
                             "pick a {0}-dimension model or run a migration".format(EMBEDDING_DIM))
        values = spec.model_dump()
        async with self._f() as s, s.begin():
            stmt = pg_insert(ModelProfile).values(**values)
            await s.execute(stmt.on_conflict_do_update(
                index_elements=["name"], set_={k: v for k, v in values.items() if k != "name"}))
        self.invalidate()

    async def delete_profile(self, name: str) -> bool:
        if name in (RULES_ONLY, DEFAULT):
            raise ValueError("built-in profiles cannot be deleted")
        snap = await self.snapshot(fresh=True)
        if name == snap.active:
            raise ValueError("cannot delete the active profile")
        async with self._f() as s, s.begin():
            res = await s.execute(delete(ModelProfile).where(ModelProfile.name == name))
        self.invalidate()
        return bool(res.rowcount)  # type: ignore[attr-defined]

    async def _set(self, key: str, value: dict[str, Any],
                   session: AsyncSession | None = None) -> None:
        stmt = pg_insert(PlatformSetting).values(key=key, value=value)
        upsert = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value})
        if session is not None:  # join the caller's transaction (used for the atomic swap)
            await session.execute(upsert)
        else:
            async with self._f() as s, s.begin():
                await s.execute(upsert)
        self.invalidate()

    async def set_active(self, name: str) -> None:
        await self._set(ACTIVE_KEY, {"name": name})

    async def set_embedding_effective(self, model: str, dim: int,
                                      session: AsyncSession | None = None) -> None:
        await self._set(EFFECTIVE_KEY, {"model": model, "dim": dim}, session)

    async def set_migration(self, state: dict[str, Any]) -> None:
        await self._set(MIGRATION_KEY, state)


def env_profile(s: Settings) -> ProfileSpec:
    """The `default` profile as described by the environment (used only to seed)."""
    return ProfileSpec(name=DEFAULT, fast=s.llm_fast_model, strong=s.llm_strong_model,
                       fallback=s.llm_fallback_model, embedding_model=s.embedding_model,
                       embedding_dim=s.embedding_dim)
