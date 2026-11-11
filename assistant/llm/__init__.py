"""LLM layer.

LiteLLM downloads its model-pricing table from GitHub when first imported. A chat service must
not make surprise outbound calls at startup (reliability, air-gapped deploys), so we pin the
bundled copy BEFORE anything imports litellm. Costs for brand-new models may be missing from
the bundled map; they then log `cost_unknown` and record cost 0 (tokens are still recorded).
"""
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
