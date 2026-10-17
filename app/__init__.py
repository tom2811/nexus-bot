"""Nexus-bot: Hybrid AI assistant backend."""

import os

# ChromaDB 0.6 is incompatible with posthog 3+ capture(); disable telemetry
# before any chromadb import so startup logs stay clean in host apps.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

__version__ = "1.0.0"
