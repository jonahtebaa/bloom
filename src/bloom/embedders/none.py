"""No-op embedder. Returns an empty vector; recall falls back to keyword scoring.

`dim = 0` is the sentinel that callers (recall, tools) check to skip the
embedding code path entirely. Importing numpy is deferred so the no-op path
stays lightweight on installs without the optional embedding extras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class NoneEmbedder:
    name = "none"
    dim = 0

    def embed_doc(self, content: str) -> "np.ndarray":  # noqa: ARG002
        import numpy as np

        return np.zeros(0, dtype=np.float32)

    def embed_query(self, query: str) -> "np.ndarray":  # noqa: ARG002
        import numpy as np

        return np.zeros(0, dtype=np.float32)
