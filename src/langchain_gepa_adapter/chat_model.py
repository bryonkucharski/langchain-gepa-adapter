"""Build a LangChain chat model with optional JSON-file overrides.

By default, follows standard LangChain convention: pass a model spec like
`"openai:gpt-4o-mini"` and it goes through `init_chat_model` as usual.

If `config_path` is provided, the JSON file is loaded as kwargs to
`init_chat_model(**...)` instead — useful for keeping gateway-specific config
(base_url, custom headers, API keys) out of source files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def load_chat_model(
    model: str | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    **kwargs,
) -> BaseChatModel:
    if config_path is not None:
        return init_chat_model(**json.loads(Path(config_path).read_text()))
    if model is None:
        raise ValueError("Provide either `model` or `config_path`.")
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return init_chat_model(model, **kwargs)
