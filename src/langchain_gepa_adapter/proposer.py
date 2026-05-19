from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from gepa.strategies.instruction_proposal import InstructionProposalSignature
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger("langchain_gepa_adapter.proposer")

ReflectionLM = Callable[[str], str]


def make_reflection_lm(model: BaseChatModel | Mapping[str, Any] | str) -> ReflectionLM:
    """Wrap a LangChain chat model into the `(prompt: str) -> str` callable GEPA expects.

    `model` accepts either:
      - a `BaseChatModel` instance,
      - a `provider:model` string (passed to `init_chat_model`), or
      - a kwargs mapping (passed via `init_chat_model(**model)`).
    """
    if not isinstance(model, BaseChatModel):
        from langchain.chat_models import init_chat_model

        model = (
            init_chat_model(model)
            if isinstance(model, str)
            else init_chat_model(**dict(model))
        )

    def reflection_lm(prompt: str) -> str:
        logger.debug("reflection prompt (%d chars):\n%s", len(prompt), prompt)
        result = model.invoke([HumanMessage(content=prompt)])
        content = result.content
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content)
        logger.debug("reflection response (%d chars):\n%s", len(text), text)
        return text

    return reflection_lm


def make_default_proposer(reflection_lm: ReflectionLM):
    """GEPA's default reflective proposer, parameterized by a reflection LM callable."""

    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        logger.info("proposing new texts for components=%s", components_to_update)
        results: dict[str, str] = {}
        for name in components_to_update:
            records = reflective_dataset.get(name)
            if not records:
                logger.warning("no reflective records for component %s; skipping", name)
                continue
            logger.info("  %s: reflecting on %d records", name, len(records))
            new_text = InstructionProposalSignature.run(
                lm=reflection_lm,
                input_dict={
                    "current_instruction_doc": candidate[name],
                    "dataset_with_feedback": list(records),
                },
            )["new_instruction"]
            logger.info(
                "  %s: proposed new text (%d chars, was %d)",
                name,
                len(new_text),
                len(candidate[name]),
            )
            results[name] = new_text
        return results

    return propose_new_texts
