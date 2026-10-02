"""Run one configured chat-model request without embedding credentials in code."""

from __future__ import annotations

import argparse
import json

from langchain_core.messages import HumanMessage

from multi_agent_research.utils.llm import load_chat_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-ref",
        default="deepseek/deepseek-chat",
        help="Provider/model reference understood by load_chat_model().",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句话解释什么是闭包。",
        help="Prompt sent to the configured model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_chat_model(args.model_ref)
    response = model.invoke([HumanMessage(content=args.prompt)])
    print(response.content)
    print(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
