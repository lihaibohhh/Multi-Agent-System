"""
compare_analyst_models.py — 对比 DeepSeek 与本地 vLLM 模型在 analyst_agent 审查任务上的表现

设计原则:
    不复制粘贴 analyst_agent.py 里的 prompt 拼接逻辑(容易写得不一致导致对比失真),
    而是直接 import 生产模块、直接调用真实的 analyst_agent_node() 函数,
    只在调用前临时把模块级的 _structured_llm 换成指向本地 vLLM 的版本,跑完立刻换回去。
    这样两次调用走的是完全相同的一份代码,唯一变量就是底下的模型。

本版新增"反向验证"机制:不止跑一组真实样例,而是跑三组对比鲜明的场景——
    1. 原始真实样例(来自一次真实的 search_agent 运行)
    2. 人工构造-信息明显充分(覆盖了 DeepSeek 之前指出的全部缺口)
    3. 人工构造-信息明显严重不足(低相关/离题内容)
目的是检验本地模型是否存在"系统性乐观"倾向——如果它在场景 3 也给出 pass/高置信度,
说明问题不是这次样本恰好踩中弱点，而是模型本身没有真正执行批判性审查。

用法:
    1. 确认 vllm serve 正在跑,且 LOCAL_LLM_BASE_URL 已经指向它
    2. 把下面三个 json 文件(sample_search_results.json / sample_results_rich.json /
       sample_results_poor.json)放在跟本脚本同一个 scripts/ 文件夹下
    3. 把 LOCAL_MODEL_REF 改成你当前实际跑着的本地模型名
    4. 在项目根目录下运行:
       python -m scripts.compare_analyst_models
"""

import asyncio
import json
import pathlib
from pydantic import ValidationError

import multi_agent_research.agents.analyst_agent as analyst_module


QUESTION = "2026年以来国内AI教育行业有哪些重大进展？结合市场格局分析当前投资价值"
SCRIPT_DIR = pathlib.Path(__file__).parent

# 按你当前实际跑着的本地模型调整这一行
LOCAL_MODEL_REF = "local/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

SCENARIOS = [
    ("原始真实样例(7条,分数0.31~0.99)", "sample_search_results.json"),
    ("人工构造-信息明显充分(覆盖DeepSeek之前指出的全部缺口)", "sample_results_rich.json"),
    ("人工构造-信息明显严重不足(2条,低相关或离题)", "sample_results_poor.json"),
]


def load_results(filename: str) -> list:
    path = SCRIPT_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_verdict(label: str, node_output: dict):
    verdict = node_output["analyst_verdict"]
    if hasattr(verdict, "model_dump"):
        d = verdict.model_dump()
        gaps = d["specific_gaps"] if d["specific_gaps"] else "(无)"
        print(f"  [{label:<10}] verdict={d['verdict']:<7} confidence={d['confidence_score']:.2f}  gaps={gaps}")
        print(f"  [{label:<10}] reason={d['reason']}")
    else:
        print(f"  [{label:<10}] {verdict}")


def build_local_structured_llm():
    local_llm = analyst_module.load_chat_model(LOCAL_MODEL_REF)
    return (
        local_llm
        .with_structured_output(analyst_module._AnalystVerdictOutput, method="json_mode")
        .with_retry(
            retry_if_exception_type=(ValidationError, ValueError),
            stop_after_attempt=2,
        )
    )


async def run_scenario(scenario_label: str, filename: str, local_structured_llm):
    results = load_results(filename)
    state = {
        "research_question": QUESTION,
        "search_results": results,
        "iteration_count": 1,
    }

    print(f"\n=== 场景: {scenario_label}  (共 {len(results)} 条结果) ===")

    # 1. DeepSeek(生产基线，模块导入时已初始化好)
    ds_output = await analyst_module.analyst_agent_node(state)
    print_verdict("DeepSeek", ds_output)

    # 2. 临时把模块内的 _structured_llm 换成本地 vLLM 版本
    original_structured_llm = analyst_module._structured_llm
    analyst_module._structured_llm = local_structured_llm
    try:
        local_output = await analyst_module.analyst_agent_node(state)
        print_verdict("本地vLLM", local_output)
    finally:
        # 无论成功失败都要换回去，避免污染后续对生产代码的调用
        analyst_module._structured_llm = original_structured_llm


async def main():
    local_structured_llm = build_local_structured_llm()
    for label, filename in SCENARIOS:
        if not (SCRIPT_DIR / filename).is_file():
            print(f"跳过未提供的本地样例：{filename}")
            continue
        await run_scenario(label, filename, local_structured_llm)

    print("\n=== 跑完了。重点看本地模型在第3个场景(信息明显不足)的 verdict —— ===")
    print("=== 如果它依然给 pass 或者给出高置信度，说明这不是偶然，是系统性的乐观倾向 ===")


if __name__ == "__main__":
    asyncio.run(main())
