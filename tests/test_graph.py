"""
test_graph.py — 图路由逻辑单元测试
Day 3-4 就可以跑，不依赖 LLM，纯逻辑验证。

运行方式：
    pytest tests/test_graph.py -v
"""

from multi_agent_research.core.state import ResearchState, AnalystVerdict, initial_state
from multi_agent_research.core.supervisor import (
    MAX_ITERATIONS,
    TOKEN_BUDGET,
    route_from_supervisor,
    should_terminate,
)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def make_state(**overrides) -> ResearchState:
    """创建测试用 State，支持覆盖任意字段"""
    base = initial_state("测试研究问题")
    base.update(overrides)
    return base


# ─────────────────────────────────────────────
# 终止条件测试
# ─────────────────────────────────────────────

class TestTerminationConditions:

    def test_no_terminate_at_start(self):
        state = make_state(iteration_count=0)
        assert should_terminate(state) is False

    def test_terminate_on_max_iterations(self):
        state = make_state(iteration_count=MAX_ITERATIONS)
        assert should_terminate(state) is True

    def test_not_terminate_below_max(self):
        state = make_state(iteration_count=MAX_ITERATIONS - 1)
        assert should_terminate(state) is False

    def test_terminate_on_token_budget(self):
        state = make_state(token_budget_used=TOKEN_BUDGET)
        assert should_terminate(state) is True

    def test_task_status_does_not_bypass_resource_guards(self):
        state = make_state(task_status={
            "task_1": "pass",
            "task_2": "pass",
            "task_3": "pass",
        })
        assert should_terminate(state) is False

    def test_no_terminate_when_some_tasks_pending(self):
        state = make_state(task_status={
            "task_1": "pass",
            "task_2": "revise",
        })
        assert should_terminate(state) is False

    def test_no_terminate_when_task_status_empty(self):
        state = make_state(task_status={})
        assert should_terminate(state) is False


# ─────────────────────────────────────────────
# 路由测试
# ─────────────────────────────────────────────

class TestRouting:

    def test_routes_to_search_by_default(self):
        state = make_state(next_agent="search_agent")
        result = route_from_supervisor(state)
        assert result == "search_agent"

    def test_routes_to_analyst(self):
        state = make_state(next_agent="analyst_agent")
        result = route_from_supervisor(state)
        assert result == "analyst_agent"

    def test_routes_to_writer(self):
        state = make_state(next_agent="writer_agent")
        result = route_from_supervisor(state)
        assert result == "writer_agent"

    def test_routes_to_end_when_terminated(self):
        from langgraph.graph import END
        state = make_state(iteration_count=MAX_ITERATIONS)
        result = route_from_supervisor(state)
        assert result == END


# ─────────────────────────────────────────────
# State 初始化测试
# ─────────────────────────────────────────────

class TestStateInitialization:

    def test_initial_state_has_required_keys(self):
        state = initial_state("测试问题")
        required_keys = [
            "research_question", "parent_context", "task_plan", "next_agent",
            "task_status", "iteration_count", "token_budget_used",
            "messages", "search_results", "analyst_verdict",
            "writer_status", "events", "final_report"
        ]
        for key in required_keys:
            assert key in state, f"缺少必要字段：{key}"

    def test_initial_iteration_count_is_zero(self):
        state = initial_state("测试问题")
        assert state["iteration_count"] == 0

    def test_initial_verdict_is_not_reviewed(self):
        state = initial_state("测试问题")
        assert state["analyst_verdict"].verdict == "not_reviewed"
        assert state["analyst_verdict"].is_reviewed is False

    def test_research_question_preserved(self):
        question = "这是一个具体的研究问题"
        state = initial_state(question)
        assert state["research_question"] == question

    def test_parent_context_is_explicit_and_optional(self):
        assert initial_state("测试")["parent_context"] is None
        context = {"source_run_id": "run-parent", "report_excerpt": "evidence"}
        assert initial_state("测试", parent_context=context)["parent_context"] == context


# ─────────────────────────────────────────────
# Analyst Verdict 结构测试
# ─────────────────────────────────────────────

class TestAnalystVerdict:

    def test_verdict_values(self):
        for v in ["pass", "revise", "reject"]:
            verdict = AnalystVerdict(
                verdict=v, reason="测试", specific_gaps=[], confidence_score=0.5
            )
            assert verdict.verdict == v

    def test_confidence_score_range(self):
        """置信度应在 0-1 之间（业务规则，不是类型约束）"""
        verdict = AnalystVerdict(
            verdict="pass", reason="测试", specific_gaps=[], confidence_score=0.8
        )
        assert 0.0 <= verdict.confidence_score <= 1.0

    def test_specific_gaps_is_list(self):
        verdict = AnalystVerdict(
            verdict="revise",
            reason="需要补充",
            specific_gaps=["缺少近期数据", "缺少对比实验"],
            confidence_score=0.4,
        )
        assert isinstance(verdict.specific_gaps, list)
        assert len(verdict.specific_gaps) == 2
