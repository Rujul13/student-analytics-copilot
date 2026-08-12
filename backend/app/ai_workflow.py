from __future__ import annotations

import logging
from typing import Any

from groq import AsyncGroq
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from .copilot import answer_question
from .data_agent import (
    GeneratedPandasProgram,
    build_schema_context,
    deterministic_answer_from_rows,
    generate_validate_execute,
    synthesize_answer,
)
from .models import QueryResponse
from .pandas_worker import WorkerExecutionResult
from .repository import DatasetContext
from .scope_validation import ScopeFilters, extract_scope, missing_field_answer


logger = logging.getLogger(__name__)


class ScopeCheckedQuestion(Event):
    question: str
    history: list[dict[str, str]]
    scope: ScopeFilters
    schema_context: dict[str, Any]
    short_circuit: QueryResponse | None


class ExecutedProgram(Event):
    question: str
    program: GeneratedPandasProgram
    execution: WorkerExecutionResult
    used_repair: bool


class AnalyticsWorkflow(Workflow):
    def __init__(self, dataset: DatasetContext, api_key: str, pandas_agent_model: str, answer_model: str):
        super().__init__(timeout=30, verbose=False)
        self.dataset = dataset
        self.pandas_agent_model = pandas_agent_model
        self.answer_model = answer_model
        self.client = AsyncGroq(api_key=api_key, timeout=12, max_retries=1)

    @step
    async def check_scope(self, ev: StartEvent) -> ScopeCheckedQuestion:
        question = str(ev.question)
        history = getattr(ev, "history", []) or []
        scope = extract_scope(question, self.dataset)
        short_circuit = None
        if scope.missing_fields:
            short_circuit = QueryResponse(
                answer=missing_field_answer(scope.missing_fields),
                result_type="unsupported",
                rows=[],
                execution_mode="unsupported",
                ai_used=False,
            )
        schema_context = build_schema_context(self.dataset)
        return ScopeCheckedQuestion(
            question=question, history=history, scope=scope, schema_context=schema_context, short_circuit=short_circuit
        )

    @step
    async def plan_and_execute(self, ev: ScopeCheckedQuestion) -> ExecutedProgram | StopEvent:
        if ev.short_circuit is not None:
            return StopEvent(result=ev.short_circuit)

        program, execution, error_message, previous_code = await generate_validate_execute(
            self.client, self.pandas_agent_model, ev.question, ev.schema_context, ev.history,
            self.dataset, ev.scope, None, None,
        )
        used_repair = False
        if program is None:
            used_repair = True
            program, execution, error_message, previous_code = await generate_validate_execute(
                self.client, self.pandas_agent_model, ev.question, ev.schema_context, ev.history,
                self.dataset, ev.scope, previous_code, error_message,
            )

        if program is None or execution is None:
            return StopEvent(
                result=QueryResponse(
                    answer="I could not compute a verified answer for that question from the active dataset.",
                    result_type="error",
                    rows=[],
                    execution_mode="unsupported",
                    ai_used=True,
                )
            )
        return ExecutedProgram(question=ev.question, program=program, execution=execution, used_repair=used_repair)

    @step
    async def synthesize(self, ev: ExecutedProgram) -> StopEvent:
        normalized_rows = ev.execution.rows or []
        try:
            answer = await synthesize_answer(
                self.client, self.answer_model, ev.question, ev.program.interpretation,
                normalized_rows, self.dataset.name, self.dataset.version,
            )
        except Exception:
            answer = deterministic_answer_from_rows(normalized_rows, ev.execution.result_type or ev.program.result_type)
        return StopEvent(
            result=QueryResponse(
                answer=answer,
                result_type="metric" if ev.execution.result_type == "scalar" else "table",
                rows=normalized_rows,
                execution_mode="generated-pandas-repaired" if ev.used_repair else "generated-pandas",
                ai_used=True,
            )
        )


async def run_copilot(
    dataset: DatasetContext,
    question: str,
    workflow: AnalyticsWorkflow | None,
    history: list[dict[str, str]] | None = None,
) -> QueryResponse:
    if workflow is None:
        return answer_question(dataset, question, ai_enabled=False)
    try:
        return await workflow.run(question=question, history=history or [])
    except Exception as error:
        logger.warning("Generated-Pandas workflow failed; using deterministic fallback (%s)", type(error).__name__)
        return answer_question(dataset, question, ai_enabled=False)
