from __future__ import annotations

import functools
from typing import Any

from sqlalchemy import UUID, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.sqltypes import Boolean, Float, Integer, String, Text

from nanoeval.recorder import SystemError
from nanoeval.recorders.progress_reporter import ProgressReporter
from nanoeval.solvers.computer_tasks.code_execution_interface import ExecutionResult
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ExecutionResult as ExecutionResultDbObject,
)
from nanoeval.solvers.computer_tasks.legacy_api import (
    TraceAssistantEntity,
    TraceCompletion,
    TraceStep,
)
from nanoeval.solvers.computer_tasks.pausable_timer import PausableTimerEntity
from nanoeval.solvers.computer_tasks.task import TerminalEvaluation, TerminalStream
from nanoeval.solvers.computer_tasks.versioning.constants import STEP_API_VERSION
from nanoeval.solvers.mcq_api import AgentMessage, AgentTraceEntity
from nanoeval.solvers.step_closure import AbstractStep, StepClosure, with_concurrency
from nanoeval.sql_session import ObjectMapper


class AgentEntity(AgentTraceEntity, TraceStep.Entity, PausableTimerEntity):
    root_start: PausableTimerEntity


class TerminalStreamEntity(TerminalStream, PausableTimerEntity):
    __tablename__ = "TerminalStream"
    __mapper_args__ = {"polymorphic_identity": "TerminalStream"}

    uuid: Mapped[str] = mapped_column(String(), primary_key=True)
    task_uuid: Mapped[str] = mapped_column(String(), ForeignKey("Task.uuid"))
    run: Mapped[int] = mapped_column(Integer())
    group_id: Mapped[str] = mapped_column(String())
    step_id: Mapped[int] = mapped_column(Integer())
    agent_type: Mapped[str] = mapped_column(String())
    paused_stream_id: Mapped[int | None] = mapped_column(Integer())
    owner: Mapped[str] = mapped_column(String())
    message_type: Mapped[str] = mapped_column(String())

    timer_id: Mapped[int | None] = mapped_column(Integer())
    start_time: Mapped[int | None] = mapped_column(Float())

    step_resets: Mapped[int] = mapped_column(
        Integer(), doc="Number of times the terminal component was reset after this step occured"
    )

    type: Mapped[str] = mapped_column(String(), nullable=False)

    _paused_time: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict)

    _expected_prompt: Mapped[list[str]] = mapped_column(JSONB, default=list)
    _expected_code: Mapped[str | None] = mapped_column(String())

    _terminal_screen: Mapped[list[tuple[int, str]]] = mapped_column(JSONB, default=list)


class TerminalEvaluationEntity(TerminalEvaluation, PausableTimerEntity):
    __tablename__ = "TerminalEvaluation"
    __mapper_args__ = {"polymorphic_identity": "TerminalEvaluation"}

    uuid: Mapped[str] = mapped_column(String(), primary_key=True)
    task_uuid: Mapped[str] = mapped_column(String(), ForeignKey("Task.uuid"))
    run: Mapped[int] = mapped_column(Integer())
    group_id: Mapped[str] = mapped_column(String())
    step_id: Mapped[int] = mapped_column(Integer())
    agent_type: Mapped[str] = mapped_column(String())
    paused_stream_id: Mapped[int | None] = mapped_column(Integer())
    owner: Mapped[str] = mapped_column(String())
    message_type: Mapped[str] = mapped_column(String())

    timer_id: Mapped[int | None] = mapped_column(Integer())
    start_time: Mapped[int | None] = mapped_column(Float())

    step_resets: Mapped[int] = mapped_column(
        Integer(), doc="Number of times the terminal component was reset after this step occured"
    )

    entity_id: Mapped[int] = mapped_column(primary_key=True, init=False)

    type: Mapped[str] = mapped_column(String(), default="TerminalEvaluation")
    expression: Mapped[str] = mapped_column(Text())
    output: Mapped[str] = mapped_column(Text(), default="", doc="The full output of the cell")
    _output_digest: Mapped[
        str | None
    ] = mapped_column(String(), default=None, doc="The output of the cell truncated to 3kB")
    parsed_final_expression_output: Mapped[Any] = mapped_column(JSONB(), default=None)
    _formatted_output: Mapped[
        str
    ] = mapped_column(Text(), default="", doc="Formatted JSON for Jingting for a better graded markdown report")

    _paused_time: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict)

    __table_args__ = (Index("idx_terminal_evaluation_type", "type"),)


class TraceStepClosure(StepClosure):
    @staticmethod
    def _compatible_with(step: AbstractStep) -> bool:
        return isinstance(step, TraceStep)

    # new api
    computer_resets: int

    config: dict[str, Any] | None

    try_number: int
    eval: TerminalEvaluation | None

    # old api
    requested_utc: float
    cell_input: str
    cell_output: str | dict[str, Any] | None
    expected_utc: float | None
    missing_time: float
    silent_time_s: float
    agent_message: AgentMessage | None
    grading_time_s: float
    shutdown_utc: float | None
    system_error: SystemError | None
    raw_terminal: list[dict] | None
    terminal_buffer: list[str] | None
    agent_message_list: list[AgentMessage]
    terminal_messages: list[TraceCompletion]

    index_hierarchy: list[str]
    public_execution_id: UUID | None
    graded_timestamp: float
    graded_output: str | dict[str, Any] | None
    computed_input: str | None

    cls: type[TraceStep]
    task: Any
    parent: Any
    message_log_context: dict[str, Any] | None
    task_trace: TraceAssistantEntity

    def get_cell_input(self: TraceStep) -> str:
        return ""

    @property
    def checked(self: TraceStep) -> bool:
        return False

    def check_and_record(
        self: TraceStep,
        agent_response: TraceCompletion,
        try_number: int,
    ) -> None:
        pass

    def compute_grader_output(self: TraceStep):
        pass

    def compute_parsed_output(self: TraceStep) -> Any:
        pass

    def get_children(self: TraceStep) -> list[TraceStep]:
        return []

    def get_eval(self: TraceStep) -> TerminalEvaluation | None:
        """
        Returns the eval object for this step, which may be None if this step has not yet been
        graded.
        """
        return self.eval

    @functools.lru_cache(1)
    def get_execution_result(self: TraceStep) -> ExecutionResult:
        """
        Returns the execution result for this step. This is the raw output of the code that was
        executed in this step.
        """
        # Eventually we'll get rid of the eval field and this will be just fetches from the
        # database.
        eval = self.get_eval()
        if eval is None:
            raise ValueError("No eval found for step")

        return ExecutionResultDbObject(output=eval.output.rstrip("\n"))


# This closure must be subclassed.
class TaskStep(TraceStep.Traced):
    parent_goal: TraceStep
    _ComputerInterface: type
    UUID: str | None

    def get_cell_input(self) -> str:
        return ""

    def _create_goal_closure(self) -> TraceStepClosure:
        step_closure = TraceStepClosure()
        step_closure.cls = TraceStep
        step_closure.computer_resets = 0
        step_closure.task = self.get_task()
        step_closure.parent = ObjectMapper.unwrap(self)
        step_closure.index_hierarchy = [
            with_concurrency(self.get_index_in_group(), self.get_group().get_concurrency())
        ]
        return step_closure

    def assert_entities_consistent(self) -> None:
        pass

    def get_eval(self) -> TerminalEvaluation:
        pass


ProgressReporter.register_step(TraceStep)
