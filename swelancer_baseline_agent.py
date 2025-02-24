from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
)
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    FinalResultSuccessful,
    FinalResultWithException,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from typing_extensions import override
from alcatraz.clusters.local import LocalConfig

import asyncio
import functools
import os
import re
import subprocess
import threading
import traceback
from contextlib import AsyncExitStack, contextmanager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, AsyncGenerator, ContextManager, Generator, Generic, TypeVar, cast

from contextvars import ContextVar

from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface


class SwelancerBaselineAgent(PythonCodingSolver):
    name: str = "swelancer_baseline_agent"
    model: str = "swelancer_baseline"

    @override
    def shortname(self) -> str:
        return self.name

    @asynccontextmanager
    async def _start_computer(self, task: ComputerTask) -> AsyncGenerator[ComputerInterface, None]:
        # replace with LocalCluster semantics

        alcatraz_env = task_to_alcatraz_config(task, LocalConfig(pull_from_registry=False))

        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        try:
            async with self._start_computer(task) as computer:
                print(computer)
                # 1. Run the task setup

                # 2. Query the API / some agent

                # 3. Grade and yield the final result

                grade = Grade(score=0)
                yield FinalResultSuccessful(grade=grade)
        except Exception as e:
            print(f"Error: {e}")
            raise
            yield FinalResultSuccessful(
                grade=Grade(score=0, grader_log=f"Grading failed with error: {str(e)}")
            )
