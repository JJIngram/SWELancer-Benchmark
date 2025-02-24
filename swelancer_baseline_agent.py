import aiohttp
import asyncio
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
)
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    FinalResultSuccessful,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from alcatraz.clusters.local import LocalConfig
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface


class SwelancerBaselineAgent(PythonCodingSolver):
    name: str = "swelancer_baseline_agent"
    model: str = "swelancer_baseline"

    def shortname(self) -> str:
        return self.name

    @asynccontextmanager
    async def _start_computer(self, task: ComputerTask) -> AsyncGenerator[ComputerInterface, None]:
        alcatraz_env = task_to_alcatraz_config(task, LocalConfig(pull_from_registry=False))
        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    async def _start_task(self, domain: str, project_id: str) -> str:
        """Starts a task by calling the external API and returns the task ID."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{domain}/task/start", json={"projectId": project_id}) as response:
                response.raise_for_status()
                result = await response.json()
                return result["taskId"]

    async def _poll_task_status(self, domain: str, task_id: str) -> None:
        """Polls the task status every 10 seconds until it is marked as complete."""
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(f"{domain}/task/{task_id}") as response:
                    response.raise_for_status()
                    status_data = await response.json()
                    if status_data.get("status") == "complete":
                        break
                await asyncio.sleep(10)

    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        try:
            async with self._start_computer(task) as computer:
                # 1. Run the task setup
                await task.setup(computer)

                # 2. Start and poll the task
                domain = "https://example.com"  # Replace with actual domain
                project_id = "some_project_id"  # Replace with actual project ID retrieval
                task_id = await self._start_task(domain, project_id)
                await self._poll_task_status(domain, task_id)

                # 3. Grade and yield the final result
                grade = await task.grade(computer)
                yield FinalResultSuccessful(grade=grade)

        except Exception as e:
            print(f"Error: {e}")
            raise
            yield FinalResultSuccessful(
                grade=Grade(score=0, grader_log=f"Grading failed with error: {str(e)}")
            )
