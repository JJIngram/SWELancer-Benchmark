import os
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
from typing import AsyncGenerator, List, Dict

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

    async def _start_task(self, domain: str, project_id: str, api_key: str) -> str:
        """Starts a task by calling the external API and returns the task ID."""
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{domain}/task/start", 
                json={"projectId": project_id}, 
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["taskId"]

    async def _poll_task_status(self, domain: str, task_id: str, api_key: str) -> List[Dict[str, str]]:
        """Polls the task status every 10 seconds until it is marked as complete with changes."""
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(
                    f"{domain}/task/{task_id}", 
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    status_data = await response.json()
                    if status_data.get("status") == "complete":
                        return status_data.get("changes", [])
                await asyncio.sleep(10)

    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        try:
            async with self._start_computer(task) as computer:
                # 1. Run the task setup
                await task.setup(computer)

                # 2. Load configuration from environment variables
                domain = os.getenv("DOMAIN")
                project_id = os.getenv("PROJECT_ID")
                api_key = os.getenv("API_KEY")

                if not domain:
                    raise ValueError("Missing required environment variable: DOMAIN")
                if not project_id:
                    raise ValueError("Missing required environment variable: PROJECT_ID")
                if not api_key:
                    raise ValueError("Missing required environment variable: API_KEY")

                # 3. Start and poll the task
                task_id = await self._start_task(domain, project_id, api_key)
                changes = await self._poll_task_status(domain, task_id, api_key)

                # 4. Mimic changes from the task to the computer
                for change in changes:
                    filename = change['filename']
                    filecontent = change['filecontent']
                    await computer.upload(filename, filecontent)

                # 5. Grade and yield the final result
                grade = await task.grade(computer)
                yield FinalResultSuccessful(grade=grade)

        except Exception as e:
            print(f"Error: {e}")
            raise
            yield FinalResultSuccessful(
                grade=Grade(score=0, grader_log=f"Grading failed with error: {str(e)}")
            )
