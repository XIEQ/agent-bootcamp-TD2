from src.utils.tools.code_interpreter import (
    _enumerate_files,
    _upload_files,
    CodeInterpreterOutput,
    _CodeInterpreterOutputError,
)
from typing import *
from pathlib import Path
import os
from e2b_code_interpreter import AsyncSandbox


class CodeInterpreterOptimized:
    """Code Interpreter tool for the agent."""

    def __init__(
        self,
        local_files: "Sequence[Path | str]| None" = None,
        timeout_seconds: int = 30,
        template_name: str | None = None,
    ):
        """Configure your Code Interpreter session.

        Note that the sandbox is not persistent, and each run_code will
        execute in a fresh sandbox! (e.g., variables need to be re-declared each time.)

        Parameters
        ----------
            local_files : list[pathlib.Path | str] | None
                Optionally, specify a list of local files (as paths)
                to upload to sandbox working directory. Folders will be flattened.
            timeout_seconds : int
                Limit executions to this duration.
            template_name : str | None
                Optionally, override the default e2b template name.
                See e2b_template.md for details.
        """
        self.timeout_seconds = timeout_seconds
        self.local_files = []
        self.template_name = template_name

        # Recursively find files if the given path is a folder.
        if local_files:
            for _path in local_files:
                self.local_files.extend(_enumerate_files(_path))
        self.template_name = template_name

    async def _async_init(self):
        self.sbx = await AsyncSandbox.create(
            timeout=self.timeout_seconds, template=self.template_name
        )

        await _upload_files(self.sbx, self.local_files)

    @classmethod
    async def create(cls, **kwargs):
        instance = cls(**kwargs)
        await instance._async_init()
        return instance

    async def sandbox_close(self):
        if self.sbx:
            await self.sbx.kill()

    async def run_code(self, code: str) -> str:
        """Run the given Python code in a sandbox environment.

        Parameters
        ----------
            code : str
                Python logic to execute.
        """
        sbx = self.sbx

        result = await sbx.run_code(code, on_error=lambda error: print(error.traceback))
        response = CodeInterpreterOutput.model_validate_json(result.logs.to_json())

        error = result.error
        if error is not None:
            response.error = _CodeInterpreterOutputError.model_validate_json(
                error.to_json()
            )

        return response.model_dump_json()
        # finally:
        #     await sbx.kill()


def read_file_as_string(filepath: str) -> str:
    """
    Reads the entire content of a file into a single string.

    Args:
        filepath: The path to the Python script file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an error occurs during file reading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file was not found at path: {filepath}")

    # The 'with open(...)' structure ensures the file handle is closed
    # automatically, even if exceptions occur.
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        return file_content
    except IOError as e:
        # Re-raise the IO error for external handling
        raise IOError(f"An error occurred while reading the file: {e}")


class FinanceDataCodeInterpreter:
    def __init__(
        self,
        init_module_path: str,
        local_files=None,
        timeout_seconds: int = 30,
        template_name: str | None = None,
    ):
        self.local_files = local_files
        self.timeout_seconds = timeout_seconds
        self.template_name = template_name or "0v90rfl2s90xby53zujh"
        self.init_module_path = init_module_path

    async def _async_init(self):
        self.cio = await CodeInterpreterOptimized.create(
            local_files=self.local_files,
            timeout_seconds=self.timeout_seconds,
            template_name=self.template_name,
        )

        await self.cio.run_code(read_file_as_string(self.init_module_path))

    @classmethod
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._async_init()
        return instance

    async def run_code(self, code):
        return await self.cio.run_code(code)

    async def run_query(self, sql_command):
        # sql_command = 'select count(*) as count_cid from users_data'
        code = f'query("{sql_command}")'
        print(f"C:======== {sql_command} =========== ")
        return await self.cio.run_code(code)

    async def sandbox_close(self):
        await self.cio.sandbox_close()
