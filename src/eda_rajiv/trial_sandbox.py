from src.eda_rajiv.finance_data_code_interpreter import CodeInterpreterOptimized
from src.eda_rajiv.finance_data_code_interpreter import FinanceDataCodeInterpreter
from pathlib import Path
from src import eda_rajiv
import asyncio


async def main0():
    ci = await CodeInterpreterOptimized.create(
        template_name="0v90rfl2s90xby53zujh", timeout_seconds=300
    )
    result = await ci.run_code("x = 10")
    print("R1")
    print(result)
    result = await ci.run_code("print(x + 10)")
    print("R2")
    print(result)

    await ci.sandbox_close()


async def main():

    init_module_path = Path(eda_rajiv.__file__).parent / "sql.py"
    ci = await FinanceDataCodeInterpreter.create(
                        init_module_path = init_module_path,
                        template_name="0v90rfl2s90xby53zujh",
                                               timeout_seconds=30)
    # Use case 1
    result = await ci.run_query('select count(*) as count_cid from users_data')

    # Use Case 2
    # pragma_command = "PRAGMA table_info(users_data);"
    # result = await ci.run_query(pragma_command)

    print("Result")
    print(result)
    await ci.sandbox_close()


asyncio.run(main0())
# asyncio.run(main()) # comment above line and then uncomment this line


