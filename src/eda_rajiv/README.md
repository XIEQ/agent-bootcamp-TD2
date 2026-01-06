# Run SQLite Code Interpreter

## Understanding Code Interpreter
```shell
# In src/eda_rajiv/trial_sandbox.py, see main0. uncomment ` asyncio.run(main0())` if commented
python ./src/eda_rajiv/trial_sandbox.py
# then check main(), uncomment `asyncio.run(main())` if commented and run
python ./src/eda_rajiv/trial_sandbox.py
```

# Run SQLite Interpreter App
## Single Agent
```shell
uv run --env-file .env gradio src/2_frameworks/3_code_interpreter/app_sqlite.py
```

## Multi Agent
```shell
uv run --env-file .env gradio src/2_frameworks/3_code_interpreter/app_sqlite_multiagent.py"

```

## Langchain V1 Notebook
```shell
cd ./src/eda_rajiv
uv sync # should create a .venv folder named rajiv-agent-bootcamp-202507
source .venv/bin/activate
python -m ipykernel install --user --name=.venv_rajiv
PYTHONPATH=../../ jupyter notebook
# open trial_langchain_v1.ipynb
```