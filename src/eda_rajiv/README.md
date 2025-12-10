# Run SQLite Code Interpreter

## Understanding Code Interpreter
```shell
# In src/eda_rajiv/trial_sandbox.py, see main0. uncomment ` asyncio.run(main0())` if commented
python ./src/eda_rajiv/trial_sandbox.py
# then check main(), uncomment `asyncio.run(main())` if commented and run
python ./src/eda_rajiv/trial_sandbox.py
```

# Run SQLite Interpreter App

```shell
uv run --env-file .env gradio src/2_frameworks/3_code_interpreter/app_sqlite.py
```