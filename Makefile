setup_preprocess:
    UV_PROJECT_ENVIRONMENT=.venv_preprocess uv sync


setup:
    uv sync

clean:
	./scripts/clean.sh
