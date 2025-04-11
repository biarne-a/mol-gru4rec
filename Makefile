setup_preprocess:
	mv pyproject.toml _pyproject.toml
	mv pyproject_preprocess.toml pyproject.toml
	UV_PROJECT_ENVIRONMENT=.venv_preprocess uv sync
	mv pyproject.toml pyproject_preprocess.toml
	mv _pyproject.toml pyproject.toml

setup:
	uv sync

clean:
	./scripts/clean.sh
