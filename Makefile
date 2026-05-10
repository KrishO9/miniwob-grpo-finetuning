fine-tune:
	uv run modal run -m src.browser_control.fine_tune --config-file-name $(config)

evaluation:
	uv run python -m src.browser_control.evaluate

kaggle-fine-tune:
	PYTHONPATH=src python -m browser_control.fine_tune_kaggle $(config)
