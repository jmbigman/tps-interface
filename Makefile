# Always run these commands, even if files of the same name exist
.PHONY: lint check

lint:
	python -m flake8 scripts tps_interface test --extend-ignore=F722

check:
	python -m pytest -v test
