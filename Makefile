PYTHON ?= python
CPU_TESTS ?= tests/cpu

.PHONY: check-cpu lint-cpu test-cpu

# The bootstrap lint lane checks fatal errors without reformatting legacy code.
# Use the existing pre-commit hooks for broader, explicitly requested style work.
lint-cpu:
	git diff --check
	$(PYTHON) -m ruff check --select E9,F63,F7,F82 scripts/validate/check_alignment_report.py \
		scripts/validate/reference_identity.py tests/cpu

test-cpu:
	timeout --signal=TERM --kill-after=10s 600s env \
		CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		TOKENIZERS_PARALLELISM=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
		$(PYTHON) -m pytest -p pytest_timeout -p pytest_socket \
		-o addopts='--strict-markers --strict-config --import-mode=importlib' \
		--disable-socket --timeout=60 --timeout-method=signal -q $(CPU_TESTS)

check-cpu: lint-cpu
	$(MAKE) test-cpu PYTHON="$(PYTHON)" CPU_TESTS=tests/cpu
