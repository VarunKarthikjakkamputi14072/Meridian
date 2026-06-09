.PHONY: install mlflow train serve drift simulate up down logs test clean

export PYTHONPATH := src

install:
	pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
	pip install -r requirements.txt

# --- local (no docker) targets; run each in its own terminal ---
mlflow:        ## start MLflow tracking + registry on :5500
	mlflow server --host 0.0.0.0 --port 5500 \
		--backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns

train:         ## train baseline + register + promote to production
	python -m meridian.train --reason baseline

serve:         ## FastAPI serving on :8000
	uvicorn meridian.serving.app:app --host 0.0.0.0 --port 8000

drift:         ## Evidently drift monitor + retrain trigger (metrics on :8001)
	python -m meridian.drift.monitor

simulate:      ## drive normal then skewed traffic to trigger drift
	python -m meridian.simulate.traffic --normal 600 --skew 800 --rps 30

# --- full stack ---
up:            ## bring up the whole pipeline in docker
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f drift-monitor serving

test:
	pytest -q

clean:
	rm -rf mlruns mlflow.db data/*.csv data/*.parquet .pytest_cache
