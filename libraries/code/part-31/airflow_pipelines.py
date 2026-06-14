# Apache Airflow DAG examples — Part 31 code examples
# apache-airflow 2.10.x | Python 3.11+
# Install: pip install apache-airflow
# Usage: place in ~/airflow/dags/ then run: airflow standalone

from datetime import datetime, timedelta


# ── 1. Simple ETL pipeline DAG ────────────────────────────────────────────────
# File: dags/simple_etl.py

ETL_DAG = '''
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import json, logging

@dag(
    dag_id="simple_etl_pipeline",
    schedule="@hourly",
    start_date=days_ago(1),
    catchup=False,
    tags=["etl", "demo"],
    default_args={
        "retries": 2,
        "retry_delay": __import__("datetime").timedelta(minutes=5),
        "owner": "ml-team",
    },
)
def simple_etl_pipeline():
    """ETL pipeline: extract → validate → transform → load → notify."""

    @task
    def extract() -> dict:
        """Simulate extracting data from an API."""
        records = [{"id": i, "value": i * 1.5, "category": "A" if i % 2 else "B"}
                   for i in range(100)]
        logging.info(f"Extracted {len(records)} records")
        return {"records": records, "count": len(records)}

    @task
    def validate(data: dict) -> dict:
        """Validate data quality."""
        records = data["records"]
        valid   = [r for r in records if r["value"] > 0]
        logging.info(f"Valid: {len(valid)}/{len(records)}")
        if len(valid) < len(records) * 0.9:
            raise ValueError(f"Too many invalid records: {len(records) - len(valid)}")
        return {"records": valid, "count": len(valid)}

    @task
    def transform(data: dict) -> dict:
        """Apply transformations."""
        records = data["records"]
        transformed = [{
            "id":          r["id"],
            "value_log":   __import__("math").log(r["value"] + 1),
            "category":    r["category"],
            "is_large":    r["value"] > 50,
        } for r in records]
        return {"records": transformed, "count": len(transformed)}

    @task
    def load(data: dict) -> dict:
        """Simulate loading to data warehouse."""
        n_large = sum(1 for r in data["records"] if r["is_large"])
        logging.info(f"Loaded {data['count']} records, {n_large} are large values")
        # In production: conn = get_db_conn(); conn.executemany(...)
        return {"loaded": data["count"], "n_large": n_large}

    @task
    def notify(stats: dict):
        """Send completion notification."""
        msg = f"ETL complete: {stats['loaded']} records, {stats['n_large']} large"
        logging.info(f"NOTIFICATION: {msg}")
        # In production: send Slack message, email, etc.

    # Chain tasks
    raw       = extract()
    validated = validate(raw)
    cleaned   = transform(validated)
    stats     = load(cleaned)
    notify(stats)

dag = simple_etl_pipeline()
'''

# ── 2. ML training pipeline DAG ───────────────────────────────────────────────

ML_PIPELINE_DAG = '''
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import timedelta

@dag(
    dag_id="ml_training_pipeline",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "training"],
)
def ml_training_pipeline():

    @task
    def prepare_data() -> str:
        """Preprocess and save training data."""
        import pandas as pd, numpy as np
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
        df   = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)])
        df["target"] = y
        path = "/tmp/train_data.parquet"
        df.to_parquet(path)
        return path  # pass path via XCom (not the dataframe!)

    @task
    def train_model(data_path: str) -> str:
        """Train model and return model path."""
        import pandas as pd, joblib
        from sklearn.ensemble import RandomForestClassifier

        df    = pd.read_parquet(data_path)
        X, y  = df.drop("target", axis=1).values, df["target"].values
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        model_path = "/tmp/model.pkl"
        joblib.dump(model, model_path)
        return model_path

    @task
    def evaluate_model(model_path: str) -> dict:
        """Evaluate model and return metrics."""
        import joblib, numpy as np
        from sklearn.datasets import make_classification
        from sklearn.metrics import accuracy_score, f1_score

        model   = joblib.load(model_path)
        X_te, y_te = make_classification(n_samples=1000, n_features=20, random_state=99)
        preds   = model.predict(X_te)
        metrics = {
            "accuracy": round(accuracy_score(y_te, preds), 4),
            "f1_score": round(f1_score(y_te, preds), 4),
        }
        print(f"Evaluation metrics: {metrics}")
        return metrics

    @task.branch
    def decide_deployment(metrics: dict) -> str:
        """Branch: deploy if accuracy > threshold."""
        return "deploy_model" if metrics["accuracy"] > 0.85 else "retrain_needed"

    @task
    def deploy_model(model_path: str):
        print(f"DEPLOYING model from {model_path} to production endpoint!")

    @task
    def retrain_needed():
        print("Model accuracy too low — scheduling retraining with more data.")

    data_path  = prepare_data()
    model_path = train_model(data_path)
    metrics    = evaluate_model(model_path)
    branch     = decide_deployment(metrics)
    deploy_model(model_path)
    retrain_needed()

dag = ml_training_pipeline()
'''

# ── 3. Dynamic task mapping ────────────────────────────────────────────────────

DYNAMIC_MAPPING_DAG = '''
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

@dag(
    dag_id="parallel_hyperparameter_search",
    schedule="@weekly",
    start_date=days_ago(1),
    catchup=False,
)
def hp_search():

    @task
    def get_configs() -> list:
        return [
            {"n_estimators": 50,  "max_depth": 3},
            {"n_estimators": 100, "max_depth": 5},
            {"n_estimators": 200, "max_depth": 7},
            {"n_estimators": 100, "max_depth": 10},
        ]

    @task
    def train_config(config: dict) -> dict:
        """Train with ONE config — KFP runs this in parallel for each config."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X, y   = make_classification(n_samples=500, n_features=10, random_state=0)
        model  = RandomForestClassifier(**config, random_state=42)
        scores = cross_val_score(model, X, y, cv=3)
        return {"config": config, "val_accuracy": round(float(scores.mean()), 4)}

    @task
    def pick_best(results: list) -> dict:
        best = max(results, key=lambda r: r["val_accuracy"])
        print(f"Best config: {best}")
        return best

    configs = get_configs()
    results = train_config.expand(config=configs)  # DYNAMIC MAPPING: parallel
    pick_best(results)

dag = hp_search()
'''


def save_dags_to_disk():
    """Optionally save DAG files to Airflow's dags folder."""
    import pathlib
    airflow_dags = pathlib.Path.home() / "airflow" / "dags"
    airflow_dags.mkdir(parents=True, exist_ok=True)

    for name, content in [
        ("simple_etl.py",          ETL_DAG),
        ("ml_pipeline.py",         ML_PIPELINE_DAG),
        ("dynamic_hp_search.py",   DYNAMIC_MAPPING_DAG),
    ]:
        path = airflow_dags / name
        path.write_text(content)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Airflow DAG examples — Part 31")
    print("=" * 60)
    print("\n3 DAGs available:")
    print("  1. simple_etl_pipeline    — ETL with validation")
    print("  2. ml_training_pipeline   — train + evaluate + branch deploy")
    print("  3. parallel_hp_search     — dynamic task mapping (parallel grid search)")
    print()

    save = input("Save DAG files to ~/airflow/dags/? (y/n): ")
    if save.lower() == "y":
        save_dags_to_disk()
        print("\nStart Airflow with: airflow standalone")
        print("Open UI at:        http://localhost:8080")
    else:
        print("\nDAG source code printed below:\n")
        print(ML_PIPELINE_DAG[:1000] + "...")
        print("\nTo run a task manually (without scheduler):")
        print("  airflow tasks test ml_training_pipeline prepare_data 2024-01-01")
