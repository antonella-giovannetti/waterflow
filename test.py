from mlflow.tracking import MlflowClient

client = MlflowClient()
experiments = client.search_experiments(view_type="ALL")

print("📋 Liste complète des expériences enregistrées (y compris supprimées) :\n")
for exp in experiments:
    print(f"Name: {exp.name} | ID: {exp.experiment_id} | Status: {exp.lifecycle_stage}")
