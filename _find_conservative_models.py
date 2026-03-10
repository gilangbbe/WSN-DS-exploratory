"""Find Conservative SMOTE model artifacts in MLflow."""
import mlflow
import os
import glob

mlflow.set_tracking_uri('mlruns')
client = mlflow.tracking.MlflowClient()
exp_id = '410992055011183175'

target_models = ['Extra_Trees', 'Random_Forest', 'Gradient_Boosting', 'HistGradient_Boosting']
runs = client.search_runs(exp_id, filter_string="tags.oversampling = 'Conservative_SMOTE'")

print("=== Conservative SMOTE Runs ===")
for run in runs:
    model_name = run.data.tags.get('model_name', 'unknown')
    if model_name in target_models:
        acc = run.data.metrics.get('accuracy', 0)
        f1m = run.data.metrics.get('f1_macro', 0)
        f1w = run.data.metrics.get('f1_weighted', 0)
        print(f'{model_name}: acc={acc:.4f} f1m={f1m:.4f} f1w={f1w:.4f} run_id={run.info.run_id}')
        
        # List artifacts
        artifacts = client.list_artifacts(run.info.run_id)
        for a in artifacts:
            print(f'  artifact: {a.path}')

# Check model registry for Conservative SMOTE
print("\n=== Model Registry (Conservative SMOTE) ===")
for rm in client.search_registered_models():
    if 'Conservative' in rm.name:
        print(f'  Model: {rm.name}')
        for v in rm.latest_versions:
            print(f'    v{v.version}: source={v.source} run_id={v.run_id}')

# Also check for PKL files in models/ directory
print("\n=== PKL files in models/ ===")
for f in glob.glob('models/*Conservative*'):
    print(f'  {f}')
for f in glob.glob('models/*conservative*'):
    print(f'  {f}')

# Check for pkl files in mlruns model registry
print("\n=== Model Registry PKL ===")
for root, dirs, files in os.walk('mlruns/410992055011183175/models'):
    for f in files:
        if f.endswith('.pkl'):
            full_path = os.path.join(root, f)
            print(f'  {full_path}')
