import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"

def calculateMetrics(actuals, predictions):
    if actuals is None or predictions is None or len(actuals) == 0:
        return {'accuracy': np.nan, 'precisionMacro': np.nan, 'recallMacro': np.nan, 'f1Macro': np.nan, 'f1PerClass': {}}

    labels = sorted(set(actuals))
    f1PerClass = f1_score(actuals, predictions, labels=labels, average=None)

    metrics = { 
        'accuracy': float(accuracy_score(actuals, predictions)), 
        'precisionMacro': float(precision_score(actuals, predictions, average='macro')), 
        'recallMacro': float(recall_score(actuals, predictions, average='macro')), 
        'f1Macro': float(f1_score(actuals, predictions, average='macro')),
        'f1PerClass': {label: float(value) for label, value in zip(labels, f1PerClass)},
    }

    return metrics

def classificationReport(actuals, predictions):
    report = classification_report(actuals, predictions, output_dict=True)
    return pd.DataFrame(report).T.round(4)

def compareModels(metricsByModel):
    rows = {}
    for name, metrics in metricsByModel.items():
        row = {key: value for key, value in metrics.items() if key != 'f1PerClass'}
        row.update({f'f1 {label}': value for label, value in metrics['f1PerClass'].items()})
        rows[name] = row

    return pd.DataFrame(rows).T.round(4).sort_values('f1Macro', ascending=False)

def saveMetrics(name, metrics, metricsDir=METRICS_DIR):
    Path(metricsDir).mkdir(parents=True, exist_ok=True)
    with open(Path(metricsDir) / f'{name}.json', 'w') as file:
        json.dump(metrics, file, indent=2)

def loadMetrics(metricsDir=METRICS_DIR):
    return {path.stem: json.loads(path.read_text()) for path in sorted(Path(metricsDir).glob('*.json'))}
