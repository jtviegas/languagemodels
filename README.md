# tgedr-languagemodels

![Coverage](./coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/tgedr-languagemodels)](https://pypi.org/project/tgedr-languagemodels/)



## development
- main requirements:
  - _uv_  
  - _bash_
- Clone the repository like this:

  ``` bash
  git clone git@github.com:jtviegas/languagemodels
  ```
- cd into the folder: `cd languagemodels`
- install requirements: `./helper.sh reqs`

## Optuna quickstart

Optuna is a hyperparameter optimization library. It runs multiple training trials,
tests different parameter combinations, and keeps the ones that improve your target metric.

In this project, it can be used with Hugging Face Trainer to search values like:
- learning rate
- weight decay
- number of epochs
- batch size
- warmup ratio

### Why use it
- avoids manual trial-and-error tuning
- finds stronger parameter combinations faster
- optimizes directly for your metric (for example, eval_accuracy)

### Minimal Trainer integration

```python
from transformers import Trainer

trainer = Trainer(
  model_init=model_init,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=val_dataset,
  compute_metrics=compute_metrics,
)

best_run = trainer.hyperparameter_search(
  backend="optuna",
  direction="maximize",
  n_trials=8,
  hp_space=lambda trial: {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
    "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
  },
  compute_objective=lambda metrics: metrics["eval_accuracy"],
)

print(best_run.hyperparameters)
```

### Practical workflow
1. Run a small search first (for example, 8 to 20 trials).
2. Apply best hyperparameters to final TrainingArguments.
3. Retrain once on your full setup and keep the best checkpoint.

### Recommended search spaces by budget

| Budget | Trials | learning_rate | weight_decay | num_train_epochs | per_device_train_batch_size | warmup_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Small (quick check) | 8-12 | 1e-5 to 5e-4 (log) | 0.0 to 0.1 | 2 to 6 | [4, 8] | 0.0 to 0.15 |
| Medium (balanced) | 20-40 | 5e-6 to 7e-4 (log) | 0.0 to 0.2 | 2 to 10 | [4, 8, 16] | 0.0 to 0.2 |
| Large (thorough) | 60-120 | 1e-6 to 1e-3 (log) | 0.0 to 0.3 | 2 to 16 | [4, 8, 16, 32] | 0.0 to 0.3 |

Notes:
- Keep eval metric and objective aligned (for example, maximize eval_accuracy).
- For small datasets, use fewer epochs in search and retrain best params on full epochs.
- If GPU memory is tight, cap batch-size candidates and use gradient accumulation.

### Copy-paste hp_space presets

```python
def hp_space_small(trial):
  return {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
    "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
  }


def hp_space_medium(trial):
  return {
    "learning_rate": trial.suggest_float("learning_rate", 5e-6, 7e-4, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
    "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
    "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
  }


def hp_space_large(trial):
  return {
    "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 16),
    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
    "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
  }
```


Use one preset in Trainer search:

```python
best_run = trainer.hyperparameter_search(
  backend="optuna",
  direction="maximize",
  n_trials=20,
  hp_space=hp_space_medium,
  compute_objective=lambda metrics: metrics["eval_accuracy"],
)
```
