"""Unit tests for GPT2 classifier ETL data extraction with synthetic datasets."""

from pathlib import Path

from tgedr_lm.classifier.gpt2.etl import Gpt2ClassifierEtl


class _FakeTrainDataset:
    """Synthetic dataset stub exposing train_test_split."""

    def train_test_split(self, test_size: float = 0.2):
        _ = test_size
        return {
            "train": {
                "sentence": [
                    "profit increased",
                    "market is stable",
                    "shares dropped",
                    "revenue improved",
                ],
                "label": [2, 1, 0, 2],
            },
            "test": {
                "sentence": ["guidance lowered", "earnings beat"],
                "label": [0, 2],
            },
        }


def test_fetch_data_uses_mocked_hf_dataset_and_caches(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_load_dataset(dataset: str, split: str = "train"):
        calls["count"] += 1
        assert dataset == "synthetic/financial"
        assert split == "train"
        return _FakeTrainDataset()

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.load_dataset", _fake_load_dataset)

    etl = Gpt2ClassifierEtl(config={})
    etl._fetch_data("synthetic/financial")
    etl._fetch_data("synthetic/financial")

    assert calls["count"] == 1
    assert etl._data["train"]["texts"][0] == "profit increased"
    assert etl._data["train"]["labels"] == [2, 1, 0, 2]
    assert etl._data["test"]["texts"] == ["guidance lowered", "earnings beat"]
    assert etl._data["test"]["labels"] == [0, 2]


def test_get_datasets_respects_fraction_with_synthetic_data() -> None:
    etl = Gpt2ClassifierEtl(config={})
    etl._data = {
        "train": {
            "texts": ["a", "b", "c", "d"],
            "labels": [0, 1, 0, 2],
        },
        "test": {
            "texts": ["x", "y"],
            "labels": [2, 1],
        },
    }

    train_dataset, val_dataset = etl.get_datasets(fraction=0.5)

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
    assert train_dataset[0]["labels"] == 0
    assert val_dataset[0]["labels"] == 2


def test_extract_uses_configuration_injected_dataset(monkeypatch) -> None:
    captured = {}

    def _fake_fetch_data(dataset: str) -> None:
        captured["dataset"] = dataset

    etl = Gpt2ClassifierEtl(config={"dataset": "synthetic/from-config"})
    monkeypatch.setattr(etl, "_fetch_data", _fake_fetch_data)

    etl.extract()

    assert captured["dataset"] == "synthetic/from-config"


def test_extract_loads_pretrained_weights_when_url_is_provided(monkeypatch) -> None:
    captured = {}

    def _fake_fetch_data(dataset: str) -> None:
        captured["dataset"] = dataset

    def _fake_load_pickle_compressed(weights_url: str):
        captured["weights_url"] = weights_url
        return {"weights": "loaded"}

    etl = Gpt2ClassifierEtl(
        config={
            "dataset": "synthetic/from-config",
            "weights_url": "/tmp/pretrained-weights.pickle.gz",
        }
    )
    monkeypatch.setattr(etl, "_fetch_data", _fake_fetch_data)
    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.load_pickle_compressed", _fake_load_pickle_compressed)

    etl.extract()

    assert captured["dataset"] == "synthetic/from-config"
    assert captured["weights_url"] == "/tmp/pretrained-weights.pickle.gz"
    assert etl._weights == {"weights": "loaded"}


def test_search_hyperparameters_uses_synthetic_datasets_and_default_trials(monkeypatch) -> None:
    etl = Gpt2ClassifierEtl(config={})
    train_dataset, val_dataset = object(), object()
    captured = {}

    monkeypatch.setattr(etl, "get_datasets", lambda fraction=1.0: (train_dataset, val_dataset))
    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.ClassifierBaseConfiguration", lambda n_classes=3: {"n": n_classes})

    class _FakeModel:

        @staticmethod
        def compute_metrics(eval_pred):
            _ = eval_pred
            return {"accuracy": 1.0}

        def __init__(self, cfg) -> None:
            captured["model_cfg"] = cfg

    class _FakeHyperParamSearch:

        def __init__(self, compute_metrics, train_args=None):
            captured["compute_metrics"] = compute_metrics
            captured["train_args"] = train_args

        def search(self, model, train_dataset, val_dataset, trials=8):
            captured["model"] = model
            captured["train_dataset"] = train_dataset
            captured["val_dataset"] = val_dataset
            captured["trials"] = trials
            return {"learning_rate": 1e-4}

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.GPT2Classifier", _FakeModel)
    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.HyperParamSearch", _FakeHyperParamSearch)

    result = etl._search_hyperparameters()

    assert result == {"learning_rate": 1e-4}
    assert captured["train_dataset"] is train_dataset
    assert captured["val_dataset"] is val_dataset
    assert captured["trials"] == 3


def test_get_model_pretrains_when_weights_are_available(monkeypatch) -> None:
    etl = Gpt2ClassifierEtl(config={})
    etl._weights = {"weights": "loaded"}
    captured = {}

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.ClassifierBaseConfiguration", lambda n_classes=3: {"n": n_classes})

    class _FakeModel:

        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        def pretrain(self, weights) -> None:
            captured["pretrained_weights"] = weights

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.GPT2Classifier", _FakeModel)

    model = etl._get_model()

    assert model is not None
    assert captured["cfg"] == {"n": 3}
    assert captured["pretrained_weights"] == {"weights": "loaded"}


def test_transform_trains_with_mocked_components(monkeypatch, tmp_path: Path) -> None:
    etl = Gpt2ClassifierEtl(config={})
    train_dataset, val_dataset = object(), object()
    captured = {}

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        etl,
        "_search_hyperparameters",
        lambda: {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 4,
            "warmup_steps": 0.05,
        },
    )
    monkeypatch.setattr(etl, "get_datasets", lambda fraction=1.0: (train_dataset, val_dataset))
    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.ClassifierBaseConfiguration", lambda n_classes=3: {"n": n_classes})

    class _FakeModel:

        @staticmethod
        def compute_metrics(eval_pred):
            _ = eval_pred
            return {"accuracy": 1.0}

        def __init__(self, cfg) -> None:
            captured["model_cfg"] = cfg

    class _FakeTrainer:

        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            captured["trainer_model"] = model
            captured["trainer_args"] = args
            captured["trainer_train_dataset"] = train_dataset
            captured["trainer_eval_dataset"] = eval_dataset
            captured["trainer_compute_metrics"] = compute_metrics
            captured["train_called"] = False
            captured["evaluate_called"] = False

        def train(self):
            captured["train_called"] = True

        def evaluate(self):
            captured["evaluate_called"] = True
            return {"eval_accuracy": 1.0, "eval_loss": 0.0}

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.GPT2Classifier", _FakeModel)
    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.Trainer", _FakeTrainer)

    etl.transform()

    assert captured["trainer_train_dataset"] is train_dataset
    assert captured["trainer_eval_dataset"] is val_dataset
    assert captured["train_called"] is True
    assert captured["evaluate_called"] is True
    assert etl._trainer is not None


def test_load_pushes_trainer_to_hub() -> None:
    etl = Gpt2ClassifierEtl(config={})
    captured = {}

    class _FakeTrainer:

        def push_to_hub(self, commit_message: str) -> None:
            captured["commit_message"] = commit_message

    etl._trainer = _FakeTrainer()
    etl.load()

    assert captured["commit_message"] == "Training completed!"


def test_load_copies_weights_file_before_push(monkeypatch, tmp_path: Path) -> None:
    etl = Gpt2ClassifierEtl(config={})
    captured = {}

    source_weights = tmp_path / "params.pickle.gz"
    source_weights.write_bytes(b"weights")

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.OUTPUT_DIR", tmp_path / "out")

    def _fake_copy2(src, dst):
        captured["copy_src"] = src
        captured["copy_dst"] = dst

    class _FakeTrainer:

        def push_to_hub(self, commit_message: str) -> None:
            captured["commit_message"] = commit_message

    monkeypatch.setattr("tgedr_lm.classifier.gpt2.etl.shutil.copy2", _fake_copy2)
    etl._weights_url = source_weights
    etl._trainer = _FakeTrainer()

    etl.load()

    assert captured["copy_src"] == source_weights
    assert captured["copy_dst"] == (tmp_path / "out" / "params.pickle.gz")
    assert captured["commit_message"] == "Training completed!"
