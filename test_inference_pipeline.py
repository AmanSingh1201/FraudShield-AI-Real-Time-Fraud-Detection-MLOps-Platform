import pandas as pd
import pytest

from src.fraudshield.models.inference_pipeline import FraudInferencePipeline


@pytest.fixture(scope="module")
def pipeline():
    return FraudInferencePipeline()


@pytest.fixture(scope="module")
def test_rows():
    df = pd.read_csv("data/processed/test.csv")
    fraud = df[df.Class == 1].iloc[0].drop("Class").to_dict()
    normal = df[df.Class == 0].iloc[0].drop("Class").to_dict()
    return fraud, normal


def test_pipeline_loads(pipeline):
    assert pipeline.model is not None
    assert pipeline.threshold > 0


def test_predict_known_fraud_flags_high_risk(pipeline, test_rows):
    fraud, _ = test_rows
    result = pipeline.predict_one(fraud)
    assert result["fraud_probability"] > pipeline.threshold
    assert result["decision"] in ("BLOCK", "REVIEW")


def test_predict_known_normal_is_low_probability(pipeline, test_rows):
    _, normal = test_rows
    result = pipeline.predict_one(normal)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_missing_field_raises_value_error(pipeline, test_rows):
    _, normal = test_rows
    bad = dict(normal)
    del bad["V14"]
    with pytest.raises(ValueError):
        pipeline.predict_one(bad)


def test_batch_predict_matches_single_predict(pipeline, test_rows):
    fraud, normal = test_rows
    batch = pipeline.predict_batch([fraud, normal])
    assert len(batch) == 2
    assert batch[0]["fraud_probability"] == pipeline.predict_one(fraud)["fraud_probability"]
