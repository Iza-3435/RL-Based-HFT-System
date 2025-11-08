"""Tests for ML prediction models."""

import pytest
import numpy as np
import torch

from src.ml.models.latency_models import LSTMLatencyModel, GRULatencyModel, TransformerLatencyModel
from src.ml.training.model_trainer import LatencyModelTrainer
from src.ml.inference.inference_engine import InferenceEngine
from src.ml.features.feature_engineering import LatencyFeatureExtractor


class TestLatencyModels:
    """Test ML latency models."""

    def test_lstm_model_forward(self, sample_features):
        """Test LSTM forward pass."""
        model = LSTMLatencyModel(input_size=45)
        model.eval()

        with torch.no_grad():
            features = torch.from_numpy(sample_features).unsqueeze(0)
            output = model(features)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()
        assert output.item() > 0

    def test_gru_model_forward(self, sample_features):
        """Test GRU forward pass."""
        model = GRULatencyModel(input_size=45)
        model.eval()

        with torch.no_grad():
            features = torch.from_numpy(sample_features).unsqueeze(0)
            output = model(features)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

    def test_transformer_model_forward(self, sample_features):
        """Test Transformer forward pass."""
        model = TransformerLatencyModel(input_size=45)
        model.eval()

        with torch.no_grad():
            features = torch.from_numpy(sample_features).unsqueeze(0)
            output = model(features)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()


class TestModelTrainer:
    """Test model training."""

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        trainer = LatencyModelTrainer(feature_size=45)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_training_reduces_loss(self):
        """Test that training reduces loss."""
        trainer = LatencyModelTrainer(feature_size=45)

        features = np.random.randn(100, 45).astype(np.float32)
        targets = np.random.uniform(500, 1500, 100).astype(np.float32)

        initial_loss = trainer.train(features, targets, epochs=1)
        final_loss = trainer.train(features, targets, epochs=5)

        assert final_loss < initial_loss * 1.5


class TestInferenceEngine:
    """Test inference engine."""

    def test_inference_speed(self, sample_features):
        """Test inference is fast (<1ms)."""
        trainer = LatencyModelTrainer(feature_size=45)
        engine = InferenceEngine(trainer.model, trainer.device)

        latency, confidence, pred_time = engine.predict("NYSE", sample_features)

        assert pred_time < 1.0  # Less than 1ms
        assert 0 <= confidence <= 1.0
        assert latency > 0

    def test_batch_inference(self):
        """Test batch inference."""
        trainer = LatencyModelTrainer(feature_size=45)
        engine = InferenceEngine(trainer.model, trainer.device)

        features = np.random.randn(10, 45).astype(np.float32)

        for i in range(10):
            latency, confidence, pred_time = engine.predict("NYSE", features[i])
            assert latency > 0
            assert 0 <= confidence <= 1.0


class TestFeatureExtractor:
    """Test feature extraction."""

    def test_feature_extraction_shape(self):
        """Test feature vector has correct shape."""
        extractor = LatencyFeatureExtractor(feature_size=45)

        tick_data = {'timestamp': 1234567890, 'price': 150.0}
        network_data = {'latency_us': 850, 'jitter_us': 50}
        order_book = {'bid_depth': 1000, 'ask_depth': 1200}
        market_features = {'volatility': 0.02, 'volume': 100000}

        features = extractor.extract(tick_data, network_data, order_book, market_features)

        assert features.shape == (45,)
        assert not np.isnan(features).any()

    def test_temporal_features(self):
        """Test temporal feature extraction."""
        extractor = LatencyFeatureExtractor(feature_size=45)

        tick_data = {
            'timestamp': 1234567890,
            'price': 150.0,
            'hour': 10,
            'minute': 30
        }

        temporal = extractor._extract_temporal(tick_data)

        assert len(temporal) > 0
        assert all(isinstance(f, (int, float)) for f in temporal)
