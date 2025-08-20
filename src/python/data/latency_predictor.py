import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import json
import pickle
from collections import deque
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ModelType(Enum):

    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    
@dataclass
class LatencyPrediction:

    venue: str
    timestamp: float
    predicted_latency_us: float
    confidence: float
    prediction_time_ms: float
    model_version: str
    features_used: Dict[str, float]
class LatencyDataset(Dataset):


    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 sequence_length: int = 50):
        self.sequence_length = sequence_length

        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        self.features = torch.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=-1.0)
        self.targets = torch.nan_to_num(self.targets, nan=1000.0, posinf=10000.0, neginf=50.0)

        self.targets = torch.clamp(self.targets, min=50.0, max=50000.0)

        self.feature_mean = self.features.mean(dim=0)
        self.feature_std = self.features.std(dim=0) + 1e-6

        mask = self.feature_std > 1e-5
        self.features[:, mask] = (self.features[:, mask] - self.feature_mean[mask]) / self.feature_std[mask]

        self.targets = torch.log(self.targets)
        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std() + 1e-6
        self.targets = (self.targets - self.target_mean) / self.target_std

        logger.info(f"Dataset created: {len(self.features)} samples, "
                   f"feature range: [{self.features.min():.3f}, {self.features.max():.3f}], "
                   f"target range: [{self.targets.min():.3f}, {self.targets.max():.3f}]")

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx):
        seq_features = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        return seq_features, target

    def denormalize_prediction(self, pred: torch.Tensor) -> torch.Tensor:

        pred = pred * self.target_std + self.target_mean
        pred = torch.exp(pred)
        pred = torch.clamp(pred, min=50.0, max=50000.0)
        return pred
    
class LSTMLatencyModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self._init_lstm_weights()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_lstm_weights(self):

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def _init_weights(self):

        for module in [self.attention, self.fc, self.confidence_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.input_norm(x)

        lstm_out, (hidden, cell) = self.lstm(x)

        if torch.isnan(lstm_out).any():
            logger.warning("NaN detected in LSTM output, using zeros")
            lstm_out = torch.zeros_like(lstm_out)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        prediction = self.fc(context)
        confidence = self.confidence_head(context)

        prediction = torch.nan_to_num(prediction, nan=0.0)
        confidence = torch.nan_to_num(confidence, nan=0.5)

        return prediction, confidence
    
class GRULatencyModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        for module in [self.fc, self.confidence_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.input_norm(x)

        gru_out, hidden = self.gru(x)

        if torch.isnan(gru_out).any():
            gru_out = torch.zeros_like(gru_out)

        last_hidden = hidden[-1]

        prediction = self.fc(last_hidden)
        confidence = self.confidence_head(last_hidden)

        prediction = torch.nan_to_num(prediction, nan=0.0)
        confidence = torch.nan_to_num(confidence, nan=0.5)

        return prediction, confidence
    
class LatencyPredictor:

    def __init__(self, venues: List[str], feature_size: int = 45,
                 sequence_length: int = 50, model_type: ModelType = ModelType.LSTM):
        self.venues = venues
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        self.model_type = model_type

        self.models: Dict[str, nn.Module] = {}
        self.model_versions: Dict[str, str] = {}
        self.best_models: Dict[str, Dict[str, Any]] = {}

        self.feature_buffers: Dict[str, deque] = {
            venue: deque(maxlen=sequence_length) for venue in venues
        }

        self.prediction_history: Dict[str, deque] = {
            venue: deque(maxlen=1000) for venue in venues
        }
        self.model_performance: Dict[str, Dict[str, float]] = {
            venue: {'mae': float('inf'), 'accuracy': 0, 'predictions': 0} for venue in venues
        }

        self.online_learning_enabled = False
        self.update_buffer: Dict[str, List[Tuple[np.ndarray, float]]] = {
            venue: [] for venue in venues
        }
        self.update_threshold = 100

        self._initialize_models()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LatencyPredictor initialized on {self.device}")

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _initialize_models(self):

        for venue in self.venues:
            if self.model_type == ModelType.LSTM:
                model = LSTMLatencyModel(self.feature_size)
            elif self.model_type == ModelType.GRU:
                model = GRULatencyModel(self.feature_size)
            else:
                model = LSTMLatencyModel(self.feature_size)

            self.models[venue] = model
            self.model_versions[venue] = f"v1.0_{int(time.time())}"

            if hasattr(self, 'device'):
                self.models[venue] = self.models[venue].to(self.device)
            self.models[venue].eval()

            logger.info(f"Initialized {self.model_type.value} model for {venue}")
            
    def set_fast_mode(self, enabled: bool = True):
        self._fast_mode = enabled
        if enabled:
           self.sequence_length = 10
           self.update_threshold = 10
        for venue in self.venues:
            self.feature_buffers[venue] = deque(maxlen=self.sequence_length)
            logger.info("Fast mode enabled - reduced sequence length to 10")

    def extract_features(self, tick_data: Dict, network_data: Dict,
                        order_book_data: Dict, market_features: Dict) -> np.ndarray:

        features = []

        timestamp = tick_data.get('timestamp', time.time())
        dt = datetime.fromtimestamp(timestamp)

        features.extend([
            dt.hour + dt.minute / 60.0,
            dt.weekday() / 6.0,
            dt.microsecond / 1e6,
            np.sin(2 * np.pi * dt.hour / 24),
            np.cos(2 * np.pi * dt.hour / 24),
        ])

        features.extend([
            min(network_data.get('latency_us', 1000), 10000) / 10000.0,
            min(network_data.get('jitter_us', 100), 1000) / 1000.0,
            min(network_data.get('packet_loss_rate', 0), 1.0),
            min(network_data.get('congestion_score', 0.5), 1.0),
            min(network_data.get('bandwidth_utilization', 0.5), 1.0),
        ])

        mid_price = tick_data.get('mid_price', 100)
        volume = max(tick_data.get('volume', 1000), 1)
        bid_price = tick_data.get('bid_price', mid_price - 0.01)
        ask_price = tick_data.get('ask_price', mid_price + 0.01)

        features.extend([
            mid_price / 1000.0,
            np.log(volume) / 10.0,
            max(ask_price - bid_price, 0.001) / mid_price,
            min(tick_data.get('trade_intensity', 0.5), 10.0) / 10.0,
            min(tick_data.get('volatility', 0.01), 0.1) / 0.1,
        ])

        features.extend([
            min(order_book_data.get('bid_depth', 10000), 1000000) / 100000.0,
            min(order_book_data.get('ask_depth', 10000), 1000000) / 100000.0,
            max(-1, min(1, order_book_data.get('order_imbalance', 0))),
            max(-1, min(1, order_book_data.get('book_pressure', 0))),
            min(order_book_data.get('level2_spread', 0.01), 0.1) / 0.1,
        ])

        features.extend([
            max(-0.1, min(0.1, market_features.get('vwap_deviation', 0))) / 0.1,
            max(-0.05, min(0.05, market_features.get('momentum_1min', 0))) / 0.05,
            max(-0.05, min(0.05, market_features.get('momentum_5min', 0))) / 0.05,
            market_features.get('rsi', 50) / 100.0,
            max(0, min(1, market_features.get('bollinger_position', 0.5))),
        ])

        features.extend([
            max(0.1, min(10, market_features.get('venue_spread_ratio', 1.0))) / 10.0,
            max(0.1, min(10, market_features.get('venue_volume_ratio', 1.0))) / 10.0,
            max(-1, min(1, market_features.get('arbitrage_signal', 0))),
            max(0, min(1, market_features.get('venue_correlation', 0.8))),
            max(-1, min(1, market_features.get('lead_lag_indicator', 0))),
        ])

        features.extend([
            market_features.get('ema_20', mid_price) / 1000.0,
            market_features.get('ema_50', mid_price) / 1000.0,
            max(-0.1, min(0.1, market_features.get('macd_signal', 0))) / 0.1,
            market_features.get('stochastic_k', 50) / 100.0,
            min(market_features.get('atr', 1.0), 10.0) / 10.0,
        ])

        while len(features) < self.feature_size:
            features.append(0.0)

        features_array = np.array(features[:self.feature_size], dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)

        return features_array

    def predict(self, venue: str, features: np.ndarray) -> LatencyPrediction:

        start_time = time.time()

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        self.feature_buffers[venue].append(features)

        if len(self.feature_buffers[venue]) < self.sequence_length:
            return LatencyPrediction(
                venue=venue,
                timestamp=time.time(),
                predicted_latency_us=1000.0,
                confidence=0.1,
                prediction_time_ms=0.0,
                model_version=self.model_versions[venue],
                features_used={}
            )

        sequence = np.array(list(self.feature_buffers[venue]))
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=1.0, neginf=-1.0)
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                self.models[venue].eval()
                prediction, confidence = self.models[venue](input_tensor)

            predicted_latency = float(prediction.squeeze().cpu().numpy())
            predicted_latency = np.clip(abs(predicted_latency) * 1000 + 500, 50, 10000)

            confidence_score = float(confidence.squeeze().cpu().numpy())
            confidence_score = np.clip(confidence_score, 0.1, 1.0)

        except Exception as e:
            logger.warning(f"Prediction failed for {venue}: {e}")
            predicted_latency = 1000.0
            confidence_score = 0.1

        prediction_time_ms = (time.time() - start_time) * 1000

        prediction_obj = LatencyPrediction(
            venue=venue,
            timestamp=time.time(),
            predicted_latency_us=predicted_latency,
            confidence=confidence_score,
            prediction_time_ms=prediction_time_ms,
            model_version=self.model_versions[venue],
            features_used={
                'temporal': float(features[0]) if len(features) > 0 else 0.0,
                'network_latency': float(features[5]) if len(features) > 5 else 0.0,
                'market_volatility': float(features[14]) if len(features) > 14 else 0.0,
                'order_imbalance': float(features[17]) if len(features) > 17 else 0.0
            }
        )

        self.prediction_history[venue].append({
            'timestamp': prediction_obj.timestamp,
            'predicted': predicted_latency,
            'confidence': confidence_score,
            'features': features.tolist()
        })

        return prediction_obj

    def train_model(self, venue: str, training_data: Dict[str, np.ndarray],
                   epochs: int = 100, batch_size: int = 64,
                   learning_rate: float = 0.001) -> Dict[str, float]:

        logger.info(f"Training {self.model_type.value} model for {venue}...")

        features = training_data['features']
        targets = training_data['targets']

        if len(features) < self.sequence_length + 100:
            logger.warning(f"Insufficient data for {venue}: {len(features)} samples")
            return {'mae': float('inf'), 'accuracy': 0, 'error': 'insufficient_data'}

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        targets = np.nan_to_num(targets, nan=1000.0, posinf=10000.0, neginf=50.0)
        targets = np.clip(targets, 50.0, 50000.0)

        logger.info(f"Training data: {len(features)} samples, "
                   f"target range: [{targets.min():.1f}, {targets.max():.1f}] μs")

        try:
            dataset = LatencyDataset(features, targets, self.sequence_length)
        except Exception as e:
            logger.error(f"Failed to create dataset for {venue}: {e}")
            return {'mae': float('inf'), 'accuracy': 0, 'error': 'dataset_creation_failed'}

        if len(dataset) < 50:
            logger.warning(f"Dataset too small after processing: {len(dataset)} sequences")
            return {'mae': float('inf'), 'accuracy': 0, 'error': 'dataset_too_small'}

        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size

        if val_size < 1:
            val_size = 1
            train_size = len(dataset) - 1

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, train_size),
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, val_size),
            shuffle=False,
            drop_last=False
        )

        if self.model_type == ModelType.LSTM:
            model = LSTMLatencyModel(self.feature_size).to(self.device)
        else:
            model = GRULatencyModel(self.feature_size).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
        )

        best_val_loss = float('inf')
        best_model_state = None
        training_history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        patience_counter = 0
        max_patience = 20

        for epoch in range(epochs):
            model.train()
            train_losses = []

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                if torch.isnan(batch_features).any() or torch.isnan(batch_targets).any():
                    logger.warning(f"NaN detected in batch, skipping...")
                    continue

                optimizer.zero_grad()
                predictions, _ = model(batch_features)
                loss = criterion(predictions.squeeze(), batch_targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected, skipping batch")
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            if not train_losses:
                logger.error("No valid training batches!")
                break

            model.eval()
            val_losses = []
            val_mae = []

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    if torch.isnan(batch_features).any() or torch.isnan(batch_targets).any():
                        continue

                    predictions, _ = model(batch_features)
                    loss = criterion(predictions.squeeze(), batch_targets)

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    try:
                        pred_us = dataset.denormalize_prediction(predictions.squeeze())
                        target_us = dataset.denormalize_prediction(batch_targets)
                        mae = torch.mean(torch.abs(pred_us - target_us))

                        if not (torch.isnan(mae) or torch.isinf(mae)):
                            val_losses.append(loss.item())
                            val_mae.append(mae.item())
                    except Exception as e:
                        logger.warning(f"MAE calculation failed: {e}")
                        continue

            if not val_losses:
                logger.warning("No valid validation batches!")
                continue

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_val_mae = np.mean(val_mae) if val_mae else float('inf')

            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_mae'].append(avg_val_mae)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                           f"Val Loss={avg_val_loss:.4f}, Val MAE={avg_val_mae:.1f}μs, "
                           f"LR={optimizer.param_groups[0]['lr']:.6f}")

            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch} (patience exceeded)")
                break

        if best_model_state is None:
            logger.warning(f"No valid model state found for {venue}, using current state")
            best_model_state = model.state_dict().copy()
            best_val_loss = float('inf')

        try:
            model.load_state_dict(best_model_state)
        except Exception as e:
            logger.error(f"Failed to load best model state: {e}")

        model.eval()
        final_metrics = self._evaluate_model(model, val_loader, dataset)

        current_performance = self.model_performance[venue]['mae']
        if (final_metrics['mae'] < current_performance or
            current_performance == float('inf')):

            self.models[venue] = model
            self.model_versions[venue] = f"v{epoch}_{int(time.time())}"
            self.model_performance[venue].update(final_metrics)

            self._save_model_checkpoint(venue, model, final_metrics)

            logger.info(f"Updated {venue} model: Accuracy={final_metrics['accuracy']:.1f}%, "
                       f"MAE={final_metrics['mae']:.1f}μs")
        else:
            logger.info(f"Model performance did not improve for {venue}")

        return final_metrics

    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader,
                       dataset: LatencyDataset) -> Dict[str, float]:

        model.eval()

        all_predictions = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                if torch.isnan(features).any() or torch.isnan(targets).any():
                    continue

                try:
                    predictions, confidences = model(features)

                    if torch.isnan(predictions).any() or torch.isnan(confidences).any():
                        continue

                    pred_us = dataset.denormalize_prediction(predictions.squeeze())
                    target_us = dataset.denormalize_prediction(targets)

                    if torch.isnan(pred_us).any() or torch.isnan(target_us).any():
                        continue

                    all_predictions.extend(pred_us.cpu().numpy())
                    all_targets.extend(target_us.cpu().numpy())
                    # Fix 0-d array issue
                    conf_numpy = confidences.cpu().numpy()
                    if conf_numpy.ndim == 0:
                        all_confidences.append(float(conf_numpy))
                    else:
                        all_confidences.extend(conf_numpy.squeeze())

                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue

        if not all_predictions:
            logger.warning("No valid predictions in evaluation")
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'accuracy': 0.0,
                'confidence_correlation': 0.0,
                'predictions': 0
            }

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)

        valid_mask = (
            np.isfinite(all_predictions) &
            np.isfinite(all_targets) &
            np.isfinite(all_confidences)
        )

        if not np.any(valid_mask):
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'accuracy': 0.0,
                'confidence_correlation': 0.0,
                'predictions': 0
            }

        all_predictions = all_predictions[valid_mask]
        all_targets = all_targets[valid_mask]
        all_confidences = all_confidences[valid_mask]

        mae = np.mean(np.abs(all_predictions - all_targets))
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)

        threshold = all_targets * 0.1
        within_threshold = np.abs(all_predictions - all_targets) <= threshold
        accuracy = np.mean(within_threshold) * 100

        try:
            confidence_correlation = np.corrcoef(
                all_confidences,
                within_threshold.astype(float)
            )[0, 1]
            if np.isnan(confidence_correlation):
                confidence_correlation = 0.0
        except:
            confidence_correlation = 0.0

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'accuracy': float(accuracy),
            'confidence_correlation': float(confidence_correlation),
            'predictions': len(all_predictions)
        }

    def update_online(self, venue: str, features: np.ndarray,
                     actual_latency: float) -> Optional[Dict[str, float]]:

        if not self.online_learning_enabled:
            return None

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        actual_latency = np.clip(actual_latency, 50.0, 50000.0)

        self.update_buffer[venue].append((features, actual_latency))

        if len(self.update_buffer[venue]) >= self.update_threshold:
            batch_features = np.array([f for f, _ in self.update_buffer[venue]])
            batch_targets = np.array([t for _, t in self.update_buffer[venue]])

            try:
                mini_dataset = LatencyDataset(batch_features, batch_targets, self.sequence_length)

                if len(mini_dataset) > 0:
                    metrics = self._online_update_step(venue, mini_dataset)

                    self.update_buffer[venue] = []

                    return metrics
            except Exception as e:
                logger.warning(f"Online update failed for {venue}: {e}")
                self.update_buffer[venue] = []

        return None

    def _online_update_step(self, venue: str, dataset: LatencyDataset,
                           steps: int = 10) -> Dict[str, float]:

        model = self.models[venue]
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
        criterion = nn.MSELoss()

        data_loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

        update_losses = []

        for step in range(steps):
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                if torch.isnan(features).any() or torch.isnan(targets).any():
                    continue

                optimizer.zero_grad()
                predictions, _ = model(features)
                loss = criterion(predictions.squeeze(), targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                update_losses.append(loss.item())

        model.eval()

        timestamp = int(time.time())
        self.model_versions[venue] = f"{self.model_versions[venue]}_online_{timestamp}"

        return {
            'update_loss': np.mean(update_losses) if update_losses else float('inf'),
            'samples_processed': len(dataset),
            'update_steps': len(update_losses)
        }

    def get_ensemble_prediction(self, features: Dict[str, np.ndarray]) -> Dict[str, LatencyPrediction]:

        predictions = {}

        for venue in self.venues:
            if venue in features:
                pred = self.predict(venue, features[venue])
                predictions[venue] = pred

        return predictions

    def enable_online_learning(self, enabled: bool = True):

        self.online_learning_enabled = enabled
        logger.info(f"Online learning {'enabled' if enabled else 'disabled'}")

    def _save_model_checkpoint(self, venue: str, model: nn.Module,
                              metrics: Dict[str, float]):

        try:
            checkpoint = {
                'venue': venue,
                'model_state': model.state_dict(),
                'model_type': self.model_type.value,
                'metrics': metrics,
                'version': self.model_versions[venue],
                'timestamp': time.time(),
                'feature_size': self.feature_size,
                'sequence_length': self.sequence_length
            }

            import os
            os.makedirs("outputs/models/latency_models", exist_ok=True)
            filename = f"outputs/models/latency_models/latency_model_{venue}_{self.model_versions[venue]}.pt"
            torch.save(checkpoint, filename)

            if (venue not in self.best_models or
                metrics['accuracy'] > self.best_models[venue].get('accuracy', 0)):
                self.best_models[venue] = {
                    'filename': filename,
                    'metrics': metrics,
                    'version': self.model_versions[venue]
                }

            logger.info(f"Saved checkpoint for {venue}: {filename}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for {venue}: {e}")

    def load_model(self, venue: str, checkpoint_path: str):

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if checkpoint['model_type'] == 'lstm':
                model = LSTMLatencyModel(checkpoint['feature_size'])
            else:
                model = GRULatencyModel(checkpoint['feature_size'])

            model.load_state_dict(checkpoint['model_state'])
            model.to(self.device)
            model.eval()

            self.models[venue] = model
            self.model_versions[venue] = checkpoint['version']
            self.model_performance[venue] = checkpoint['metrics']

            logger.info(f"Loaded model for {venue}: {checkpoint['version']}")

        except Exception as e:
            logger.error(f"Failed to load model for {venue}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:

        summary = {
            'overall_metrics': {},
            'venue_performance': self.model_performance.copy(),
            'prediction_latencies': {},
            'model_versions': self.model_versions.copy(),
            'online_learning_status': self.online_learning_enabled
        }

        valid_performances = [p for p in self.model_performance.values()
                             if p['predictions'] > 0 and p['mae'] != float('inf')]

        if valid_performances:
            summary['overall_metrics'] = {
                'average_accuracy': np.mean([p['accuracy'] for p in valid_performances]),
                'average_mae': np.mean([p['mae'] for p in valid_performances]),
                'total_predictions': sum(p['predictions'] for p in valid_performances)
            }

        for venue in self.venues:
            if self.prediction_history[venue]:
                recent_preds = list(self.prediction_history[venue])[-100:]
                pred_times = [p.get('prediction_time_ms', 0) for p in recent_preds
                             if isinstance(p, dict) and 'prediction_time_ms' in p]
                if pred_times:
                    summary['prediction_latencies'][venue] = {
                        'mean_ms': np.mean(pred_times),
                        'p50_ms': np.percentile(pred_times, 50),
                        'p95_ms': np.percentile(pred_times, 95),
                        'p99_ms': np.percentile(pred_times, 99)
                    }

        return summary

    def export_models(self, export_dir: str = "models/"):

        import os
        try:
            os.makedirs(export_dir, exist_ok=True)

            for venue, model in self.models.items():
                model_path = os.path.join(export_dir, f"{venue}_model.pt")
                torch.save({
                    'model_state': model.state_dict(),
                    'model_config': {
                        'feature_size': self.feature_size,
                        'model_type': self.model_type.value,
                        'venue': venue
                    }
                }, model_path)

                try:
                    dummy_input = torch.randn(1, self.sequence_length, self.feature_size).to(self.device)
                    onnx_path = os.path.join(export_dir, f"{venue}_model.onnx")

                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['features'],
                        output_names=['prediction', 'confidence'],
                        dynamic_axes={
                            'features': {0: 'batch_size'},
                            'prediction': {0: 'batch_size'},
                            'confidence': {0: 'batch_size'}
                        }
                    )
                except Exception as e:
                    logger.warning(f"ONNX export failed for {venue}: {e}")

                logger.info(f"Exported models for {venue} to {export_dir}")

        except Exception as e:
            logger.error(f"Model export failed: {e}")
class TimeSeriesPredictor:


    def __init__(self, venues: List[str], prediction_horizons: List[int] = [1, 5, 10]):
        self.venues = venues
        self.prediction_horizons = prediction_horizons

        self.predictors = {
            horizon: LatencyPredictor(venues, sequence_length=50+horizon)
            for horizon in prediction_horizons
        }

        self.feature_engineer = AdvancedFeatureEngineer()

        self.model_configs = self._get_optimized_configs()

        logger.info(f"TimeSeriesPredictor initialized for {len(venues)} venues")

    def _get_optimized_configs(self) -> Dict[str, Dict]:

        return {
            'NYSE': {
                'model_type': ModelType.LSTM,
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.2,
                'learning_rate': 0.001
            },
            'NASDAQ': {
                'model_type': ModelType.GRU,
                'hidden_size': 192,
                'num_layers': 2,
                'dropout': 0.15,
                'learning_rate': 0.0015
            },
            'CBOE': {
                'model_type': ModelType.LSTM,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.25,
                'learning_rate': 0.001
            },
            'IEX': {
                'model_type': ModelType.GRU,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.002
            },
            'ARCA': {
                'model_type': ModelType.LSTM,
                'hidden_size': 192,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001
            }
        }

    def predict_multi_horizon(self, venue: str, current_features: np.ndarray) -> Dict[int, LatencyPrediction]:

        predictions = {}

        for horizon in self.prediction_horizons:
            pred = self.predictors[horizon].predict(venue, current_features)
            predictions[horizon] = pred

        return predictions

    def get_prediction_intervals(self, venue: str, confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:

        if venue not in self.predictors[1].prediction_history:
            return {}

        history = list(self.predictors[1].prediction_history[venue])
        if len(history) < 100:
            return {}

        errors = []
        for i in range(1, len(history)):
            if ('actual' in history[i] and 'predicted' in history[i-1] and
                isinstance(history[i], dict) and isinstance(history[i-1], dict)):
                error = history[i]['actual'] - history[i-1]['predicted']
                if np.isfinite(error):
                    errors.append(error)

        if not errors:
            return {}

        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100

        intervals = {}
        for horizon in self.prediction_horizons:
            error_std = np.std(errors) * np.sqrt(horizon)
            intervals[f'horizon_{horizon}'] = (
                np.percentile(errors, lower_percentile) - error_std,
                np.percentile(errors, upper_percentile) + error_std
            )

        return intervals
class AdvancedFeatureEngineer:


    def __init__(self):
        self.feature_importance = {}
        self.pca_components = None
        self.feature_scalers = {}
        self.interaction_features = []

    def engineer_features(self, raw_data: Dict[str, Any]) -> np.ndarray:

        features = []

        if 'timestamp' in raw_data:
            features.extend(self._extract_temporal_features(raw_data['timestamp']))

        if 'network_data' in raw_data:
            features.extend(self._extract_network_features(raw_data['network_data']))

        if 'market_data' in raw_data:
            features.extend(self._extract_market_features(raw_data['market_data']))

        if 'cross_venue_data' in raw_data:
            features.extend(self._extract_cross_venue_features(raw_data['cross_venue_data']))

        if len(features) >= 10:
            features.extend(self._create_interaction_features(features))

        features_array = np.array(features, dtype=np.float32)
        return np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)

    def _extract_temporal_features(self, timestamp: float) -> List[float]:

        dt = datetime.fromtimestamp(timestamp)

        features = [
            dt.hour / 24.0,
            dt.minute / 60.0,
            dt.second / 60.0,
            dt.microsecond / 1e6,

            np.sin(2 * np.pi * dt.hour / 24),
            np.cos(2 * np.pi * dt.hour / 24),
            np.sin(2 * np.pi * dt.minute / 60),
            np.cos(2 * np.pi * dt.minute / 60),

            float(9.5 <= dt.hour <= 16),
            float(dt.hour < 9.5),
            float(dt.hour >= 16),

            dt.weekday() / 6.0,
            float(dt.weekday() in [0, 4]),

            self._time_until_open(dt),
            self._time_until_close(dt),
            self._time_until_half_hour(dt),
        ]

        return features

    def _extract_network_features(self, network_data: Dict) -> List[float]:

        features = []

        features.extend([
            min(network_data.get('latency_us', 1000), 10000) / 10000.0,
            min(network_data.get('jitter_us', 100), 1000) / 1000.0,
            min(network_data.get('packet_loss_rate', 0), 1.0),
            min(network_data.get('bandwidth_utilization', 0.5), 1.0),
            min(network_data.get('congestion_score', 0.5), 1.0),
        ])

        if 'history' in network_data and network_data['history']:
            latencies = [h.get('latency_us', 1000) for h in network_data['history'][-100:]]
            latencies = [l for l in latencies if np.isfinite(l) and l > 0]

            if latencies:
                features.extend([
                    np.mean(latencies) / 10000.0,
                    np.std(latencies) / 1000.0,
                    np.percentile(latencies, 50) / 10000.0,
                    np.percentile(latencies, 95) / 10000.0,
                    (max(latencies) - min(latencies)) / 10000.0,
                ])
            else:
                features.extend([0.1, 0.01, 0.1, 0.2, 0.1])
        else:
            features.extend([0.1, 0.01, 0.1, 0.2, 0.1])

        return features

    def _extract_market_features(self, market_data: Dict) -> List[float]:

        features = []

        mid_price = market_data.get('mid_price', 100)
        volume = max(market_data.get('volume', 1000), 1)

        features.extend([
            mid_price / 1000.0,
            np.log(volume) / 10.0,
            max(market_data.get('spread', 0.01), 0.001) / mid_price,
            min(market_data.get('volatility', 0.01), 0.1) / 0.1,
            min(market_data.get('trade_intensity', 0.5), 10.0) / 10.0,
        ])

        features.extend([
            min(market_data.get('bid_depth', 10000), 1000000) / 100000.0,
            min(market_data.get('ask_depth', 10000), 1000000) / 100000.0,
            max(-1, min(1, market_data.get('order_imbalance', 0))),
            max(-1, min(1, market_data.get('book_pressure', 0))),
            np.log1p(max(market_data.get('order_count', 100), 1)) / 10.0,
        ])

        features.extend([
            market_data.get('rsi', 50) / 100.0,
            max(-0.1, min(0.1, market_data.get('macd_signal', 0))) / 0.1,
            max(0, min(1, market_data.get('bollinger_position', 0.5))),
            max(-0.1, min(0.1, market_data.get('vwap_deviation', 0))) / 0.1,
            max(-0.05, min(0.05, market_data.get('momentum_5min', 0))) / 0.05,
        ])

        return features

    def _extract_cross_venue_features(self, cross_venue_data: Dict) -> List[float]:

        features = []

        features.extend([
            max(-0.1, min(0.1, cross_venue_data.get('price_dispersion', 0))) / 0.1,
            max(0, min(1, cross_venue_data.get('volume_concentration', 0.5))),
            max(0, min(1, cross_venue_data.get('venue_correlation', 0.8))),
            max(-1, min(1, cross_venue_data.get('arbitrage_signal', 0))),
            max(-1, min(1, cross_venue_data.get('lead_lag_indicator', 0))),
            max(0, min(1, cross_venue_data.get('fragmentation_index', 0.3))),
        ])

        return features

    def _create_interaction_features(self, base_features: List[float]) -> List[float]:

        if len(base_features) < 10:
            return []

        interactions = []

        try:
            if len(base_features) > 25:
                vol_idx, volume_idx = min(24, len(base_features)-1), min(21, len(base_features)-1)
                interactions.append(base_features[vol_idx] * base_features[volume_idx])

            if len(base_features) > 15:
                latency_idx = min(15, len(base_features)-1)
                activity_idx = min(25, len(base_features)-1) if len(base_features) > 25 else min(10, len(base_features)-1)
                interactions.append(base_features[latency_idx] * base_features[activity_idx])

            if len(base_features) > 4:
                time_idx = 0
                congestion_idx = min(19, len(base_features)-1) if len(base_features) > 19 else min(4, len(base_features)-1)
                interactions.append(base_features[time_idx] * base_features[congestion_idx])

        except (IndexError, TypeError):
            pass

        return interactions

    def _time_until_open(self, dt: datetime) -> float:

        try:
            market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
            if dt >= market_open:
                market_open += timedelta(days=1)

            seconds_until = (market_open - dt).total_seconds()
            return max(0, min(1, seconds_until / 86400.0))
        except:
            return 0.5

    def _time_until_close(self, dt: datetime) -> float:

        try:
            market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
            if dt >= market_close:
                market_close += timedelta(days=1)

            seconds_until = (market_close - dt).total_seconds()
            return max(0, min(1, seconds_until / 86400.0))
        except:
            return 0.5

    def _time_until_half_hour(self, dt: datetime) -> float:

        try:
            next_half = dt.replace(minute=30 if dt.minute < 30 else 0, second=0, microsecond=0)
            if dt.minute >= 30:
                next_half += timedelta(hours=1)

            seconds_until = (next_half - dt).total_seconds()
            return max(0, min(1, seconds_until / 1800.0))
        except:
            return 0.5

    def calculate_feature_importance(self, features: np.ndarray,
                                   targets: np.ndarray) -> Dict[str, float]:

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance

            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            targets = np.nan_to_num(targets, nan=1000.0, posinf=10000.0, neginf=50.0)

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(features, targets)

            perm_importance = permutation_importance(rf, features, targets, n_repeats=10)

            self.feature_importance = {
                f'feature_{i}': importance
                for i, importance in enumerate(perm_importance.importances_mean)
            }

            return self.feature_importance

        except ImportError:
            logger.warning("sklearn not available for feature importance calculation")
            return {}
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}
class LatencyPredictorIntegration:


    def __init__(self, feature_extractor, network_simulator, market_generator):
        self.feature_extractor = feature_extractor
        self.network_simulator = network_simulator
        self.market_generator = market_generator

        venues = list(market_generator.venues)
        self.latency_predictor = LatencyPredictor(venues)
        self.time_series_predictor = TimeSeriesPredictor(venues)

        self.training_buffer = {venue: [] for venue in venues}
        self.buffer_size = 10000

        logger.info("LatencyPredictorIntegration initialized")

    async def process_tick(self, market_tick):

        try:
            feature_vector = self.feature_extractor.extract_features(
                market_tick.symbol,
                market_tick.venue,
                market_tick.timestamp
            )

            network_data = {
                'latency_us': self.network_simulator.get_current_latency(market_tick.venue),
                'jitter_us': self.network_simulator.get_jitter(market_tick.venue),
                'packet_loss_rate': self.network_simulator.get_packet_loss_rate(market_tick.venue),
                'congestion_score': self.network_simulator.get_congestion_score(market_tick.venue),
                'bandwidth_utilization': 0.5
            }

            features = self._prepare_features(market_tick, network_data, feature_vector)

            prediction = self.latency_predictor.predict(market_tick.venue, features)

            actual_latency = self.network_simulator.measure_latency(
                market_tick.venue,
                market_tick.timestamp
            ).latency_us

            self._store_training_data(market_tick.venue, features, actual_latency)

            if self.latency_predictor.online_learning_enabled:
                self.latency_predictor.update_online(market_tick.venue, features, actual_latency)

            return prediction

        except Exception as e:
            logger.warning(f"Failed to process tick for {market_tick.venue}: {e}")
            return LatencyPrediction(
                venue=market_tick.venue,
                timestamp=time.time(),
                predicted_latency_us=1000.0,
                confidence=0.1,
                prediction_time_ms=0.0,
                model_version="default",
                features_used={}
            )

    def _prepare_features(self, market_tick, network_data, feature_vector) -> np.ndarray:

        try:
            tick_data = {
                'timestamp': market_tick.timestamp,
                'mid_price': getattr(market_tick, 'mid_price', 100),
                'volume': getattr(market_tick, 'volume', 1000),
                'bid_price': getattr(market_tick, 'bid_price', 99.9),
                'ask_price': getattr(market_tick, 'ask_price', 100.1),
                'volatility': feature_vector.features.get('volatility_1min', 0.01),
                'trade_intensity': feature_vector.features.get('trade_intensity', 0.5)
            }

            order_book_data = {
                'bid_depth': feature_vector.features.get('bid_depth_total', 10000),
                'ask_depth': feature_vector.features.get('ask_depth_total', 10000),
                'order_imbalance': feature_vector.features.get('order_imbalance', 0),
                'book_pressure': feature_vector.features.get('book_pressure', 0),
                'level2_spread': feature_vector.features.get('spread_bps', 1) / 10000.0
            }

            market_features = {
                'vwap_deviation': feature_vector.features.get('vwap_deviation', 0),
                'momentum_1min': feature_vector.features.get('price_momentum_1min', 0),
                'momentum_5min': feature_vector.features.get('price_momentum_5min', 0),
                'rsi': feature_vector.features.get('rsi', 50),
                'bollinger_position': feature_vector.features.get('bollinger_position', 0.5),
                'ema_20': feature_vector.features.get('ema_20', tick_data['mid_price']),
                'ema_50': feature_vector.features.get('ema_50', tick_data['mid_price']),
                'macd_signal': feature_vector.features.get('macd_histogram', 0),
                'stochastic_k': feature_vector.features.get('stochastic_k', 50),
                'atr': feature_vector.features.get('atr', 1.0),
                'venue_spread_ratio': feature_vector.features.get('cross_venue_spread_ratio', 1.0),
                'venue_volume_ratio': feature_vector.features.get('venue_volume_share', 0.2),
                'arbitrage_signal': feature_vector.features.get('arbitrage_opportunity', 0),
                'venue_correlation': 0.8,
                'lead_lag_indicator': feature_vector.features.get('price_leadership_score', 0)
            }

            return self.latency_predictor.extract_features(
                tick_data, network_data, order_book_data, market_features
            )

        except Exception as e:
            logger.warning(f"Feature preparation failed: {e}")
            return np.zeros(45, dtype=np.float32)

    def _store_training_data(self, venue: str, features: np.ndarray, actual_latency: float):

        try:
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            actual_latency = np.clip(actual_latency, 50.0, 50000.0)

            self.training_buffer[venue].append({
                'features': features,
                'target': actual_latency,
                'timestamp': time.time()
            })

            if len(self.training_buffer[venue]) > self.buffer_size:
                self.training_buffer[venue].pop(0)

        except Exception as e:
            logger.warning(f"Failed to store training data for {venue}: {e}")

    async def train_models(self):

        for venue in self.training_buffer:
            try:
                if len(self.training_buffer[venue]) >= 1000:
                    features = np.array([d['features'] for d in self.training_buffer[venue]])
                    targets = np.array([d['target'] for d in self.training_buffer[venue]])

                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    targets = np.nan_to_num(targets, nan=1000.0, posinf=10000.0, neginf=50.0)
                    targets = np.clip(targets, 50.0, 50000.0)

                    training_data = {
                        'features': features,
                        'targets': targets
                    }

                    metrics = self.latency_predictor.train_model(
                        venue,
                        training_data,
                        epochs=50,
                        batch_size=64
                    )

                    logger.info(f"Trained model for {venue}: {metrics}")

            except Exception as e:
                logger.error(f"Training failed for {venue}: {e}")

    def get_routing_recommendation(self, symbol: str) -> str:

        try:
            predictions = {}

            for venue in self.latency_predictor.venues:
                try:
                    feature_vector = self.feature_extractor.get_latest_features(symbol, venue)
                    if feature_vector:
                        features = self._prepare_features_from_vector(feature_vector)
                        pred = self.latency_predictor.predict(venue, features)
                        predictions[venue] = pred
                except Exception as e:
                    logger.warning(f"Failed to get prediction for {venue}: {e}")
                    continue

            if not predictions:
                return self.latency_predictor.venues[0] if self.latency_predictor.venues else "NYSE"

            best_venue = None
            best_score = float('inf')

            for venue, pred in predictions.items():
                score = pred.predicted_latency_us * (2 - pred.confidence)
                if score < best_score:
                    best_score = score
                    best_venue = venue

            return best_venue or self.latency_predictor.venues[0]

        except Exception as e:
            logger.warning(f"Routing recommendation failed: {e}")
            return self.latency_predictor.venues[0] if self.latency_predictor.venues else "NYSE"

    def _prepare_features_from_vector(self, feature_vector) -> np.ndarray:

        try:
            features = np.zeros(45, dtype=np.float32)

            feature_map = {
                'timestamp': 0,
                'volatility_1min': 14,
                'mid_price': 10,
                'volume': 11,
                'spread_bps': 12,
                'order_imbalance': 17,
                'rsi': 23,
                'momentum_1min': 21,
                'momentum_5min': 22
            }

            for feat_name, idx in feature_map.items():
                if feat_name in feature_vector.features and idx < len(features):
                    value = feature_vector.features[feat_name]
                    if np.isfinite(value):
                        features[idx] = float(value)

            return features

        except Exception as e:
            logger.warning(f"Feature vector preparation failed: {e}")
            return np.zeros(45, dtype=np.float32)