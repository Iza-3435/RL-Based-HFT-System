"""Production feature extraction with clean interface."""

from typing import Dict, List
import numpy as np

from src.core.logging_config import get_logger
from src.core.types import VenueConfig
from data.feature_extractor import FeatureExtractor as LegacyFeatureExtractor

logger = get_logger()


class ProductionFeatureExtractor:
    """Production wrapper for feature extraction."""

    def __init__(self, symbols: List[str], venues: Dict[str, VenueConfig]):
        self.symbols = symbols
        self.venues = venues

        self._extractor = LegacyFeatureExtractor(symbols, venues)

        logger.verbose("Feature extractor initialized", symbols=len(symbols), venues=len(venues))

    def extract(self, tick_data: Dict) -> Dict:
        """Extract features from market tick."""
        try:
            return self._extractor.extract_features(tick_data)

        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return {}

    def extract_full(
        self, tick_data: Dict, network_data: Dict, order_book_data: Dict
    ) -> np.ndarray:
        """Extract full feature vector."""
        try:
            return self._extractor.extract_full_features(tick_data, network_data, order_book_data)

        except Exception as e:
            logger.debug(f"Full feature extraction error: {e}")
            return np.zeros(45, dtype=np.float32)
