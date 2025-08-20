__version__ = "1.0.0"
__author__ = "izaa"
__email__ = "saishxnde@gmail.com"
try:
    from .core import ReplayBuffer, Experience, ExperienceBatch
except ImportError:
    ReplayBuffer = None
    Experience = None
    ExperienceBatch = None
try:
    from .bindings import HFTReplayBufferCpp, HFTRoutingIntegration
except ImportError:
    HFTReplayBufferCpp = None
    HFTRoutingIntegration = None
def benchmark_performance():

    if HFTReplayBufferCpp:
        print("Running HFT replay buffer benchmarks...")
    else:
        print("HFT bindings not available for benchmarking")
def create_random_experience(state_dim=10):

    import numpy as np
    if Experience:
        return Experience(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, 4),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=np.random.rand() < 0.1
        )
    return None
__all__ = [
    'ReplayBuffer',
    'Experience',
    'ExperienceBatch',
    'HFTReplayBufferCpp',
    'HFTRoutingIntegration',
    'benchmark_performance',
    'create_random_experience'
]