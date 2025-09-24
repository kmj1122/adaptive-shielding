from dataclasses import dataclass

# This will be usedin envs.safety_gymnasium.builder.py, world.py, and etc.
@dataclass
class EnvironmentConfig:
    FIX_HIDDEN_PARAMETERS: bool = False
    USE_ORACLE: bool = False
    IS_OUT_OF_DISTRIBUTION: bool = False
    # These are parameters to control the variability of the environment
    MIN_MULT: float = 0.3
    MAX_MULT: float = 1.7
    ENV_ID: str = 'SafetyPointGoal1-v0'
    # These parameters are used to define the number of gremlins and static obstacles only for Goal2-Tasks
    NBR_OF_GREMLINS: int = 1
    NBR_OF_GOALS: int = 1
    NBR_OF_HAZARDS: int = 1
    NBR_OF_PILLARS: int = 1
    NBR_OF_VASES: int = 1
    PLACEMENT_EXTENTS: int = 2
    NBR_OF_BASIS: int = 2

@dataclass
class DynamicPredictorConfig:
    MAX_HISTORY: int = 1
    VOLUME: int = 1
    N_BASIS: int = 2
    MAX_LEN: int = 100
    LEARNING_RATE: float = 1e-3
    EPOCH: int = 1000
    ENSEMBLE_SIZE: int = 2
    LEAST_SQUARES: bool = True
    AVERAGE_FUNCTION: bool = True
    HIDDEN_SIZE: int = 512
    USE_ATTENTION: bool = False
    LEARNING_DOMAIN: str = 'ds'

@dataclass
class MovingObstaclePredictorConfig:
    MAX_HISTORY: int = 5
    PREDICTION_HORIZON: int = 1
    LEARNING_RATE: float = 1e-3
    EPOCH: int = 10
    MAX_LEN: int = 100
    ENSEMBLE_SIZE: int = 2
    N_BASIS: int = 20
    LEAST_SQUARES: bool = True
    AVERAGE_FUNCTION: bool = True
    HIDDEN_SIZE: int = 256
    USE_ATTENTION: bool = True
    LEARNING_DOMAIN: str = 'ts'