from enum import Enum

class HatchLevel(Enum):
    LEVEL3 = 1
    LEVEL4 = 10
    LEVEL5 = 40

class SpeciesMultiplier(Enum):
    SAME_SPECIES = 3.0
    DIFFERENT_SPECIES = 1.0

class ChargeMultiplier(Enum):
    REGULAR_CHARGE = 1.0
    HYPER_CHARGE = 1.6
