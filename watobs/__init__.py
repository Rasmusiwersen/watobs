from .dmi import DMIOceanObsRepository
from .altimetry import DHIAltimetryRepository, CMEMSSatObsRepository
from . import cmems

__version__ = "0.1.1"

__all__ = ["DMIOceanObsRepository", "DHIAltimetryRepository", "CMEMSSatObsRepository", "cmems"]
