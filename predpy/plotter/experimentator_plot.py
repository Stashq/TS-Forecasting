from typing import List, Dict
from dataclasses import dataclass, field
from predpy.experimentator import Experimentator


@dataclass
class ExperimentatorPlot:
    '''Data class representing experimentator plotting parameters

    *datasets_to_models* should be a List of dictionaries mapping every
    dataset to models which predictions you want to plot.
    '''
    experimentator: Experimentator
    datasets_to_models: Dict[int, List[int]] = field(default_factory=list)
    rescale: bool = True
