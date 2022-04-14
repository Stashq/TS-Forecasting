import numpy as np
import pandas as pd
from literature.dagmm.dagmm_adopted import DAGMM

t = np.arange(0, 10000)

df = pd.DataFrame(np.sin(t * 0.1) + 0.25 * np.random.randn(len(t)))
model = DAGMM()
model.fit(df)
