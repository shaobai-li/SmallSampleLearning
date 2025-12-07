import nrrd
import numpy as np
import sys
data,_=nrrd.read(sys.argv[1])
print(f"shape={data.shape}, min={data.min()}, max={data.max()}")
