
import numpy as np
from PIL import Image

heightmap = Image.open("Heightmap.png")

np.save("heightmap.npy",heightmap)
