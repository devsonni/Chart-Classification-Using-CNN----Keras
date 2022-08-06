import os
import shutil
import random
import glob
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir('charts/Train/Horizontal Bar')

for c in random.sample(glob.glob('*.png'), 40):
    shutil.move(c, '../../Val/Horizontal Bar')