import os
import sys
os.chdir('FISH DATASET')
sys.path.append('.')

# Import and run the contamination analysis
exec(open('investigate_dataset_contamination.py').read())