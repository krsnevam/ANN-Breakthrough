import nbformat
from nbclient import NotebookClient
import os
import matplotlib
matplotlib.use('Agg')

print("Executing notebook to pre-populate all final outputs and graphs...")
notebook_filename = '/Applications/annhack/ANN-Breakthrough/ANN_Hackathon_Predictive_Maintenance.ipynb'

with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python314-ml')
client.execute()

with open(notebook_filename, 'w') as f:
    nbformat.write(nb, f)

print("NOTEBOOK EXECUTION COMPLETE! All cell outputs and plots saved permanently to the notebook!")
