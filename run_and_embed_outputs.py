import json
import sys
import io
import os
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

notebook_path = '/Applications/annhack/ANN-Breakthrough/ANN_Hackathon_Predictive_Maintenance.ipynb'

with open(notebook_path) as f:
    nb = json.load(f)

scope = {}
execution_counter = 1

print("Executing all notebook cells and embedding outputs...")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        print(f"Executing Code Cell {execution_counter} (Cell index {i})...")
        
        # Capture stdout
        stdout_buf = io.StringIO()
        plt.close('all')
        
        with contextlib.redirect_stdout(stdout_buf):
            try:
                exec(source, scope)
            except Exception as e:
                print(f"Error in cell {execution_counter}: {e}", file=sys.stderr)
                raise
        
        captured_text = stdout_buf.getvalue()
        cell_outputs = []
        
        if captured_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in captured_text.splitlines()]
            })
            print(f"Captured {len(captured_text.splitlines())} lines of output.")
        
        # Capture any matplotlib figures produced by this cell
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
            img_buf.seek(0)
            b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_data,
                    "text/plain": ["<Figure size ... with ... Axes>"]
                },
                "metadata": {}
            })
            print("Captured matplotlib figure.")
        
        plt.close('all')
        
        cell['outputs'] = cell_outputs
        cell['execution_count'] = execution_counter
        execution_counter += 1

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print("\nSUCCESS: All cells executed and all outputs/plots embedded into the notebook!")
