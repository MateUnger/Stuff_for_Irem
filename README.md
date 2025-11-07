## Files and Folders

- `utils.py`: Contains core functions for gait analysis and processing.
- `vGait.py`: Main script for running gait analysis workflows.
- `properties.json`: Configuration file with gait analysis parameters.

## What to do?

1. export relevant reconding from Qualisys Tracking Manager thingy in .tsv (
tab-separated-values). Also get the start and stop .json files for synch.

2. use "process_qualisys_data.ipynb" to parse metadata and marker data from files

3. use vGait.py to analyze the marker positions (all helper functions should be in utils.py)

4. (optional) visualize the results with vis.py (no need to perfom gait analysis in advance, it is included in vis.py)

5. (optional) ask Máté if you are stuck with something (vscode, virtual env, any of the steps/scripts)
