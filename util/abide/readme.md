# ABIDE

These scripts will download and preprocess ABIDE dataset.

## Usage

```bash
cd util/abide/

# If you meet time-out error, execute this command repeatly. The script can continue to download from the last failed file.
python 01-fetch_data.py --root_path /path/to/the/save/folder/ --id_file_path subject_IDs.txt --download True

# Generate correlation matrices.
python 02-process_data.py --root_path /path/to/the/save/folder/ --id_file_path subject_IDs.txt

# Generate the final dataset.
python 03-generate_abide_dataset.py --root_path /path/to/the/save/folder/
```