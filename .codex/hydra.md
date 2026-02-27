Crucial Updates to Loading Logic (Implement Strictly):

Data Loading (GCS ONLY):

Remove all options/logic for loading test data (ERA5/HRES) from local storage. - Data must only be fetched from DeepMind's public GCS bucket using an anonymous client.

You must implement the data fetching using exactly this logic pattern:

Python

from google.cloud import storage
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
# Use the appropriate dir_prefix based on the dataset required
Model/Checkpoint Loading (GCS OR Local):

I am developing my own custom models alongside DeepMind's. Therefore, the script must support a toggle in the Hydra config (e.g., model.source: "gcs" | "local").

If "gcs", fetch the pre-trained DeepMind weights using the exact same anonymous GCS logic mentioned above (using dir_prefix = "gencast/" or "params/" as appropriate).

If "local", load the .npz or Haiku checkpoint from a local filepath specified in the config.


Constraints:

Use clear type hints.

Separate the GCS fetching logic into a cleanly abstracted helper function so it can be reused for both data and model weights.


Hydra Configuration Formatting Constraint (STRICT):

When generating or updating YAML files inside subdirectories (e.g., config/model/gencast.yaml or config/data/era5.yaml), do not nest the variables under the top-level group key. - Hydra automatically assigns the namespace based on the folder structure. Including the top-level key inside the file creates a double-nested error (e.g., model.model.kind).

Write the variables directly at the root level of the file.

BAD Example (config/model/gencast.yaml):

YAML

model:
  kind: gencast
  ensemble_size: 8
GOOD Example (config/model/gencast.yaml):

YAML

kind: gencast
ensemble_size: 8

