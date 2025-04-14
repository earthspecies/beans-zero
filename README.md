# BEANS-Zero

A zero-shot audio + text, bioacoustic benchmark dataset.


## Installation
Create a virtual environment, we recommend "uv", but you could also use "venv" or "conda".
```curl -LsSf https://astral.sh/uv/install.sh | sh```

### First, install a java runtime
The `captioning` task needs a java 8 run-time for the evaluation code to run. If you are on
mac visit https://www.java.com/en/download/ and install the appropriate version (Java SE 8).
If you are on a UNIX machine, visit https://openjdk.org/install/ and find your distribution.
You can also do:
```bash
sudo apt update
sudo apt install openjdk-8-jdk
# if you already of more recent version of java, you can do
sudo update-alternatives --config java
# then select the java 8 version
```

If you are on Windows: https://www.java.com/en/download/manual.jsp


### Install with pip.
COMING SOON...

## From source

1. Clone this repo.

2. ```cd beans-zero```

3. ```uv sync``` (or `pip install -e .`)

4. Optionally, for uv, activate the virtual environment: `source .venv/bin/activate`

## How can I evaluate my audio+text multimodal model ?

The beans-zero package offers a cli tool to:
    - fetch the dataset from Huggingface
    - run your model on the dataset and evaluate the results
    - or just evaluate the results if you already have in a predictions file

Run ```cli --help``` to see the available commands.

### Fetch the dataset
```bash
# if you have activated the virtual environment
beanz-fetch
# or, if you are using uv, and have *not* activated the virtual environment
uv run beanz-fetch
```

### Evaluate your model predictions
You should generate a csv / jsonl (orient='records') file with the following fields:
- `prediction`: the text output of your model
- `label`: the expected output (just copy the `output` field from the dataset)
- `dataset_name`: the name of the dataset (e.g. 'esc50' or 'unseen-family-sci', again just copy the `dataset_name` field from the dataset)

Then run:
```bash
# if you have activated the virtual environment
beanz-evaluate /path/to/your/predictions_file.jsonl /path/to/save/metrics.json
# or, if you are using uv, and have *not* activated the virtual environment
uv run beanz-evaluate /path/to/your/predictions_file.jsonl /path/to/save/metrics.json
```
The output metrics per dataset component (e.g. esc50 or unseen-family-sci) will be saved in the `metrics.json` file.
Currently, the supported output file format is json.

### Run your model on the dataset and get evaluation metrics
Create a file called `model.py` (or any other name) and add your model code there.
You may also be use an installed module in your virtual environment,  like `mymodule.classifier`.

IMPORTANT: your module or `model.py` MUST contain a function called `predict`,
that takes a single example (or a batch of examples) from the dataset as a dictionary
and returns the model's predictions (a string or a list of strings).

For example, your `model.py` could look like this:
```python
import torch

class MyModel(torch.nn.Module):
    "A simple example model."
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize your model here

    def forward(self, audio: torch.Tensor, text: str) -> torch.Tensor:
        "A simple forward pass."
        # Do something with the input tensor
        return x

model = MyModel()

def predict(example: dict) -> str | list[str]:
    # The example contains the 'audio' and "instruction_text" fields
    # You can use your model to make a prediction.

    # if batched is True, 'example' will be a dict of arrays. The 'audio'
    # field will be a list[list[float]]
    # and the 'instruction_text' field will be a list[str].

    # if batched is False, 'example' will be a dict of single elements
    # The 'audio' field will be a list[float] and the 'instruction_text'
    # field will be a str.

    audio = torch.Tensor(example["audio"])
    text = example["instruction_text"]
    # You can use your model to make a prediction.
    with torch.no_grad():
        prediction = model(audio, text)
    # Return the model's prediction
    # the prediction can be a single string or a list of strings
    # depending on the batched = False / True respectively.
    return prediction
```

Then run:
```bash
# if you have activated the virtual environment
beanz-benchmark --path-to-model-module /mydir/mypackage/model.py \
--streaming \  # optional, if you want to stream the dataset
--batched \ # optional, if you want to process the dataset in batches
--batch-size 32 \ # the batch size to use if batched is True, default is 32
--output_path /path/to/save/metrics.json \ # optional, the path to save the metrics, default is ./metrics.json
```

### Dataset exploration
If you would first like to explore the BEANS-Zero dataset [link], you can use this code snippet.
```python
import numpy as np
from datasets import load_dataset

# streaming=True will not download the dataset, rather evey example will be downloaded on the fly
# This can be slower but saves disk space
ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test", streaming=True)

# or not streaming (full download, about 100GB), much faster
# ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test")

# get information on the columns in the dataset
print(ds)

# iterate over the samples
for sample in ds:
    break

# each sample is dict
print(sample.keys())

# the component datasets are
dataset_names, dataset_sample_counts = np.unique(ds["dataset_name"], return_counts=True)

# if you want to select a subset of the data
idx = np.where(np.array(ds["dataset_name"]) == "esc50")[0]
esc50 = ds.select(idx)
print(esc50)
```
