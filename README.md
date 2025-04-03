# BEANS-Zero

A zero-shot audio + text, bioacoustic benchmark dataset.


## Installation
Create a virtual environment, we recommend "uv".
```curl -LsSf https://astral.sh/uv/install.sh | sh```

Install with pip.
TODO

## From source

1. Clone this repo

2. ```cd beans-zero```

3. ```uv sync```

4. Activate the virtual environment: `source .venv/bin/activate`

## How can I evaluate my audio-text bioacoustic model ?

You can evaluate your model on BEANS-Zero by first:
1. Downloading or streaming the dataset from Huggingface `EarthSpeciesProject/BEANS-Zero`. [link]
```python
from datasets import load_dataset

# streaming
ds = load_dataset("EarthSpeciesProject/BEANS-Zero", streaming=True)

# get information on the columns in the dataset
print(ds)

# iterate over the samples
for sample in ds:
    break

# each sample is dict
print(sample.keys())
```

2. For each sample in the dataset, your model should generate a text `prediction`. Use the `audio` and `instruction`
fields in the sample to make your prediction. The `output` key is the expected output. Save this as `label`. Also,
aggregate

3. Aggregate `prediction`, `label` and `dataset_name` fields in your evaluation loop. Save these to a jsonl or csv
file (jsonl preferred for data type encoding issues).

4. Call `python evaluate.py /path/to/your/predictions_file.jsonl /path/to/save/metrics.json`

### Run the benchmark

1. Use the `ModelWrapper` class in `beans_zero/ to create your model.
2. Run `beans_data/benchmark.py` like so
