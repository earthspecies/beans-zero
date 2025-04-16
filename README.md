# BEANS-Zero

A zero-shot audio + text, bioacoustic benchmark dataset. This dataset is described in our companion paper
here: [Robinson et al 2025](https://openreview.net/forum?id=hJVdwBpWjt)

The dataset is available on [Huggingface](https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero) where we provide a detailed description of the dataset and its components.

## Installation
Create a virtual environment, we recommend *uv*, but you could also use *venv* or *conda*.
To install uv run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install a java runtime
The `captioning` task needs a java 8 run-time for the evaluation code to run. If you are on
mac visit https://www.java.com/en/download/ and install the appropriate version (Java SE 8).
If you are on a UNIX machine, visit https://openjdk.org/install/ and find your distribution.
You can also do:
```bash
sudo apt update
sudo apt install openjdk-8-jre
```
If you already have a more recent version of java, you can run
```bash
sudo update-alternatives --config java
```
and then select the java 8 version

If you are on Windows, visit: https://www.java.com/en/download/manual.jsp and download the appropriate installer.


### Install with pip.
COMING SOON...

### Install from source

1. Clone this repo.

2. ```cd beans-zero```

3. ```uv sync``` (or `pip install -e .`)

4. Optionally, for uv, activate the virtual environment: `source .venv/bin/activate`

## Task descriptions

There are three types of tasks in the BEANS-Zero dataset:
1. `classification`: in a classification task, the ground truth label is a single class label, e.g. a species name or an environmental sound (see the next section to get info on the labels). Your model's `prediction` should be a single label.
2. `detection`: in a detection task, the label *can* be a comma-separated list of class labels or a single label.
   - For example, the label can be "species1, species2, species3" or "species1" depending on how many species are detected in the audio.
   - Your model
3. `captioning`: in a captioning task, the label is a string that describes the content of the audio. Your model's `prediction` should be a string that is similar to the label. We use `SPICE` and `CIDer` scores to evaluate captioning
performance.

#### Notes on important dataset fields / columns
1. Each example in the dataset contains an `output` field which is the ground truth label(s) for that example.
2. Each example contains the `audio` as a list of floats.
3. And the `instruction_text` field which is the instruction text for that example.
All fields are described here
[Huggingface](https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero)

## What can I do with beanz-zero ?
beanz-zero offers several tools.

1. A cli tool to:
    - fetch the dataset from Huggingface `beanz-fetch`
    - list all component datasets `beanz-info` or info on a particular dataset `beanz-info <dataset_name>`
    - or just evaluate the results if you already have them in a predictions file `beanz-evaluate`
2. A `run_benchmark` function to run your model on the dataset and get evaluation metrics directly.

Run ```python cli.py --help``` to see the available commands.

### Fetch the dataset
Make sure you have installed beans-zero and have activated the virtual environment
```bash
beanz-fetch
```
### I have my predictions ready. How can I evaluate my model predictions ?
Your predictions file should be a csv or a jsonl (json lines, oriented as 'records') file with the following fields:
- `prediction`: the predicted string output of your model for each example
- `label`: the expected output (just copy the `output` field from the dataset for that example)
- `dataset_name`: the name of the dataset (e.g. 'esc50' or 'unseen-family-sci', again just copy the `dataset_name` field from the dataset)

Make sure you have installed beans-zero and have activated the virtual environmen, then run:
```bash
beanz-evaluate /path/to/your/predictions_file.jsonl /path/to/save/metrics.json
```
The output metrics per dataset component will be saved in the `metrics.json` file.
Currently, the supported output file format is json.

### How can I run my model on the BEANS-Zero dataset and get evaluation metrics directly ?
We provide a `run_benchmark` function that you can use to run your model on the dataset and get evaluation metrics.

1. Import the run_benchmark function into your model file.
2. Make sure your model class / prediction function must be a Callable. It will
   be called with a dictionary containing the following useful fields (amongst others):
   - `audio`: the audio input as a list of floats
   - `instruction_text`: the instruction text as a string
   NOTE: If the `batched` argument in `run_benchmark` is set to True, the
    `audio` field will be a list of lists of floats, and the `instruction_text` field
    will be a list of strings with `len(instruction_text) == batch_size`.
3. Call the run_benchmark function with your model class / prediction function
as the value of the `model` argument.

Here is an example of how to use the run_benchmark function:
```python
from beans_zero.benchmark import run_benchmark

class MyModel():
    "An example model class"
    def __init__(self):
        # Initialize your model and text tokenizer here

    def predict(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        "A simple forward pass."
        # Do something with the input tensor
        return x

    def __call__(self, example: dict) -> str:
        "A simple call method."
        # Extract the audio and text from the example
        audio = torch.Tensor(example["audio"])
        instruction = self.tokenizer(example["instruction_text"])

        # Perform inference
        prediction = self.predict(audio, instruction)
        # Convert the prediction to a string
        prediction = self.tokenizer.decode(prediction)

        return prediction

# Create an instance of your model
my_model = MyModel()
path_to_dataset = "EarthSpeciesProject/BEANS-Zero"
# path_to_dataset can be a local folder if you've downloaded the dataset somewhere else on your machine
```

Now, run the benchmark.
```python
run_benchmark(
    model=my_model,
    path_to_dataset=path_to_dataset,
    streaming=False,
    batched=False,
    batch_size=0,
    output_path="metrics.json",
)
```
> NOTE: streaming=True will not download the dataset, rather every example will be downloaded on the fly
> which is slower but saves disk space. Set batched=True if your model can handle batches of examples of size batch_size

### Dataset exploration
If you would first like to explore the BEANS-Zero dataset you can use this code snippet.
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

## Citation
If you use this dataset in your research, please cite the following paper:

```bibtex
@inproceedings{robinson2025naturelm,
  title     = {NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics},
  author    = {David Robinson and Marius Miron and Masato Hagiwara and Olivier Pietquin},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=hJVdwBpWjt}
}
```
