# Transfer Testing

#### Running a test on data-small

- Download data-small from /local_storage/gwo/public/transfer-testing/imagenet/data-small located on sanna server to your home dir
- Run testSplitter.py to split prechosen class into 20 test samples 5 val samples and tempDir
- Run smallDataTest.py with args defining path to data-small in your home-dir and what % of tempDir do you want to add to train set
- Results are saved in small_data_results directory in the form of json file

Sidenote: in order to keep the results actually meaningful there should be no fine tuning, an architecture (optimally one that is widely known and covered in papers) and a set of hyperparameters should be decided on and kept constant.
The reason for that is we want to eliminate as many moving parts as possible and focus on impact of the thing we actually want to research.

#### Getting started

First you need to have PDM installed, run this command:

Linux/Mac `curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -`

Windows `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py -UseBasicParsing).Content | python -`

Now simply install dependencies by running

`pdm install`

And you are ready to go :) venv is created in working folder in .venv in case you need to setup your IDE.


#### Downloading datasets

You need to have Kaggle API Token

Just run:

`pdm run transfer-testing/utils/datasets.py`

It's temporary, every dataset will be downloaded within lightning data module


#### Roadmap (WIP, ugly, it's a draft):

- [ ]  Basic deep learning CV model in PyTorch with easy setup
- [ ]  Dataloaders for Imagenet (Or any image classification data)
- [ ]  Pipeline to support training and validating (Dagster?, DVC?)
- [ ]  Metric server (MLFlow, WandB)
- [ ]  Transfer testing pipeline setup, current idea, not validated yet, needs fixing:
  - [ ]  Subset data from imagenet from one domain (for example: mammals)
  - [ ]  Train CV model on Imagenet - Training dataset: original augmented, Validation dataset: original augmented, Test dataset: original
  - [ ]  Check for test error
  - [ ]  Subset new classes from similar domain (for example birds)
  - [ ]  Retrain model on small part of that classes (10% of birds)
  - [ ]  Hypothesis: test error on rest of the dataset for similar dataset should be small or similar to test error from original dataset, that is model trained on similar data should generalize on unseen data with being presented only subset of new classes (or examples, we need to check this) during training
- [ ]  Check how we can utilize approaches from cited articles
