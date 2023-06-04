# Transfer Testing


Roadmap (WIP, ugly, it's a draft):

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



