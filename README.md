# Text To Layout
Text-to-layout using Transformers and recurrent neural network (RNN).

Further information of the project can be found on: https://drive.google.com/file/d/1Yl1cGAmuh3OoNcpNGxad2muZvW3aNNMQ/view?usp=sharing

## Libraries
The code has been tested using Python 3.9.7 and pytorch 1.10.2.

## Dataset
Download and save the data in each model folder:

https://drive.google.com/file/d/1FQC2yEV6--yM2ejsOsILR7pOTeTknLmE/view?usp=sharing

The data contains:
- datasets: A folder containing the following datasets:
  - AMR2014: Training and Testing datasets with the captions unprocessed.
  - AMR2014train-dev-test: Training, development and testing datasets with the captions processed.
  - COCO-annotations: MSCOCO2014 training and testing annotations.
- text_encoder100.pth: Pretrained text encoder (DAMSM).
- captions.pickle: Vocabulary of the pretrained encoder.

## Training

To train the model you need to set up the following variables in *main.py* file:

- *IS_TRAINING*: True.
- *EPOCHS*: Number of epochs to train.
- *CHECKPOINTS_PATH*: The path to save the checkpoints.

Additionally, you can set up the following variables to save the outputs of the development dataset.
- *SAVE_OUTPUT*: True.
- *VALIDATION_OUTPUT*: Path to store the output.

For testing the training process is recommended to set the variable *UQ_CAP* in *main.py* to True.

## Testing
To test the model you need to set up the following variables in *main.py* file:

- *IS_TRAINING*: False.
- *EPOCH_VALIDATION*: The epoch number(s) to validate.
- *VALIDATION_OUTPUT*: Path to store the output.
- *SAVE_OUTPUT*: True.

## Pretrained model

TRAN2LY, STRAN2LY and TRAN2TRAN checkpoints

https://drive.google.com/drive/folders/1gu2cLxldUJ7Iwsa3vyL_aP7wi4FfCA97?usp=sharing

Each one with their frozen and unfrozen versions and the configuration used to train them

## Metrics

To understand the metrics check chapter 5.2 of https://drive.google.com/file/d/1Yl1cGAmuh3OoNcpNGxad2muZvW3aNNMQ/view?usp=sharing

| System        | RSCP↑  | AR↓    | RS↓       | P↑      | F1↑    | R↑      |
|---------------|--------|--------|-----------|---------|--------|---------|
| Obj-GAN       | 0.348  | 0.246  | 2216.491  | 0.866   | 0.566  | 0.499   |
| TRAN2LY_UF    | 0.37   | 0.21   | 10.64     | 0.89    | 0.64   | 0.58    |
| STRAN2LY_UF   | 0.36   | 0.21   | 10.83     | 0.89    | 0.63   | 0.57    |


## Information about the project
- Type of project: End of degree project.
- Author: Eneko Suarez Etxeberria.
- Supervisors: Gorka Azkune Galparsoro and Oier López de Lacalle Lecuona.

