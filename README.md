# PeopleCounting

This project implements [Codebook](https://www.sciencedirect.com/science/article/pii/S1077201405000057) model to count people on a video. 

## Requirements

- Numpy
- OpenCV

## Usage

- Create a folder with your training samples. Add the path to this folder on `DIR_TRAIN` atribute on `codebook.py` file.
- Create a folder with you testing samples. Adds the path to this folder on `DIR_TEST` atribute on `codebook.py` file

You can change hyperparameters used on `Codebook` model on `codebook.py` file.

1. Train your codebooks running `train_codebooks.py` file.

`$ python train_codebooks.py`

2. Create your foreground/background image file running `test_codebooks.py` file.

`$ python test_codebooks.py`

3. Count the number of people on a specific file running `count_people.py` file. Change the constant `FILE_NAME` to the path of the foreground/background image you are using.

`$ python count_people.py`

## License

This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details
