# Final project: Analysing cards game.

In this project, we will deal with image data consisting of 7 games played with cards; each game contains 13 rounds. The goal of the project is to apply image analysis and pattern recognition methods to be able to output a score depending on the set of rules of the game.

For a detailled description, open project_final_shortened.ipynb


## Running the project with a new game:
To test the code on a new game you need to run the final-project-shortened.ipynb notebook

1. To get the data you need to change the path variables in the notebook and helpers.py on line 355
1. If you have more training data you need to change the size of the loops in helpers.py function train_model()
1. You will need the MNIST pretrained model which you can get by running the notebook MNIST.ipynb

## List of the packages:
1. OpenCV
1. Skimage
1. torch
1. matplotlib
1. numpy
1. pandas
1. xgboost
1. tqdm
1. sklearn
1. torchvision