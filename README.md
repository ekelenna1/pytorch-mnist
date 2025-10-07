MNIST Handwritten Digit Recognition with PyTorch
Project Overview
This project is a fundamental implementation of a neural network designed to classify handwritten digits from the well-known MNIST dataset. The goal is to build, train, and evaluate a simple yet effective model using the PyTorch deep learning framework. This serves as a core demonstration of understanding key machine learning concepts, including data processing, model architecture, training loops, and performance evaluation.

Stack
Language: Python

Core Library: PyTorch

Utilities: Torchvision (for dataset access and image transformations)

Project Structure
main.py: The main script containing the complete workflow: data loading, model definition, training, and evaluation.

requirements.txt: A list of the necessary Python packages to run the project.

/data: This directory will be created automatically upon first run to download and store the MNIST dataset.

How to Run
Clone the Repository:

git clone (https://github.com/ekelenna1/pytorch-mnist.git)
cd <therepodirectory>


Can also Set up a Virtual Environment (probrably Recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install Dependencies:

pip install -r requirements.txt


Execute the Script:

python main.py


Results
The script will first download the MNIST dataset if it's not already present. It will then proceed to train the model for 5 epochs, printing the training loss periodically. After training is complete, it will evaluate the model's performance on the unseen test set and print the final accuracy, which should be approximately 97-98%.

Starting training...
[Epoch 1, Batch 200] loss: 0.697
[Epoch 1, Batch 400] loss: 0.312
...
Finished Training
Accuracy of the network on the 10000 test images: 97.55 %

