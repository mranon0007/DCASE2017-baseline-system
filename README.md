Acoustic Scene Classification & Event Detection
=========================

Getting started
===============

1. Install requirements with command: ``pip install -r requirements.txt``
2. Install tensorflow GPU [Setup Instructions](https://www.tensorflow.org/install). 

System description
==================

The system contains of 2 different programs.  
- `applications/task1.py`: Scene Classifier.
- `applications/task2.py`: Event detector.

## Task1

#### Training
The scene classifer can be trained using the following:

- CNN             : `python applications/task1.py -s cnn_train`
- LSTM            : `python applications/task1.py -s lstm_train`
- CnnLstm Parallel: `python applications/task2.py -s cnn_lstm_parallel_train`

The number of training epochs can be set in `applications/parameters/task1.defaults.yaml` under `sets -> SET_NAME -> learner_method_parameters -> LEARNER_NAME -> epochs`.  

Default # of Epochs: `200`

#### Testing
Testing the classifier on evaluation dataset.

- CNN             : `python applications/task1.py -s cnn_test`
- LSTM            : `python applications/task1.py -s lstm_test`
- CnnLstm Parallel: `python applications/task1.py -s cnn_lstm_parallel_test`

To overwrite results, pass an additional argument `-o`.

## Task2

#### Training
The event detector can be trained using the following:

- CNN             : `python applications/task2.py -s cnn_train`
- LSTM            : `python applications/task2.py -s lstm_train`
- CnnLstm Parallel: `python applications/task2.py -s cnn_lstm_parallel_train`

The number of training epochs can be set in `applications/parameters/task2.defaults.yaml` under `sets -> SET_NAME -> learner_method_parameters -> LEARNER_NAME -> epochs`.  

Default # of Epochs: `200`

#### Testing
Testing the event detector on evaluation dataset.

- CNN             : `python applications/task1.py -s cnn_test`
- LSTM            : `python applications/task1.py -s lstm_test`
- CnnLstm Parallel: `python applications/task2.py -s cnn_lstm_parallel_test`

To overwrite results, pass an additional argument `-o`.