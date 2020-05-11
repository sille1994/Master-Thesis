#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from os import walk


cwd = os.getcwd()
print(cwd)


class DataSet(object):

  def __init__(self, signal, labels):
    """
    Construct a DataSet.
    By using the “self” keyword we can access the attributes and methods of the class in python.
    It binds the attributes with the given arguments. 
    
    Input
    Signal: -> dtp.float, (N,)
    Label: -> dtp.float, (N,)
    
    Return
    Dataset: -> Class
    """

    self._signal = signal
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def signal(self):
    return self._signal

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._signal = self._signal[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._signal[start:end], self._labels[start:end]

def is_whole(n):
    return n % 1 == 0

def pars_data(filepath, width=512):
    """
    This function parse the data into the wanted length
    
    Input:
    Filepath: -> string, tells were the files are.
    Width: -> dtp.float, tells how the signal should be broken up
    
    Return: 
    Data: -> dtp.float, (num_signals, rows, cols, 1)
    Labels: -> dtp.float, (num_label, rows, cols, 1)

    """
    
    # Find all the filenames with .npy which does not gave label in it too.
    filenames_signal = []
    for filename in os.listdir(str(filepath + '/')):
        if 'label' in filename:
            continue
        elif '.npy' in filename:
            filenames_signal.append(filename)
        else:  
            continue
    
    # Create a place to store the batches of the signals and the labels
    data = np.zeros((width,))
    data = np.append([data], [data], axis=0)

    labels = np.zeros((width,))
    labels = np.append([labels], [labels], axis=0)

    
    for filename in filenames_signal:
        print(filename)
        
        # Load the signals and the labels
        signal = np.load(str(filepath + '/' + filename))
        label = np.load(str(filepath + '/' + filename[:-4] + '_label.npy'))
        
        # Check that the siganl and label are equal in length
        assert len(signal)==len(label)
        
        # Check if the width devided with the signal is a whole number
        if is_whole(len(signal)/width):
            
            number = int(len(signal)/width)
            
            # Take the batches of the signals
            for i in range((number*2+1)):
                batch_signal = signal[int(width/2)*i:int(width/2)*(i+2),]
                batch_label = label[int(width/2)*i:int(width/2)*(i+2),]
           
                data = np.append(data, [batch_signal], axis=0)
                labels = np.append(labels, [batch_label], axis=0)

                print(int(width/2)*i,int(width/2)*(i+2))

        else:
            # Take the batches of the signals if the width devided with the signal is not a whole number
            number = int(len(signal)/width)
            for i in range((number*2)):
                batch_signal = signal[int(width/2)*i:int(width/2)*(i+2),]
                batch_label = label[int(width/2)*i:int(width/2)*(i+2),]
                   
                print(int(width/2)*i,int(width/2)*(i+2))

                data = np.append(data, [batch_signal], axis=0)
                labels = np.append(labels, [batch_label], axis=0)
    
    # Delete the first two rows of zeros
    data = data[2:,:]
    labels = labels[2:,:]
    print(data.shape, labels.shape)
    return data, labels
    
    
    
    

def read_data_sets(filepath_train, filepath_valid, filepath_test):
    class DataSets(object):
        pass
    #  Creating a class which has the signals and labels as an attributes with the given arguments. 
    data_sets = DataSets()
  
    # Width of the batches we want of the signal. It is equal in size to the transformer network
    width = 512
    
    # Check if the filepath exits and then parse the data and put it into the class  data_sets
    if os.path.exists(filepath_train) == True :
        train_signal, train_labels = pars_data(filepath_train, width=width)
        data_sets.train = DataSet(train_signal, train_labels)
    
    # Check if the filepath exits and then parse the data and put it into the class  data_sets    
    if os.path.exists(filepath_valid) == True :
        validation_signal, validation_labels = pars_data(filepath_valid, width=width)        
        data_sets.validation = DataSet(validation_signal, validation_labels)
        
    # Check if the filepath exits and then parse the data and put it into the class  data_sets 
    if os.path.exists(filepath_test) == True :
        test_signal, test_labels = pars_data(filepath_test, width=width)        
        data_sets.test = DataSet(test_signal, test_labels)

    return data_sets

filepath_train = './Test_data'
filepath_valid = './Data/Val'
filepath_test = './Data/Test'

Dataset = read_data_sets(filepath_train, filepath_valid, filepath_test)

y_train = Dataset.train.signal
y_train






