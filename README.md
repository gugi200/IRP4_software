# IRP4_software
Author: Michael Gugala
University of Bristol MEng project 

The repository includes all the code created during Individual Research Project 4.


## Topic: Development of a pressure sensing system for object detection.

Abstract:
This study presents the development of a pressure-sensing array system with object recognition capabilities, focusing on components' accessibility, affordability, and fabrication simplicity. The sensor array with a sensing area of approximately 550 $cm^2$ consists of 576 nodes constructed using a piezoresistive material called Velostat, custom printed circuit board, copper tape, resistors, and other electronic components. Piezoresistive properties of the pressure-sensing material improved with the application of flexible polyurethane foam. Fabrication and operation require several software programs, including MCU control, GUI development, signal/image processing, and DL model training and evaluation tools. Various image processing techniques, such as exponential mapping and gamma correction, were evaluated to improve Deep Learning model prediction capabilities. A hyperparameter grid search identified the "mobilenet\_v3\_large" model architecture as the most suited to this application, achieving nearly 100\% accuracy on both test and train datasets while having minimal training time. The study demonstrates the feasibility of developing a cost-effective, high-accuracy pressure-based object detection system using easily available materials.


This repository contains the software used for this project. Software can be divided into 4 categories:
1) Arduino code
2) Image processing
3) Graphical User Interface
4) Deep Learning
5) Datasets

## Arduino Code
There are three Arduino codes:
* **fast_sensor_array** is a high sampling frequency, low precision data collection algorithm. It is not used in this application because precision is preferred over speed. The code was written using fast digital pin-driving code from Pololu Corporation. The algorithm uses low-level coding to achieve a sampling frequency of 1k Hz. The data from analog the biggest gain in speed comes from accessing the ADC registers directly. The code outputs values from 0 to 255.
```
  ADCSRA = 0;             // clear ADCSRA register
  ADCSRB = 0;             // clear ADCSRB register
  ADMUX |= (0 & 0x07);    // set A0 analog input pin
  ADMUX |= (1 << REFS0);  // set reference voltage
  ADMUX |= (1 << ADLAR);  // left align ADC value to 8 bits from ADCH register
  ADCSRA |= (1 << ADPS2); // 16 prescaler for 76.9 KHz
  ADCSRA |= (1 << ADATE); // enable auto trigger some boards have ADFR instead 
  ADCSRA |= (1 << ADEN);  // enable ADC
  ADCSRA |= (1 << ADSC);  // start ADC measurements
  ADCH                    // contains the value from the ADC chip
```

* **slow_sensor_read_v3** This high-precision, low-speed code outputs values between 0 and 1023, the sampling frequency is 10 Hz. It uses standard Arduino functions. The inner sampling loop iterates through columns (MUXs), the outer loop energizes the rows (1 to 24 3 8-bit shift registers). This algorithm is used in the sensor.


* **timing_v4** This high-precision, low-speed code outputs values between 0 and 1023, the sampling frequency is above 10 Hz. It uses standard Arduino functions. The inner sampling loop energizes the rows (1 to 24 3 8-bit shift registers), the outer loop iterates through columns (MUXs). Because MUXs need time to set this method archives higher sampling frequency as their selection address is changed only 4 times per full sample, whereas, '''slow_sensor_read_v3''' changes it 96 times.


## Image Processing
Once the dataset has been collected, it is processed the following codes are used for image processing its evaluation and visualisation. 

**data_manipulation.py** Consists of a few main functions that process the collected dataset.

```createExtendedDataset``` augments the datapoints to extend the dataset 8-fold

```createExtendedDataset``` splits the dataset into training and testing sets

```train_test_split_subset``` splits the dataset into training and testing sets to specific lengths


* **data_visualisation.py** Visualises the dataset and different image processing effects.

* **rmBadPics.py** The dataset is saved in 2 formats, as images and in JSON files (dictionaries). Thus to delete a bad datapoint this script deletes it from both formats.

* **compare_normalization.py** Trains google net models using different dataset normalization parameters and saves the results in JSON file.

* **compare_transormation_on_googlenet.py** Trains googlenet models using different image processing techniques and saves the results in json file.

* **display_DL_results.py** Loads the information from JSON files and displays it.
  
* **save_model_offline** From a saves a full model from a model's state dictionary.

* **trainLibTorch.py** Custom package library.


## Deep Learning
Having collected data from the full grid search performed in Google Colab using Weights and Biases, the models can be trained. Some models were also trained locally to speed up the grid search. 

* **googlenet_test.py**, **mobilenet_v3_large.py**, and **restneet_train_test.py** train specific models and save them in .pth files.
* **trainLibTorch.py** Custom package library.
* **visualise_model.py** Visualises the model in the form of confusion matrices for training and testing datasets and displays some of the predictions and corresponding images.

## Graphical User Interface
It runs a graphical user interface with a Deep Learning model. At the first stages of the project, the GUI was used to collect data. The button **save photo** stores the current data and displays it in a separate window for review. The user can then save or discard the data in an appropriate folder. However its primary task is to display the current output from Arduino and the prediction label which is provided by the model running in the background. There are 3 modes of prediction.

* **sensor_gui2.py** The GUI runs the "simple prediction", where prediction is computed on the latest data available once every GUI refresh. It is targeted at Raspberry Pi and applications that require a higher refresh. The Raspberry Pi used for this project has a preloaded, trained full mobilenet_v3_large model to allow for offline operation. However, if ```self.online``` is set to ```True``` the models are downloaded online, two options are available: mobilenet_v3_large and googlenet. Users can also pick between 6 class and 10 class models. ```self.model_path``` specifies the path for the model in the online mode.
```self.online = True
      self.model_path = "mobilenet_v3_large_test_6_classes.pth"
      self.class_names = ['big_fizzy', 'h_big_bottle','h_bottle', 'hand', 'mug', 'small_fizzy']
      # self.class_names = ['big_fizzy', 'can', 'h_big_bottle','h_bottle', 'hand', 'mug', 'nothing', 'small_fizzy']
```

* **pytorch_snesor_V2.py** The GUI has two modes of operation
1) Hard cumulative prediction - A class is predicted every time new data is available. Before a display refresh, the most common prediction is selected and displayed. Selected on by setting ```self.MODE_avergae_prediction = True```

2) Soft cumulative prediction - The probability of every class is computed every time new data is available. Before a display refresh prediction vectors are summed up, the class corresponding to the highest summed prediction value is selected and displayed. Selected on by setting ```self.MODE_avergae_prediction = False```


The users have also an option between 6 and 10 class model but since the two modes are not intended for Raspberry Pi only the online mode is available. ```self.model_path``` specifies the path for the model
```
      self.model_path = "mobilenet_v3_large_test_6_classes.pth"
      self.class_names = ['big_fizzy', 'h_big_bottle','h_bottle', 'hand', 'mug', 'small_fizzy']
      # self.class_names = ['big_fizzy', 'can', 'h_big_bottle','h_bottle', 'hand', 'mug', 'nothing', 'small_fizzy']
```
* **trainLibTorch.py** Custom package library.
* **mobilenet_v3_large_test_10_classes.pth** and **mobilenet_v3_large_test_10_classes.pth** are saved wights of the trained model
* **mobilenet_6_classes.pth** and **mobilenet_10_classes.pth** are the full model paths for the offline mode.
The README file contains instructions on how to run the software on a laptop.


##

There are also Jupiter notebooks present as much of the tests on the model training, testing and results visualisation were carried out in Google Colab.



