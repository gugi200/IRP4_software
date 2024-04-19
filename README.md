# IRP4_software
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
```fast_sensor_array``` is a high sampling frequency, low precision data collection algorithm. It is not used in this application because precision is preferred over speed. The code was written using fast digital pin-driving code from Pololu Corporation. The algorithm uses low-level coding to achieve a sampling frequency of 1k Hz. The data from analog the biggest gain in speed comes from accessing the ADC registers directly. The code outputs values from 0 to 255.
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

```slow_sensor_read_v3``` This high-precision, low-speed code outputs values between 0 and 1023, the sampling frequency is 10 Hz. It uses standard Arduino functions. The inner sampling loop iterates through columns (MUXs), the outer loop energizes the rows (1 to 24 3 8-bit shift registers). This algorithm is used in the sensor


```timing_v4``` This high-precision, low-speed code outputs values between 0 and 1023, the sampling frequency is above 10 Hz. It uses standard Arduino functions. The inner sampling loop energizes the rows (1 to 24 3 8-bit shift registers), the outer loop iterates through columns (MUXs). Because MUXs need time to set this method archives higher sampling frequency as their selection address is changed only 4 times per full sample, whereas, '''slow_sensor_read_v3''' changes it 96 times.


## Image Processing

There are also Jupiter notebooks present as much of the tests on the model training, testing and results visualisation were carried out in Google Colab.



