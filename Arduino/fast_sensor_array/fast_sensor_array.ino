#include <FastGPIO.h>
//  #include <FastMap.h>


#define SHIFT_REG_FAST 1

const int SREG_latch_pin = 7; // shift register latch
const int SREG_clock_pin = 8;  // shift register clock
const int SREG_data_pin = 9;  // shift register data

const int MUX0_pin = 0; // MUX selsct pin 0
const int MUX1_pin = 0; // MUX selsct pin 1
const int MUX2_pin = 0; // MUX selsct pin 2

const int row_number = 24;  // numbers of rows in the sensor array
const int col_number = 24;  // numbers of columns in the sensor array
const int mux_number = 4;
const int adc_ch_num = 6;

const int baseSampleNum = 10;
const int dataLen = row_number * col_number;
byte data[dataLen]= {};
int baseData[dataLen]= {0};
byte baseDataByte[dataLen]= {0};
// byte data_subMean[dataLen] = {0};
// byte data_diff[dataLen] = {0};
// byte data_prev[dataLen] = {0};

void calculateBase(void);
// void process_subMean(void);
// void process_diff(void);
void sendData(void);
void regSlow_muxFast(void);
void muxSlow_regFast(void);

void setup() {
  Serial.begin(9600);

  // put your setup code here, to run once:
  pinMode(SREG_latch_pin, OUTPUT);
  pinMode(SREG_clock_pin, OUTPUT);
  pinMode(SREG_data_pin, OUTPUT);

  digitalWrite(SREG_latch_pin, HIGH);
  digitalWrite(SREG_clock_pin, HIGH);
  digitalWrite(SREG_data_pin, LOW);


  pinMode(MUX0_pin, OUTPUT);
  pinMode(MUX1_pin, OUTPUT);
  pinMode(MUX2_pin, OUTPUT);
  digitalWrite(MUX0_pin, LOW);
  digitalWrite(MUX1_pin, LOW);
  digitalWrite(MUX2_pin, LOW);

  // mapper.init(0, 1023, 1, 255);

  ADCSRA = 0;             // clear ADCSRA register
  ADCSRB = 0;             // clear ADCSRB register
  ADMUX |= (0 & 0x07);    // set A0 analog input pin
  ADMUX |= (1 << REFS0);  // set reference voltage
  ADMUX |= (1 << ADLAR);  // left align ADC value to 8 bits from ADCH register

  ADCSRA |= (1 << ADPS2); // 16 prescaler for 76.9 KHz
   
   
  ADCSRA |= (1 << ADATE); // enable auto trigger some boards have ADFR instead 
  ADCSRA |= (1 << ADEN);  // enable ADC
  ADCSRA |= (1 << ADSC);  // start ADC measurements



  delay(1000);

  calculateBase();
  
}

void loop() {
  long t0 = micros();

  Serial.println();
  muxSlow_regFast();
  //process_subMean();
  Serial.print("Time taken for a single loop"); Serial.println(micros()-t0);
  Serial.println();
  Serial.println();
  /*
#if (SHIFT_REG_FAST==1)
  regSlow_muxFast();
#else 
  muxSlow_regFast();  
#endif
  //process_subMean();*/
  sendData();
}

void regSlow_muxFast(){

  FastGPIO::Pin<SREG_latch_pin>::setOutput(HIGH); 
  delayMicroseconds(3);
  for (int row=0; row<row_number; row++){ // if 24 rows - 6 ADC channels then 6 4-to-1 MUX
    FastGPIO::Pin<SREG_latch_pin>::setOutput(LOW);
    delayMicroseconds(3);
    FastGPIO::Pin<SREG_clock_pin>::setOutput(LOW);
    delayMicroseconds(3);
    if (row==0) FastGPIO::Pin<SREG_data_pin>::setOutput(1); // will only set serail data to HIGH id the sequence is at the resets
    else FastGPIO::Pin<SREG_data_pin>::setOutput(0);
    delayMicroseconds(3);
    FastGPIO::Pin<SREG_clock_pin>::setOutput(HIGH);         // HIGH clock so the data shifts
    delayMicroseconds(3);
    FastGPIO::Pin<SREG_latch_pin>::setOutput(HIGH);         // latch HIGH so the data is outputed to the strips
    delayMicroseconds(3);

    for (int mux=0; mux<mux_number; mux++){ // 4 MUXs
      /*            ____________________________________
       *            ____________________________________
       *            ____________________________________
       *ADC channel 012345    012345    012345    012345
       *channels    ||||||    ||||||    |||||||   ||||||
       *MUX number     0         1         2         3
       */
      delayMicroseconds(3);
      FastGPIO::Pin<MUX0_pin>::setOutput(mux&0x01); // set extrenal MUX
      delayMicroseconds(3);
      FastGPIO::Pin<MUX1_pin>::setOutput(mux&0x02); // set extrenal MUX
      delayMicroseconds(3);                          // wait for the MUX
              
      ADMUX &= 0xE8;          // reset ADC channel
      ADMUX |= (0 & 0x07);    // set A0 analog input pin
      delayMicroseconds(3); // used to be 1
      data[(row*col_number) + (adc_ch_num*mux)] = ADCH - baseDataByte[(row*col_number) + (adc_ch_num*mux)];       // read analog value and store in the memory
      ADMUX &= 0xE8;          // reset ADC channel
      ADMUX |= (1 & 0x07);    // set A1 analog input pin
      delayMicroseconds(3);
      data[(row*col_number) + (adc_ch_num*mux)+1] = ADCH - baseDataByte[(row*col_number) + (adc_ch_num*mux)+1];     // read analog value and store in the memory
      ADMUX &= 0xE8;          // reset ADC channel
      ADMUX |= (2 & 0x07);    // set A2 analog input pin
      delayMicroseconds(3);
      data[(row*col_number) + (adc_ch_num*mux)+2] = ADCH - baseDataByte[(row*col_number) + (adc_ch_num*mux)+2];     // read analog value and store in the memory
      ADMUX &= 0xE8;          // reset ADC channel
      ADMUX |= (3 & 0x07);    // set A3 analog input pin
      delayMicroseconds(3);
      data[(row*col_number) + (adc_ch_num*mux)+3] = ADCH - baseDataByte[(row*col_number) + (adc_ch_num*mux)+3];     // read analog value and store in the memory
      ADMUX &= 0xE8;          // reset ADC channel
      ADMUX |= (4 & 0x07);    // set A4 analog input pin
      delayMicroseconds(3);
      data[(row*col_number) + (adc_ch_num*mux)+4] = ADCH - baseDataByte[(row*col_number) + (adc_ch_num*mux)+4];     // read analog value and store in the memory
      ADMUX &= 0xE8;          // reset ADC channel
      ADMUX |= (5 & 0x07);    // set A5 analog input pin
      delayMicroseconds(3);
      data[(row*col_number) + (adc_ch_num*mux)+5] = ADCH - baseDataByte[(row*col_number) + (adc_ch_num*mux)+5];     // read analog value and store in the memory
    }
  }
  
}


void muxSlow_regFast(void){
 
  for (int row=0; row<row_number; row++){
    byte muxSeq = row%8;
    // find row sequence and find the coralating bit value
    // row = 8, row%8=1 -> 001 bitRead(001,0) -> 1 set LSB of MUX select to 1
    FastGPIO::Pin<MUX0_pin>::setOutput(muxSeq&0x01); // set MUX
    FastGPIO::Pin<MUX1_pin>::setOutput(muxSeq&0x02); // set MUX
    FastGPIO::Pin<MUX2_pin>::setOutput(muxSeq&0x04); // set MUX
    
    ADMUX &= 0xE8;
    if (row<8) ADMUX |= (0 & 0x07);         // MUX0
    else if (row<16) ADMUX |= (1 & 0x07);   // MUX1       
    else ADMUX |= (2 & 0x07);                // MUX2
    
    FastGPIO::Pin<SREG_data_pin>::setOutput(HIGH);  // feed HIGH into serial data
    FastGPIO::Pin<SREG_clock_pin>::setOutput(LOW);  //
    FastGPIO::Pin<SREG_clock_pin>::setOutput(HIGH); // shift +1 to lock HIGH as the first output
    FastGPIO::Pin<SREG_data_pin>::setOutput(LOW);   // disable HIGH data 
    FastGPIO::Pin<SREG_latch_pin>::setOutput(HIGH); // ouput parallel data
    
    for (int col=0; col<col_number; col++){
      FastGPIO::Pin<SREG_clock_pin>::setOutput(LOW);
      delayMicroseconds(3);
      // data[(row*col_number) + col] = ADCH - (byte) baseData[(row*col_number) + col]; // read analog value and store in the memory
      data[(row*col_number) + col] = ADCH; // read analog value and store in the memory
      FastGPIO::Pin<SREG_clock_pin>::setOutput(HIGH); // +1 to shift register
    }
    
    FastGPIO::Pin<SREG_latch_pin>::setOutput(LOW); // stop outputting 
    
  }

}

void calculateBase(void){
  
  for (int collectTimer=0; collectTimer<baseSampleNum; collectTimer++){
#if (SHIFT_REG_FAST==1)
    regSlow_muxFast();
#else 
    muxSlow_regFast();  
#endif
    for (int i=0; i<dataLen; i++){
      baseData[i] += (int) data[i];
    }
    for (int i=0; i<dataLen; i++){
      baseData[i] /= baseSampleNum;
    }
  }
}

/*
void process_subMean(void){
  // this will be done while collecting data but I want to see which processing, if any, is the best
  for (int i=0; i<dataLen; i++){
      data_subMean[i] = data[i] -  baseDataByte[i];
    }
}*/
/*
void process_diff(void){
  for (int i=0; i<dataLen; i++){
      data_diff[i] = data[i] -  data_prev[i]; // im afaird of overflows
      data_prev[i] = data[i];
    }
}
*//*
void process_subMeanMean(void){
  // subtract the mean mean of the whole matrix i.e. calculate the mean for each sensor and then find the mean 
  // of all sensors and subtract it from actual reading
}*/

void sendData(void){
  for (byte row=0;row<24;row++){
    for (byte col=0; col<24; col++){
       Serial.print("row: ");Serial.print(row);Serial.print(", col: ");Serial.print(col);Serial.print(", val: ");Serial.println(data[(row*24)+col]);

    }
  }
}
 
