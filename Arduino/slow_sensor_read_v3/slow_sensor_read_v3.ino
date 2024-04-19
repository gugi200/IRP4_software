
#define MAPPING_ENABLED 1
#define EXP_COEFF 15

const int SREG_latch_pin = 7; // shift register latch
const int SREG_clock_pin = 8;  // shift register clock
const int SREG_data_pin = 9;  // shift register data

const int SREG_reset_pin = 10;

const int MUXA_pin = 4; // MUX selsct pin 0
const int MUXB_pin = 5; // MUX selsct pin 1


byte data[48];
byte* dataPtr;
uint8_t endSeq[4] = {255, 0, 255, 0};

int baseData[24*24] = {0};

void callibrate(void);
void readArray(byte row);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(SREG_latch_pin, OUTPUT);
  pinMode(SREG_clock_pin, OUTPUT);
  pinMode(SREG_data_pin, OUTPUT);
  pinMode(SREG_reset_pin, OUTPUT);

  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);

  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(A5, INPUT);

  digitalWrite(SREG_latch_pin, LOW);
  digitalWrite(SREG_clock_pin, LOW);
  digitalWrite(SREG_data_pin, LOW);
  digitalWrite(SREG_reset_pin, HIGH);


  pinMode(MUXA_pin, OUTPUT);
  pinMode(MUXB_pin, OUTPUT);

  digitalWrite(MUXA_pin, LOW);
  digitalWrite(MUXB_pin, LOW);

  dataPtr = &data[0];
  callibrate();
  

  
}

void loop() {
  
  
  // put your main code here, to run repeatedly:  
 
  for (byte row=0; row<24; row++){
    Serial.write(row);
    if (row==0){
      digitalWrite(SREG_data_pin, HIGH);    // feed data into the serial shift register
      digitalWrite(SREG_clock_pin, HIGH);    // consume that data
      digitalWrite(SREG_clock_pin, LOW);    // consume that data
      digitalWrite(SREG_latch_pin, HIGH);   // display that data
      digitalWrite(SREG_latch_pin, LOW);
      digitalWrite(SREG_data_pin, LOW);    // feed data into the serial shift register
    }
    // delay(25);
    readArray(row);
    digitalWrite(SREG_clock_pin, HIGH);
    digitalWrite(SREG_latch_pin, HIGH);
    
    // delay(25);
    digitalWrite(SREG_clock_pin, LOW);
    digitalWrite(SREG_latch_pin, LOW); 
    // delay(25);
    for (byte i=0; i<48; i++){
      Serial.write(*(dataPtr + i)); // it will only send a byte or an array of bytes, to send 1024 
    }

    Serial.write(endSeq, 4);

  }  
}

void readArray(byte row){
  for (byte col=0; col<4; col++){

/*
col = 0 - 0x00 (0000 0000)
col = 1 - 0x01 (0000 0001)
col = 2 - 0x02 (0000 0010)
col = 3 - 0x03 (0000 0011)
*/


   digitalWrite(MUXA_pin, (uint8_t) col&0x01); // set MUX
   digitalWrite(MUXB_pin, (uint8_t) ((col&0x02)>>1)); // set MUX
   delay(4); // delay for the MUXs to activate


  int readPin;
    readPin = analogRead(A0) - baseData[(row*24) + (col*6)]; 
    if (readPin<0) readPin = 0;
    if (MAPPING_ENABLED==1) readPin = (int) (  exp( -( EXP_COEFF/(readPin+1) ) )*1023  );
    *(dataPtr + 2*(col*6)) = (byte) (readPin & 0x0F); 
    *(dataPtr + ( 2*(col*6) ) + 1) = (byte) ((readPin>>8) & 0x0F);

    readPin = analogRead(A1) - baseData[(row*24) + (col*6) + 1]; 
    if (readPin<0) readPin = 0;
    if (MAPPING_ENABLED==1) readPin = (int) (  exp( -( EXP_COEFF/(readPin+1) ) )*1023  );
    *(dataPtr + 2*((col*6) + 1)) = (byte) (readPin & 0x0F);
    *(dataPtr + ( 2*((col*6) + 1) ) + 1) = (byte) ((readPin>>8) & 0x0F);


    readPin = analogRead(A2) - baseData[(row*24) + (col*6) + 2];
    if (readPin<0) readPin = 0;
    if (MAPPING_ENABLED) readPin = (int) (  exp( -( EXP_COEFF/(readPin+1) ) )*1023  );
    *(dataPtr + 2*((col*6) + 2)) = (byte) (readPin & 0x0F);
    *(dataPtr + ( 2*((col*6) + 2) ) + 1) = (byte) ((readPin>>8) & 0x0F);

    readPin = analogRead(A3) - baseData[(row*24) + (col*6) + 3];
    if (readPin<0) readPin = 0;
    if (MAPPING_ENABLED) readPin = (int) (  exp( -( EXP_COEFF/(readPin+1) ) )*1023  );
    *(dataPtr + 2*((col*6) + 3)) = (byte) (readPin & 0x0F);
    *(dataPtr + ( 2*((col*6) + 3) ) + 1) = (byte) ((readPin>>8) & 0x0F);
 

    readPin = analogRead(A4) - baseData[(row*24) + (col*6) + 4]; 
    if (readPin<0) readPin = 0;
    if (MAPPING_ENABLED) readPin = (int) (  exp( -( EXP_COEFF/(readPin+1) ) )*1023  );
    *(dataPtr + 2*((col*6) + 4)) = (byte) (readPin & 0x0F); 
    *(dataPtr + ( 2*((col*6) + 4) ) + 1) =(byte) ((readPin>>8) & 0x0F);


    readPin = analogRead(A5) - baseData[(row*24) + (col*6) + 5];
    if (readPin<0) readPin = 0;
    if (MAPPING_ENABLED) readPin = (int) (  exp( -( EXP_COEFF/(readPin+1) ) )*1023  );
    *(dataPtr + 2*((col*6) + 5)) = (byte) (readPin & 0x0F); 
    *(dataPtr + ( 2*((col*6) + 5) ) + 1) =(byte) ((readPin>>8) & 0x0F);
  }
}


void callibrate(void){
  delay(3000);
  for(byte reps=1; reps<3; reps++){
      digitalWrite(SREG_data_pin, HIGH);    // feed data into the serial shift register
      digitalWrite(SREG_clock_pin, HIGH);    // consume that data
      digitalWrite(SREG_clock_pin, LOW);    // consume that data
      digitalWrite(SREG_latch_pin, HIGH);   // display that data
      digitalWrite(SREG_latch_pin, LOW);
      digitalWrite(SREG_data_pin, LOW);    // feed data into the serial shift register
      for (byte row=0; row<24; row++){
        for (byte col=0; col<4; col++){
         digitalWrite(MUXA_pin, (uint8_t) col&0x01); // set MUX
         digitalWrite(MUXB_pin, (uint8_t) ((col&0x02)>>1)); // set MUX
         delay(5);
         baseData[(row*24) + (col*6)] += analogRead(A0)>>1; 
         baseData[(row*24) + (col*6) + 1] += analogRead(A1)>>1; 
         baseData[(row*24) + (col*6) + 2] += analogRead(A2)>>1; 
         baseData[(row*24) + (col*6) + 3] += analogRead(A3)>>1; 
         baseData[(row*24) + (col*6) + 4] += analogRead(A4)>>1; 
         baseData[(row*24) + (col*6) + 5] += analogRead(A5)>>1;  
        }
        digitalWrite(SREG_clock_pin, HIGH);
        digitalWrite(SREG_latch_pin, HIGH);
        digitalWrite(SREG_clock_pin, LOW);
        digitalWrite(SREG_latch_pin, LOW); 
      }

      delay(1000);
    }
}