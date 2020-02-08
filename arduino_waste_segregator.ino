#include<String.h>
#include <Servo.h>

Servo myservo; 

int mid  = 90;
int bio  = 135;
int nbio = 45;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  myservo.attach(9);
  myservo.write(mid);
  delay(100);
  myservo.detach();
  pinMode(LED_BUILTIN, OUTPUT);
}
int flag=0;
int i=0;
void loop() {
  // put your main code here, to run repeatedly:
  char c;
  if(Serial.available()){
    c = Serial.read();
//    if(c=='4'){
//      digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
//      delay(1000);                       // wait for a second
//      digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//      Serial.write("MSG Received");
//    }
   if(c=='b'){
      if(digitalRead(3)){
        Serial.write("FULL");
        Serial.write('\n');
      }
      else{
        Serial.write("NOT FULL");
        Serial.write('\n');
      }
      myservo.attach(9);
      for(i=mid;i<=bio;i++){
        myservo.write(i);  
        delay(20);
      }
      delay(1000);
      for(i=bio;i>mid;i--){
        myservo.write(i);  
        delay(20);
      }
      delay(1000);
      myservo.detach();
   }
   else if(c=='n'){
      if(digitalRead(3)){
        Serial.write("FULL");
        Serial.write('\n');
      }
      else{
        Serial.write("NOT FULL");
        Serial.write('\n');
      }
      myservo.attach(9);
      for(i=mid;i>nbio;i--){
        myservo.write(i);  
        delay(20);
      }
      delay(1000);
      for(i=nbio;i<mid;i++){
        myservo.write(i);  
        delay(20);
      }
      delay(1000);
      myservo.detach();
      
   }

    
    Serial.write('\n');
  }

  
  
}
