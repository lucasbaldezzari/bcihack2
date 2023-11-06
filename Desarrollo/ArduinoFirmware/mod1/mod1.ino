/******************************************************************
            VERSIÓN FMR-001 Rev A
******************************************************************/

#include "definiciones.h"
#include "inicializaciones.h"
#include<SoftwareSerial.h>

/******************************************************************
  Declaración de variables para comunicación
/******************************************************************/

char uno = 1;
byte backMensaje = 0b00000000; 

const char inBuffDataFromPC = 2; //máximo valor del buffer de entrada
unsigned char incDataFromPC[2]; //variable para almacenar datos provenientes de la PC
char bufferIndex = 0;
bool sendDataFlag = 0;
bool newMessage = false;

unsigned char internalStatus[4]; //variable para enviar datos a la PC
char internalStatusBuff = 4;

unsigned char outputDataToRobot[4]; //variable para enviar datos al robot
char buffOutDataRobotSize = 4;
char buffOutDataRobotIndex = 0;

unsigned char incDataFromRobot[4]; //variable para recibir datos del robot
char incDataFromRobotSize = 4;
char incDataFromRobotIndex = 0;

/******************************************************************
  Variables para el control de flujo de programa
******************************************************************/
char sessionState = 0; //Sesión sin iniciar
char LEDTesteo = 13; //led de testeo
char LEDComando = 11;

/******************************************************************
  Declaración de variables para control de estímulos
******************************************************************/

int frecTimer = 5000; //en Hz. Frecuencia de interrupción del timer.

/******************************************************************
  Variables control de movimiento
******************************************************************/

char movimiento = 0; //Robot en STOP
int estado = 0;

SoftwareSerial BTMaestro(2,3); //TX, RX

//FUNCION SETUP
void setup()
{
  noInterrupts();//Deshabilito todas las interrupciones
  pinMode(LEDTesteo,OUTPUT);
  pinMode(LEDComando,OUTPUT);
  digitalWrite(LEDTesteo,0);
  digitalWrite(LEDComando,0);  
  Serial.begin(19200); //iniciamos comunicación serie
  // iniTimer2(); //inicio timer 2
  // BTMaestro.begin(9600);//comunicacion al BT
  delay(500);
  interrupts();//Habilito las interrupciones
}

void loop(){}

ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer0.
{
  // if(BTMaestro.available()) //Si tenemos un mensaje por bluetooth lo leemos
  // {
  //   // backMensaje = BTMaestro.read();
  //   //checkBTMessage(backMensaje);
  //   estado = !estado;
  //   digitalWrite(LEDTesteo,estado);
  // }
};

void serialEvent()
{
if (Serial.available() > 0) 
  {
    char val = char(Serial.read()) - '0';
    checkMessage(val); //chequeamos mensaje entrante    
    backMensaje = val;    
    Serial.println(backMensaje);
  }
};

void checkMessage(byte val)
/*
Función para chequear el mensaje recibido.
Desde la PC se reciben dos Bytes.
- El primer byte indica el estado de la sesión, donde 0 indica STOP y 1 indica Running
- El segundo byte indica el comando a ejecutar. Este comando puede tomar desde el valor 0 al 5.
El comando a realizar en base al número se especifica en el Módulo 2.
*/
{
  incDataFromPC[bufferIndex] = val;
  switch(bufferIndex)
  {
    case 0:
      if (incDataFromPC[bufferIndex] == 1) digitalWrite(LEDTesteo,0);
      else digitalWrite(LEDTesteo,0);
      break;
    case 1:
      if (incDataFromPC[bufferIndex] == 3) digitalWrite(LEDTesteo,1);
      else digitalWrite(LEDTesteo,0);
      
      break;    
  }

  bufferIndex++;

  if (bufferIndex >= inBuffDataFromPC) //hemos recibido todos los bytes desde la PC
  {
    // sendCommand(); //Si se alcanza el tamaño del buffer, se envían los datos a M2 por Bluetooth
    bufferIndex = 0;
  }
};

/*
Función: sendCommand()
- Se usa para enviar un comando al módulo 2
*/
void sendCommand()
{
    estado = !estado;
    // digitalWrite(LEDTesteo,estado);
    byte mensaje = (incDataFromPC[0])|(incDataFromPC[1]<<1)|(incDataFromPC[2]<<2);//Armamos el byte
    BTMaestro.write(mensaje); //enviamos byte por bluetooth
}