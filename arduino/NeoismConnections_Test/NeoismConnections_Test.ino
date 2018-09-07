
// Number of pins.
#define N_PINS 64

// Bitwise macros for unsigned long long ints (64 bits).
#define bitRead64(value, bit) (((value) >> (bit)) & 0x01ULL)
#define bitSet64(value, bit) ((value) |= (1ULL << (bit)))
#define bitClear64(value, bit) ((value) &= ~(1ULL << (bit)))
#define bitWrite64(value, bit, bitvalue) (bitvalue ? bitSet64(value, bit) : bitClear64(value, bit))

#define DISCONNECTED 0ULL

// Type to describe the connective state of one specific pin to the 64 others.
typedef uint64_t connections_t;

// Total: 64 pins
uint8_t PINS[] = {    2,  3,  4,  5,  6,
                      
                      7,  8,  9, 10, 11,
                     12, 14, 15, 16,     17, 18, 19, 20,

                     21, 22, 23, 24,                         25, 26, 27, 28,     29, 30, 31, 32,                     
                     33, 34, 35, 36,     37, 38, 39, 40,                         41, 42, 43, 44,

                     45, 46, 47, 48,                         49, 50, 51, 52,     53, A0, A1, A2,
                     A3, A4, A5,         A6, A7, A8, A9,

                    A10,A11,A12
                 };

// uint8_t PINS[] = {  O01, O02, O03, O04, O05, O06,
//                     I01, I02, I03, I04, I05, I06,
//                     O11, O12, O13, O14, O15, O16, O17, O08,
//                     I11, I12, I13, I14, I15, I16, I17, I08
//                   };

// Return the connection state of pinId.
connections_t getConnections(int pinId) {
  int pin = PINS[pinId];
  
  // Set pin as output.
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);

  connections_t connections = 0UL;

  // Set all other pins as input with internal pull-up and check state accordingly.
  for (int i=0; i<N_PINS; i++) {
    bool connected;
    if (i != pinId) {
      pinMode(PINS[i], INPUT_PULLUP);
      connected = (digitalRead(PINS[i]) == LOW);
    }
    else {
      connected = false;
    }

    bitWrite64(connections, i, connected);
  }

  // Set pin back to zero-state.
  pinMode(pin, INPUT_PULLUP);

  return connections;
}

// Prints one connective state (verbose)
void printConnections(connections_t connections, bool verbose=false) {
  for (int j=0; j<N_PINS; j++) {
    bool val = bitRead64(connections, j);
    if (verbose) {
      Serial.print(val ? "X" : "O"); Serial.print(" ");
    } else {
      if (val)
        Serial.print(PINS[j]); Serial.print(" ");        
    }
  }
  Serial.println();
}

void setup() {
  Serial.begin(9600);

  // Initialize all pins as input pull-up.
  for (int i=0; i<N_PINS; i++) {
    pinMode(PINS[i], INPUT_PULLUP);
  }
}

void loop() {
  // Go through all pins : check and print connection state.
  for (int i=0; i<N_PINS; i++) {
    connections_t connections = getConnections(i);
    if (connections != DISCONNECTED) {
      Serial.print("Pin #"); Serial.print(PINS[i]); Serial.print(" : ");
      printConnections(connections);
    }
  }
  Serial.println("===============");
}
