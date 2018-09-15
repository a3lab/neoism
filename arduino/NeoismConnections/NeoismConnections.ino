// Include AsciiMassagePacker.h for ASCII format massage packing.
#include <AsciiMassagePacker.h>

// Instantiate an AsciiMassagePacker for packing massages.
AsciiMassagePacker outbound;

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
// Reference: https://docs.google.com/spreadsheets/d/1z8Q2Ge5YN9jgCZAnuwXoPhfJdWPdnO3NLYzQ2Bg6Gg8/edit?ts=5b99dae7
uint8_t PINS[] = {    13,  12,  11,           // 3[1-3]!
                      10,   9,   8,           // 3[1-3]?
                      7,    6,   5,   4,      // 3[4-7]?
                      14,  15,  16,  17,      // 3[4-7]!
                      18,  19,  20,  21,      // 2[1-4]!
                      24,  22,  25,  23,      // 2[5-8]!
                      27,  26,  30,  31,      // 2[1-4]?
                       3,   2,  33,  32,      // 2[5-8]?
                      35,  34,  37,  36,      // 2[9-C]?
                      38,  39,  40,  41,      // 2[9-C]!
                      53,  52,  50,  51,      // 1[1-4]!
                      49,  48,  47,  46,      // 1[5-8]!
                      43,  42,  45,  44,      // 1[1-4]?
                     A13, A12, A14, A15,      // 1[5-8]?
                      A0,  A1,  A2,  A3,  A4, // 0[1-5]!
                      A7,  A8,  A9, A10, A11  // 0[1-5]?
                  };

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
  // Reset state.
  outbound.beginPacket("/reset");
  outbound.streamPacket(&Serial);
  
  // Go through all pins : check and print connection state.
  for (int i=0; i<N_PINS; i++) {
    connections_t connections = getConnections(i);
    if (connections != DISCONNECTED) {
      for (int j=0; j<N_PINS; j++) {
        bool val = bitRead64(connections, j);
        if (val) {
          outbound.beginPacket("/conn");
          outbound.addInt(i);
          outbound.addInt(j);
          outbound.streamPacket(&Serial);
        }
      }
    }
  }

  // Reset state.
  outbound.beginPacket("/done");
  outbound.streamPacket(&Serial);
}
