// Example 17-3: Scrolling headlines 

import org.multiply.processing.RandomTimedEventGenerator;

RandomTimedEventGenerator nextCharacterGen;

final int TEXT_FONT_SIZE = 128;

final String TEXT_FILE = "/home/tats/Documents/workspace/readings_neoism/data/neoism_tiny.txt";

// An array of news headlines
String fullText = "";

String loadedText;

PFont f; // Global font variable
float x; // Horizontal location
int currentChar = 0;

void setup() {
  fullScreen(P2D);
  noCursor();
//  size(1080, 320);
  f = createFont("Arial", TEXT_FONT_SIZE);
  
  loadedText = loadString(TEXT_FILE).toUpperCase();
  fullText = loadedText.substring(0, 3000);

  // Initialize headline offscreen
  x = width;
  
  //nextCharacterGen = new RandomTimedEventGenerator(this, "generateNextChar");
  //nextCharacterGen.setMinIntervalMs(50);
  //nextCharacterGen.setMaxIntervalMs(100);
}

void draw() {
  background(255, 0, 0);
  fill(255);

  // Display headline at x location
  textFont(f, TEXT_FONT_SIZE);
  textAlign(LEFT, CENTER);

  // A specific String from the array is displayed according to the value of the "index" variable.
  text(fullText, x, 275);
  
  fill(0);
  final int BOTTOM_BANNER = 433;
  rect(0, BOTTOM_BANNER, width, height-BOTTOM_BANNER);
  
  //fill(255);
  //text(mouseY, 600, 700);

  x -= 10;
  // Decrement x
}

String loadString(String uri) {
  String str = "";
  String[] allStrings = loadStrings(uri);
  for (String s : allStrings) {
    str += s;
  }
  return str;
}

void generateNextChar() {
  receiveText(loadedText.substring(currentChar, currentChar+1));
  currentChar = (currentChar + 1) % loadedText.length();
}

void receiveText(String txt) {
  fullText += txt;
  println(fullText);
//  x -= textWidth(txt);
}
