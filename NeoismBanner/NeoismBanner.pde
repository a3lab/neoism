// Example 17-3: Scrolling headlines 

import org.multiply.processing.RandomTimedEventGenerator;

RandomTimedEventGenerator nextCharacterGen;

final int TEXT_FONT_SIZE = 128;

//final String TEXT_FILE = "banner.txt";
final String TEXT_FILE = "test.txt";

// An array of news headlines
String fullText = "";

String[] loadedLines;

PFont f; // Global font variable
float x; // Horizontal location
int currentChar = 0;
int currentLine = 0;
int finishedFrame;
int waitStartTime = 0;

void setup() {
//  fullScreen(P2D);
  noCursor();
  size(1080, 320);
  f = createFont("Arial", TEXT_FONT_SIZE);
  
  loadedLines = loadStrings(TEXT_FILE);

  // Initialize headline offscreen
  x = width;
  
  initText();
  
  //nextCharacterGen = new RandomTimedEventGenerator(this, "generateNextChar");
  //nextCharacterGen.setMinIntervalMs(50);
  //nextCharacterGen.setMaxIntervalMs(100);
}

void draw() {
  background(0);
  fill(255);

  // Display headline at x location
  textFont(f, TEXT_FONT_SIZE);
  textAlign(LEFT, CENTER);

  fill(255, 0, 0);
  rect(0, 0, width, 240);

  // A specific String from the array is displayed according to the value of the "index" variable.
  if (millis() - waitStartTime > 5000) {
    
    fill(255);
    text(fullText, x, 110);
      
    //fill(255);
    //text(mouseY, 600, 700);
    
    x -= 10;
  
    // Decrement x
    updateText();
  }
}

boolean textFinished() {
  if (x > 0)
    return false;
  loadPixels();
  for (int i=0; i<pixels.length; i++)
    if (pixels[i] == color(255)) // character
      return false;
  updatePixels();
  return true;
}

void initText() {
  fullText = loadedLines[currentLine].toUpperCase();
  x = width;
  finishedFrame = frameCount;
  waitStartTime = millis();
}

void updateText() {
  if (textFinished()) {
    println("text finished : " + frameCount);
    finishedFrame = frameCount;
    currentLine = (currentLine + 1) % loadedLines.length;
    fullText = loadedLines[currentLine].toUpperCase();
    x = width;
    waitStartTime = millis();
  }
}

String loadString(String uri) {
  String str = "";
  String[] allStrings = loadStrings(uri);
  for (String s : allStrings) {
    str += s;
  }
  return str;
}
