// Scrolling Neoist Banner

final int TEXT_FONT_SIZE = 128;

final String TEXT_FILE = "banner.txt";
//final String TEXT_FILE = "test.txt";

final int BANNER_HEIGHT = 240;
final int BANNER_TEXT_HEIGHT = BANNER_HEIGHT/2;

final int INTERVAL_BETWEEN_LINES = 15000;

// An array of news headlines
String fullText = "";

String[] loadedLines;
float[] loadedLinesWidth;

PFont f; // Global font variable
float x; // Horizontal location
int currentChar = 0;
int currentLine = 0;
int finishedFrame;
int waitStartTime = 0;

void setup() {
  fullScreen(P2D);
  noCursor();
//  f = createFont("Arial", TEXT_FONT_SIZE);
  f = createFont("Lucida", TEXT_FONT_SIZE);
  
  loadedLines = loadStrings(TEXT_FILE);
  loadedLinesWidth = new float[loadedLines.length];

  // Display headline at x location
  textFont(f, TEXT_FONT_SIZE);
  textAlign(LEFT, CENTER);

  for (int i=0; i<loadedLines.length; i++)
    loadedLinesWidth[i] = textWidth(loadedLines[i]);
    
  // Initialize headline offscreen
  x = width;
  
  initText();
}

void draw() {
  background(0);
  fill(255);

  fill(255, 0, 0);
  rect(0, 0, width, BANNER_HEIGHT);

  // A specific String from the array is displayed according to the value of the "index" variable.
  if (millis() - waitStartTime > INTERVAL_BETWEEN_LINES) {
    
    fill(255);
    text(fullText, x, BANNER_TEXT_HEIGHT);
      
    //fill(255);
    //text(mouseY, 600, 700);
    
    x -= 15;
  
    // Decrement x
    updateText();
  }
}

boolean textFinished() {
  return (loadedLinesWidth[currentLine] + x < 0);
  //if (x > 0)
  //  return false;
  //loadPixels();
  //for (int i=0; i<BANNER_HEIGHT*width; i++)
  //  if (pixels[i] == color(255)) // character
  //    return false;
  //updatePixels();
  //return true;
}

void initText() {
  fullText = loadedLines[currentLine].toUpperCase();
  x = width;
  finishedFrame = frameCount;
  waitStartTime = millis() - INTERVAL_BETWEEN_LINES;
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
