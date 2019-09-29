/**
 * oscP5plug by andreas schlegel
 * example shows how to use the plug service with oscP5.
 * the concept of the plug service is, that you can
 * register methods in your sketch to which incoming 
 * osc messages will be forwareded automatically without 
 * having to parse them in the oscEvent method.
 * that a look at the example below to get an understanding
 * of how plug works.
 * oscP5 website at http://www.sojamo.de/oscP5
 */

import oscP5.*;
import netP5.*;

OscP5 oscP5;

final int RECV_PORT = 7770;

final int N_LED_MATRIX_CHARS = 7;

final int LED_MATRIX_WIDTH  = 64;
final int LED_MATRIX_HEIGHT = 16;

String text = "NEOISM?";
color textColor = color(255);

PFont font;

void setup() {
  size(1280, 320);
  
  frameRate(25);
  /* start oscP5, listening for incoming messages at port 12000 */
  oscP5 = new OscP5(this, RECV_PORT);
  
  /* osc plug service
   * osc messages with a specific address pattern can be automatically
   * forwarded to a specific method of an object. in this example 
   * a message with address pattern /test will be forwarded to a method
   * test(). below the method test takes 2 arguments - 2 ints. therefore each
   * message with address pattern /test and typetag ii will be forwarded to
   * the method test(int theA, int theB)
   */
  oscP5.plug(this, "receiveText", "/neoism/text");
  oscP5.plug(this, "changeColor", "/neoism/color");
  
  int fontSize = int(height*0.9);
  font = createFont("Courier", fontSize);
  textAlign(CENTER, CENTER);
//  textFont(font);
  textSize(fontSize);
}


public void receiveText(String str) {
  text += str.toUpperCase();
}

public void changeColor(int r, int g, int b) {
  textColor = color(r, g, b);
}

void draw() {
  background(0);
  
  fill(textColor);
//  println("["+text.substring(text.length()-N_LED_MATRIX_CHARS)+"]");
  text(text.substring(text.length()-N_LED_MATRIX_CHARS), width/2, height/2);
//  text(text.substring(N_LED_MATRIX_CHARS), 0, 0);
}

/* incoming osc message are forwarded to the oscEvent method. */
void oscEvent(OscMessage theOscMessage) {
  /* with theOscMessage.isPlugged() you check if the osc message has already been
   * forwarded to a plugged method. if theOscMessage.isPlugged()==true, it has already 
   * been forwared to another method in your sketch. theOscMessage.isPlugged() can 
   * be used for double posting but is not required.
  */  
  if(theOscMessage.isPlugged()==false) {
  /* print the address pattern and the typetag of the received OscMessage */
  println("### received an osc message.");
  println("### addrpattern\t"+theOscMessage.addrPattern());
  println("### typetag\t"+theOscMessage.typetag());
  }
}
