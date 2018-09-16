#!/bin/bash
cd /home/pi/neoism/led_display
sudo ./neoism-scrolling-text  --led-gpio-mapping=adafruit-hat-pwm --led-rows=16 --led-cols=32 --led-chain=2 --led-brightness=80  --led-pwm-lsb-nanoseconds=250 -s 5 -f "./fonts/9x18B.bdf" --led-show-refresh --led-slowdown-gpio=2 -C 128,128,128 --led-scan-mode=0

