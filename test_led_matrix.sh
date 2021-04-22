#!/bin/bash
oscsend 192.168.1.110 7770 /neoism/text s neoism?
sleep 1
oscsend 192.168.1.110 7770 /neoism/color iii 255 0 0
sleep 1
oscsend 192.168.1.110 7770 /neoism/color iii 0 255 0
sleep 1
oscsend 192.168.1.110 7770 /neoism/color iii 0 0 255
sleep 1
oscsend 192.168.1.110 7770 /neoism/text s "        "
