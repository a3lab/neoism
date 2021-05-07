Description
===========

This project contains the code to run the project
"The Sense of Neoism?! An Infinite Manifesto" by Sofian Audry & Monty Cantsin?

Credits
-------

Authors: Sofian Audry & Monty Cantsin?

* Deep Learning Programming: Sofian Audry
* Texts: Monty Cantsin?
* Hardware Design: Grégory Perrin
* Research: Eliza Bennett
* Technical Assistance: Matthew Loewen

Technical Notes
---------------

A few notes to setup the installation.

### Raspberry pi

#### LED matrix settings

Source: https://github.com/hzeller/rpi-rgb-led-matrix/

1. Disable audio: ```dtparam=audio=off``` in ```/boot/config.txt```

2. Disable some processes: ```sudo apt-get remove bluez bluez-firmware pi-bluetooth triggerhappy pigpio```

3. Disable snd_bcm2835 module:

```
cat <<EOF | sudo tee /etc/modprobe.d/blacklist-rgb-matrix.conf
blacklist snd_bcm2835
EOF

sudo update-initramfs -u
```

#### Startup

Add the following line to ```/etc/rc.local```:

```sudo bash /path/to/neoism/neoism_startup_pi.sh > neoism_startup.log 2> neoism_startup_log &```

### Ubuntu PC

1. Add the following line to ```/etc/rc.local``:
```
bash path/to/neoism/neoism_startup_pc.sh > neoism_startup.log 2> neoism_startup.log &
```

2. Edit ```/etc/default/grub``` to start in text mode (be sure to make a copy). Make the following modifications:

```
GRUB_CMDLINE_LINUX_DEFAULT="text"
GRUB_CMDLINE_LINUX="3"
GRUB_TERMINAL=console
```
(*) This is important because there seems to be a problem when running the machine in headless mode if GRUB is not set to start in text-only mode.

3. IMPORTANT: Run ```sudo update-grub``` to apply changes.


If the ```rc.local``` does not work, also run ```sudo crontab -e``` and add the following line to the file:

```@reboot /path/to/neoism/neoism_startup_crontab.sh```

### Video banner Raspberry Pi

Disable display of yellow electric bolt: add ```disable_warnings=1``` to ```/boot/config.txt```

#### Startup

Add the following line to ```/etc/xdg/lxsession/LXDE-pi/autostart```:

```/bin/bash /home/pi/neoism/neoism_startup_banner_pi.sh```

### Useful Info

| PC IP  | 192.168.1.100  |
| RPi IP  | 192.168.1.110  |
| RPi OSC Port  | 7770  |
| Arduino serial no | 7573530323135161F1D2 |
