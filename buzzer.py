import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.output(20, False)
time.sleep(1)
#GPIO.output(20, False)
time.sleep(5)    
