import Jetson.GPIO as GPIO
import time
import sys
import tty
import termios
import select
from pynput import keyboard

# 핀 설정
# 좌측 모터
ENA = 32  # PWM 핀 (좌측 속도)
IN1 = 35  # 방향
IN2 = 36

# 우측 모터
ENB = 33  # PWM 핀 (우측 속도)
IN3 = 15
IN4 = 16

# 초기화
GPIO.setmode(GPIO.BOARD), GPIO.setup([ENA, ENB], GPIO.OUT)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)

pwm_left = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)

pwm_left.start(0)
pwm_right.start(0)

key = None

def on_press(input):
    global key
    if key is None:
        key = input

def on_release(input):
    global key
    if input == key:
        key = None

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
    
def stop():
    print("Stop")
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

def forward():
    print("Forward")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(30)
    pwm_right.ChangeDutyCycle(30)

def backward():
    print("Backword")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(30)
    pwm_right.ChangeDutyCycle(30)

def left():
    print("Left")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(6)
    pwm_right.ChangeDutyCycle(30)

def right():
    print("Right")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(30)
    pwm_right.ChangeDutyCycle(6)

try:
    print("조작 키: w(전진) s(후진) a(좌회전) d(우회전) x(정지) q(종료)")
    while True:
        print(key)
        if key == None:
            stop()
        elif key.char == 'w':
            forward()
        elif key.char == 's':
            backward()
        elif key.char == 'a':
            left()
        elif key.char == 'd':
            right()
        elif key.char == 'q':
            break

finally:
    pwm_left.stop()
    pwm_right.stop()
    listener.stop()
    GPIO.cleanup()
    print("프로그램 종료")


