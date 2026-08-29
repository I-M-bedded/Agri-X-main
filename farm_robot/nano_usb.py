import serial, glob, time

port = glob.glob('/dev/serial/by-id/*')[0]
ser = serial.Serial(port, 115200, timeout=1)

time.sleep(2)              # DTR 리셋으로 Nano 재부팅됨 -> 대기 필수
ser.reset_input_buffer()

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if not line:
        continue
    if line == '1':
        print('OBJECT DETECTED')
    elif line == 'empty':
        print('EMPTY')
