import serial
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import threading



class App(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        self.root.title("Object detection")
        self.root.geometry('600x600')

        self.root.mainloop()

port = '/dev/ttyACM0'
baudrate = 115600
# Initialize serial port
ser = serial.Serial()
ser.port     = port
ser.baudrate = baudrate
ser.open()
ser.reset_input_buffer()

width = 320
height = 240
bytes_per_pixel = 2
bytes_per_frame = width * height * bytes_per_pixel

image = np.empty((height, width, bytes_per_pixel), dtype=np.uint8)

def serial_readline():
    data = ser.readline()
    return data.decode("utf-8").strip()

def rgb565_to_rgb888(val):
    r = ((val[0] >> 3) & 0x1f) << 3
    g = (((val[1] >> 5) & 0x07) | ((val[0] << 3) & 0x38)) << 2
    b = (val[1] & 0x1f) << 3
    rgb = np.array([r,g,b], dtype=np.uint8)

    return rgb

app = App()

while True:

    data_str = serial_readline()

    if str(data_str) == "<image>":
        print("Reading frame")
        data = ser.read(bytes_per_frame)
        img = np.frombuffer(data, dtype=np.uint8)
        img_rgb565 = img.reshape((height, width, bytes_per_pixel))
        img_rgb888 = np.empty((height, width, 3), dtype=np.uint8)

        for y in range(0, height):
            for x in range(0, width):
                img_rgb888[y][x] = rgb565_to_rgb888(img_rgb565[y][x])
        
        data_str = serial_readline()
        if(str(data_str) == "</image>"):
            print("Captured frame")
            image_pil = Image.fromarray(img_rgb888)
            image_pil.save("out.bmp")
            test = ImageTk.PhotoImage(image_pil)
            label = Label(app.root, image=test)
            label.image = test
            label.place(x=0, y=0)


