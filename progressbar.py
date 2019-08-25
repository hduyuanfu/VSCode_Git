from tkinter import *
from tkinter import ttk
import time
root = Tk()
'''改了个ssh-key再试试'''
'''我改一遍再上传'''
root.title('progressbar组件案例')
root.geometry('200x150')
p1 = ttk.Progressbar(root, length=200, mode='determinate', orient=HORIZONTAL)
p1.grid(row=1, column=1)
p1['maximum'] = 100
p1['value'] = 0
for i in range(100):
    p1['value'] = i+1
    root.update()
    time.sleep(0.1)
root.mainloop()