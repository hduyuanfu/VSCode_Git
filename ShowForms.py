import tkinter
MainForm = tkinter.Tk()
MainForm.geometry('500x300')
MainForm.title('三酷猫！')
#MainForm.iconbitmap('C:\Users\袁富\Pictures\Saved Pictures\vscode背景图.jpg')
MainForm['background'] = 'LightSlateGray'
button1 = tkinter.Button(MainForm, text = 'NIIT', fg = 'black')
button2 = tkinter.Button(MainForm, text = '实时', fg = 'black')
button3 = tkinter.Button(MainForm, text = '数据集', fg = 'black')

def TurnProperty(event):
    event.widget['activeforeground'] = 'red'
    event.widget['text'] = '丢'

button1.bind('<Enter>', TurnProperty)
button2.bind('<Enter>', TurnProperty)
button3.bind('<Enter>', TurnProperty)

button1.pack(side = 'left', padx = '1m')
button2.pack(side = 'left', padx = '1m')
button3.pack(side = 'left', padx = '1m')
#button1.config(fg = 'red', bg = 'blue')
MainForm.mainloop()