import os
import sys
import time

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 40  #怎么来的？？？自己设置了一个差不多的值，只要比term_width窄就行
last_time = time.time()
begin_time = last_time

'''每一个batch_train就调用一次progress_bar，但是在一个epoch中的每个batch都共享一些数据，所以才有了global last_time, begin_time'''
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    #last_time = begin_time
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1  
    '''为啥减去1呢,这个1是给最后那个>留的'''

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step time: %s' % format_time(step_time))
    L.append(' | Total time: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')  # \b退格，好重新写入一些东西，大不了覆盖掉一点进度条上的#符号
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:  #为什么减1，你看看函数传入的两个参数是什么，一个从0开始标记下标，一个是长度
        sys.stdout.write('\r')  # \r回车，把缓冲区的内容显示出来
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()  # 清空缓冲区，好写下一条


def format_time(seconds):
    days = int(seconds / 3600 / 24)  # 如果上了天了，就看还剩多少秒，否则不变
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)  # 把xxx.y 秒里面的小数搞掉
    seconds = seconds - secondsf  # 求出还有零点几秒
    millis = int(seconds * 1000)

    '''通过设置i的初始值和范围，就可以设置显示时间的范围，师兄是只让它显示紧邻的两级时间'''
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
