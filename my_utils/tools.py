# coding=utf-8

"""
@author: beyourself
@time: 2018/8/12 10:26
@file: tools.py
"""


def time_cost(start, end, epoch=None):
    secs = int(end - start)
    if epoch:
        sec_epoch = secs // epoch
        m_epoch = sec_epoch // 60
        print('\ntraining speed : %s m/epoch\n' % m_epoch)

    h = secs // 3600
    m = (secs - h * 3600) // 60

    print('\ntime cost %shour %sminute\n\n' % (h, m))
