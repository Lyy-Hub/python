from ctypes import *

from sympy import true

while true:
        user32 = windll.LoadLibrary('user32.dll')
        user32.LockWorkStation()