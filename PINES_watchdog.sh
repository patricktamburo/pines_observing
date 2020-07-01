#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3 -W ignore

from PINES_watchdog import *

if len(sys.argv)==1:
    PINES_watchdog()
else:
    if len(sys.argv)==2:
        PINES_watchdog(seeing=float(sys.argv[1]))
    else:
        print("Wrong number of arguments for PINES_watchdog.sh")



