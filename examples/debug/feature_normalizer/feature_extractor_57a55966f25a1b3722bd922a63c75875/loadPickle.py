import pickle 
import numpy

objects = []
with (open("scale_fold1.cpickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

x = 1