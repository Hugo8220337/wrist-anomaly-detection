import os
import sys
folder = './Data/'
for filename in os.listdir(folder):
    infilename = os.path.join(folder,filename)
    if not os.path.isfile(infilename): continue
    oldbase = os.path.splitext(filename)
    newname = infilename.replace('.txt', '.csv')
    output = os.rename(infilename, newname)