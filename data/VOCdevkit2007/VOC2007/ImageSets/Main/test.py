import os

f = open("val.txt", 'r')
datadir = '../../JPEGImages/'
for i in f.readlines():
    name = datadir + i[:-1] +".jpg"
    print(name)
    assert os.path.exists(name)
    
