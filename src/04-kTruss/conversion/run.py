import os
import subprocess

for filename in os.listdir("/home/dsladmin/Awadhesh/input"):
    # print filename
    # print(os.path.splitext(filename)[0])
    #filename_clean = os.path.splitext(filename)[0]
    #proc = subprocess.Popen(["./serial.o",filename_clean])
    #proc.wait()
    args = "./ktruss_static -g /home/dsladmin/Awadhesh/input/" + filename+ " -d 0 -k 68 -o > "+ filename
    subprocess.call(args,shell=True)
# proc = subprocess.Popen(["./serial.o","cit-HepPh_adj"])
# print (os.getcwd())
