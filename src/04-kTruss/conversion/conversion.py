import sys
# argumentList = sys.argv
# print argumentList
    #file1 = open("myfile.txt","r")
list1=['graph500-scale22-ef16_adj.mmio']
# argumentList.remove("conversion.py")
# print argumentList

for fileName in list1:
    file=open("/home/dtanwar/Project/input/"+fileName,"r")
    i=0
    newFile=open("/home/dtanwar/Project/Mauro/input/"+fileName,"w")
    for line in file:
        if i>2:
            if line.strip():
                newFile.write("\t".join(line.split()[:2]) + "\n")
        i+=1
    file.close()
    newFile.close()
    print fileName+" Done"
