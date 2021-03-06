import re
import os

dic = {}
ham = 0
spam = 0

for i in os.listdir('train/'):
     with open(i) as file:
         cont1 = file.read()

     cont = re.split('[^a-zA-Z]', cont1)
     cont = list(filter(None, cont))

     for item in cont:
        if (item in dic):
            if(re.search(re.compile("ham"), i)):
                dic[item][0] += 1
                ham += 1
            if(re.search(re.compile("spam"), i)):
                dic[item][2] += 1
                spam += 1
        else:
            if(re.search(re.compile("ham"), i)):
                dic[item] = [1, 0, 0, 0]
                ham =+ 1
            elif(re.search(re.compile("spam"), i)):
                dic[item] = [0, 0, 1, 0]
                spam =+ 1


for i in dic.values():
    i[1] = ( i[0] + 0.5 ) / ( ham*1.5 )
    i[3] = ( i[2] + 0.5 ) / ( spam*1.5 )

f = open("model.txt", "a")

k = 0
for i in sorted(dic.keys()):
    f.write(str(k) + "  " + str(i) + "  " + str(dic[i][0]) + "  " + str(dic[i][1]) + "  " + str(dic[i][2]) + "  " + str(dic[i][3]) + "\n")
    k += 1

f.close()
