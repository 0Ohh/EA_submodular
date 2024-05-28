
fileName = 'kzt消融-无ham'
with open(fileName) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].split()

with open(fileName, 'w') as f:
    for li in lines:
        t = li[0]
        if len(t) > 4:
            # print(t)
            if t[-4:] == '0000':
                # print(li)
                f.write(li[0]+'\n')
                f.write(li[1]+'\n')

