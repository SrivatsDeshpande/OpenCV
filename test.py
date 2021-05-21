
from backup import *
def test_cases(time_taken,names,accuracy):
    
    objs=[]
    for i in names:
            if len(names)==1:
                time = time_taken
                objs.append(i)
            elif len(names)>1:
                time = time_taken
                objs.append(i)
            else:
                continue
    print('Time taken to detect %s is %f'%(str(objs),time))
    for i in objs:      
        print('Accuracy of %s is %f'%(str(i),accuracy[i]))
    print()
        
        
for i in range(0,5):
    a = detect()
    print('Test number ',i+1)
    test_cases(a[0], a[1], a[2])





