import random
import json
ts = [[10-6.8,8-6.8,3-6.8,7-6.8,6-6.8]*40, [-7,1,6,-8,4,4]*33+[-7,1], [1-2.5,2-2.5,3-2.5,4-2.5]*50, [1-1.5,2-1.5]*100, [3-2,2-2,1-2]*66+[3-2,2-2]]
start_list = random.sample(range(1, 20001), 1000)
result = []
for i in range(1000):
    check = i%5
    l = [start_list[i]*k for k in ts[check]]
    result.append(l)
with open('data/data.json', 'w') as f:
    json.dump(result, f)
