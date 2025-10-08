#观察rdcost的分布规律
import glob, struct, json, os, math, random
import numpy as np

def set_cus_rdcost(file_name, get_w, get_h, cuSize):
    record_size = struct.calcsize("=4B17d")
    with open(file_name,'rb') as fd:
        buffer = fd.read()
    
    n = 0
    valid_flag=1

    if len(buffer)%record_size!=0:
        valid_flag=0
    else:
        while n * record_size < len(buffer):
            tmpcost=[0,0,0,0,0,0]
            time = [0,0,0,0,0,0]
            five_decision = [0, 0, 0, 0, 0]
            cuh, cuw, _, _, tmpcost[0], tmpcost[1], tmpcost[2], tmpcost[3], tmpcost[4],\
             tmpcost[5], time[0], time[1], time[2], time[3], time[4], time[5], \
            five_decision[0], five_decision[1], five_decision[2], five_decision[3], five_decision[4] = \
            struct.unpack("=4B17d", buffer[n * record_size: (n + 1) * record_size])
            n += 1

            if cuh+cuw == get_h+get_w:
                if cuh > cuw:
                    tmpcost[3], tmpcost[2] = tmpcost[2], tmpcost[3]
                    tmpcost[5], tmpcost[4] = tmpcost[4], tmpcost[5]
                    time[3], time[2] = time[2], time[3]
                    time[5], time[4] = time[4], time[5]
                    five_decision[2], five_decision[1] = five_decision[1], five_decision[2]
                    five_decision[4], five_decision[3] = five_decision[3], five_decision[4]

                cansplit_flag = False
                for q in range(1,6):
                    if tmpcost[q]!=0 and tmpcost[q]<99999999999999999.9:
                        cansplit_flag=True
                if not cansplit_flag:
                    continue
                
                for q in range(6):
                    if tmpcost[q]>99999999999999999.9:
                        tmpcost[q] = 0

                output = [tmpcost[c] for c in range(6)]+[time[c] for c in range(6)]+[five_decision[c] for c in range(5)]
                #print(output)
                rdcost["output"].append(output)
                    
    return valid_flag

for cuSize in [[64, 128], [64, 64], [32, 64], [32, 32], [16, 64], [16, 32], [16, 16], [8, 64], [8, 16], [8, 8], \
               [4, 64], [4, 32], [4, 16], [4, 8]]:
    cuSize_str = str(cuSize[0])+"_"+str(cuSize[1])
    for qp in [37, 32, 27, 22]:
        if not os.path.exists('./collected_'+cuSize_str):
            os.mkdir('./collected_'+cuSize_str)
        if not os.path.exists('./collected_'+cuSize_str+'/'+str(qp)):
            os.mkdir('./collected_'+cuSize_str+'/'+str(qp))
        for file in glob.glob("./compressed_"+str(qp)+"/*"):
            valid_flag=1
            cus={} #first: cantQT, second: canQT
            rdcost={"output":[]}

            with open(file, 'rb') as fd:
                buffer = fd.read()
            n = 0

            valid_flag*=set_cus_rdcost(file, cuSize[0], cuSize[1], cuSize)

            if valid_flag==1:
                with open('./collected_'+cuSize_str+'/'+str(qp)+'/'+file.split("/")[-1].split(".")[0]+".json","w") as write_file:
                    json.dump(rdcost, write_file)
            else:
                print(file)