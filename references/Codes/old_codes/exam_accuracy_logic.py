

def runAccuracyTest(weights_file=None, test_matfile=None, comp_base=1, threshold=-145, lbl_len=513,
                    icfilterary=None ,startPoint=1600, stepQuantity=400 \
                   ):
    """doing the accuracy of input matfile
    
    parameters:on-default argument fo
    test_matfile: to test mat file path
    camp_base: accuracy co
    
    """
    weights = spio.loadmat(weights_file)
    w1 = weights["w1"]
    w2 = weights["w2"]
    w3 = weights["w3"]
    b1 = weights["b1"]
    b2 = weights["b2"]
    b3 = weights["b3"]
    mat_data = spio.loadmat(test_matfile)
    
    inband = 4
    bandnum = 10
    count_answer_1 = 0
    count_answer_0 = 0
    count_test_1 = 0
    count_test_0 = 0

    y_answer = np.empty([0,0],dtype=int)
    y_test = np.empty([0,0])

    local_power = np.empty([0,0])
    local_power_normalize = np.empty([0,0],dtype=int)
    x_all = np.empty([0,0])
    x_normalize_all = np.empty([0,0])
    
    s_len = int(400*(np.floor(len(mat_data['s']))))
    sample_num = inband * bandnum
    test_counter = 0
    loop_record={}
    loop_record["mat_file"]=test_matfile
    loop_record["comp_base"]=comp_base
#     constant_str1 = "loop"
    for i in range(startPoint,s_len,stepQuantity):
        test_counter += 1
        #divide four segment every 25ms
        idx1 = i-1600
        idx2 = i-1200
        idx3 = i-800
        idx4 = i-400
        s1 = mat_data['s'][idx1:idx2] #0-399
        s2 = mat_data['s'][idx2:idx3] #400-799run_test_main,number
        s3 = mat_data['s'][idx3:idx4] #800-1199
        s4 = mat_data['s'][idx4:i]    #1200-1599
        loop_record={
            "s1":"{}_{}".format(idx1,idx2),
            "s2":"{}_{}".format(idx2,idx3),
            "s3":"{}_{}".format(idx3,idx4),
            "s4":"{}_{}".format(idx4,i)
        }
        loop_record["threshold"]=threshold      
#         s1 = mat_data['s'][i-1600:i-1200]#0-399
#         s2 = mat_data['s'][i-1200:i-800] #400-799
#         s3 = mat_data['s'][i-800:i-400]  #800-1199
#         s4 = mat_data['s'][i-400:i]      #1200-1599

        s_fft_1 = np.fft.fft(s1,n=1024,axis=0) 
        s_fft_2 = np.fft.fft(s2,n=1024,axis=0) 
        s_fft_3 = np.fft.fft(s3,n=1024,axis=0) 
        s_fft_4 = np.fft.fft(s4,n=1024,axis=0) 

        s_fft_1_513 = np.split(s_fft_1,[0,lbl_len-1],axis=0)[1]
        s_fft_2_513 = np.split(s_fft_2,[0,lbl_len-1],axis=0)[1]
        s_fft_3_513 = np.split(s_fft_3,[0,lbl_len-1],axis=0)[1]
        s_fft_4_513 = np.split(s_fft_4,[0,lbl_len-1],axis=0)[1]

        # there is one error element in every array in position 9 of each array
        x1 = np.log(np.matmul(icfilterary,(np.abs(s_fft_1_513)**2)) + 0.0001)
        x2 = np.log(np.matmul(icfilterary,(np.abs(s_fft_2_513)**2)) + 0.0001)
        x3 = np.log(np.matmul(icfilterary,(np.abs(s_fft_3_513)**2)) + 0.0001)
        x4 = np.log(np.matmul(icfilterary,(np.abs(s_fft_4_513)**2)) + 0.0001)

        x= np.array([x1,x2,x3,x4]).reshape(1,sample_num)
        x_40 = x[0:40]
        
        max_ele = np.amax(x_40,axis=1)
        min_ele = np.amin(x_40,axis=1)
        sum_of_x = np.sum(x_40)
        
        loop_record["max_ele"]=max_ele[0]
        loop_record["min_ele"]=min_ele[0]
        loop_record["sum_of_x"]=sum_of_x

        # normalize
        x_normalize = (x_40-min_ele)/(max_ele-min_ele+0.0001)

        #counting the test
        loop_record["answer"]=0
        if sum_of_x > threshold:
            #performing the model weight mulplications
            answer = softmax(np.matmul(relu(np.matmul(relu(np.matmul(x_normalize,w1)+b1),w2)+b2),w3)+b3)
            loop_record["answer"]=answer
            if answer[0,0] > answer[0,1]:
                y_test = np.append(y_test, 0)
                count_test_0 += 1
            else:
                y_test = np.append(y_test, 1)
                count_test_1 += 1
        else:
            loop_record["answer"]=9999
            y_test = np.append(y_test, 0)
            count_test_0 += 1

        #counting our answer
        if comp_base == 1:
            if sum_of_x > threshold:
                y_answer = np.append(y_answer,1)
                count_answer_1 += 1
            else:
                y_answer = np.append(y_answer,0)
                count_answer_0 += 1
        else:
            y_answer = np.append(y_answer,0)
            count_answer_0 += middleFreq
            
    loop_record["total_loop"] = test_counter
    acc = 0
    LED = 0
    wrong = np.empty([0,0])
    loop_record["y_answer"]=y_answer.tolist()
    y_answer_len = len(y_answer)
    loop_record["y_answer_len"]=y_answer_len
    
    loop_record["y_test"] = y_test.tolist()
    y_test_len = len(y_test)
    loop_record["y_test_len"] = y_test_len
    
    if comp_base == 1: #if our test data is human voice
        for c in range(y_answer_len):
            acc += abs(y_answer[c]-y_test[c])

        acc = (y_answer_len-acc)/y_answer_len
    else:
        acc = count_test_0/(count_test_0+count_test_1)
    
    loop_record["accuracy"]=acc
#     return loop_record
    print("acc is ", acc)
    print("{} test completed\n".format(test_matfile))
    return acc, json.dumps(loop_record,cls=NumpyEncoder)