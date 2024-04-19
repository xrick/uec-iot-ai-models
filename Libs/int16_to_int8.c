int_8* int16_to_int8(int[] data_ary)
{
    /*
        input:
            data_ary: 1D int16 array
        output:
            data_ary: 1D int8 array
        note:
            the input and output arrays length all should be 30225
    */
    
    uint16_t len = sizeof(data_ary)/sizeof(data_ary[0]);
    int16_t max_v = data_ary[0];
    float epsilon = 0.000000001;
    float zero_offset = -0.5;
    int_8 ret_ary[len]; //declare a int8 array which has the same length as data_ary                  
    for (int i = 0; i < len; i++) {      
       if(arr[i] > max_v)    
           max_v = arr[i];    
    }      
    
    //using 255 is because using it in quantizaion processing
    //need to modify to 127 for the next time, just because we use
    //int8 as our input and output
    scaling_factor = (2 * max_v)/255 + epsilon; 
    for(int i = 0; i < len; i++)
    {
        float tmp_v = data_ary[i]/scaling_factor + zero_offset;
        ret_ary[i] = int_8(math.floor(tmp_v)); //we can using math.ceil() too see which is better.
    }
    return ret_ary;
}