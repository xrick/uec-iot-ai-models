#include<stdio.h>

#define mask8 0x4000 // >> 8 : 16384
#define mask7 0x2000 // >> 7 :  8192
#define mask6 0x1000 // >> 6 :  4096
#define mask5 0x0800 // >> 5 :  2048
#define mask4 0x0400 // >> 4 :  1024
#define mask3 0x0200 // >> 3 :   512
#define mask2 0x0100 // >> 2 :   256
#define mask1 0x0080 // >> 1 :   128
#define mask0 0x0040 // >> 0 :    64 below the value, drop the value

int8_t maskOP(int16_t x){
    if (mask8&x)
        return x >> 8;
    else if (mask7&x)
        return x >> 7;
    else if (mask6&x)
        return x >> 6;
    else if (mask5&x)
        return x >> 5;
    else if (mask4&x)
        return x >> 4;
    else if (mask3&x)
        return x >> 3;
    else if (mask2&x)
        return x >> 2;
    else if (mask1&x)
        return x >> 1;
    else if (mask0&x)
        return x;
    else
        return 0;
}

void unittest_maskOP(){
	int16_t testx = 8315;
	int8_t converted_x = maskOP(testx);
	printf("the converted_x is %d", converted_x);
}