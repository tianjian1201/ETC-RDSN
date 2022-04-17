/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Perform the adaptive arithmetic encoding for the input quantized integer sequence
//
// Usage: enc_seq = encode_adaptive_ac(sym_num, data_seq)
//
// Parameters:
//    sym_num    --the number of symbols
//    data_seq   --the to-be-encoded data sequence
//    enc_seq    --the sequence encoded with arithmatic code
//
// WangCT, 2014-08-19
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"
#include "math.h"
#include "arithmetic_codec.h"
#include "arithmetic_codec.cpp"

#define MAX_CODE_BYTES	1e7   //the predefined maximum coded bytes

///////////////////////////////////////////////////
//nlhs( number left hand s):左边参数个数
//mxArray *plhs[]是一个指针数组，数组中的每一个元素都是一个指针，指向输出的矩阵
//nrhs 是右边参数个数，也就是输入参数的个数，mxArray *prhs[]数组中的每个指针指向输入矩阵
//mxGetPr()函数返回一个double*型的指针，指向矩阵的第一个元素
//
//
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	unsigned int k, len_data, num_symbols, cnt_bytes;
	unsigned int  *ptr_data_src = NULL;
	unsigned char *ptr_code     = NULL;
	
	//obtain the input parameters
	num_symbols  = (int) mxGetScalar(prhs[0]); //mxGetScalar(prhs[0]) ：把通过prhs[0]传递进来的mxArray类型的指针指向的数据（标量）赋给C程序里的变量;
	len_data     = (unsigned int) (mxGetM(prhs[1]) * mxGetN(prhs[1])); //the number of the to-be-encoded data
	ptr_data_src = (unsigned int *) mxGetPr(prhs[1]);
	//mexPrintf("num_symbols=%d, len_data=%d\n", num_symbols, len_data);
	//mexPrintf("data[%d]=%d, data[%d]=%d\n", 12, ptr_data_src[12], 52, ptr_data_src[52]); 
	
	//create objects for arithmetic coding
	Arithmetic_Codec    codec((unsigned int)MAX_CODE_BYTES);
	Adaptive_Data_Model adaptive_model(num_symbols);
	
	//perform the arithmetic coding
	codec.start_encoder();
	for (k=0; k<len_data; k++)
	{
		codec.encode(ptr_data_src[k], adaptive_model); 
	}
	cnt_bytes = codec.stop_encoder();

	//feed back the coded byte sequence to the Matlab routine
	plhs[0] = mxCreateNumericMatrix(1, cnt_bytes, mxUINT8_CLASS, mxREAL);
	plhs[1] = mxCreateDoubleScalar(0); //for saving the real bytes of encoded sequence

	ptr_code = (unsigned char *) mxGetPr(plhs[0]); //coded sequence
	memcpy(ptr_code, codec.buffer(), cnt_bytes);

	*(mxGetPr(plhs[1])) = cnt_bytes; //the number of coded bytes

	return;
}
