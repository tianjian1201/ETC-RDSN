/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Perform the adaptive arithmetic (AC) decoding for the coded sequence
//
// Usage: 
//      dec_seq = decode_adaptive_ac(num_sym, len, enc_seq);
//
// Parameters:
//      num_sym     --the number of symbols
//      len         --the length of to-be-decoded sequence
//      enc_seq     --the encoded sequence which is used to decode via the AC
//      dec_seq     --the decoded sequence via the AC
//
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

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	unsigned int k, len_data, num_symbols, cnt_bytes;	
	unsigned char *ptr_code   = NULL;
	double  *ptr_decode = NULL;
	
	//obtain the input parameters
	num_symbols  = (unsigned int) mxGetScalar(prhs[0]); 
	len_data     = (unsigned int) mxGetScalar(prhs[1]); //the number of the source data
	cnt_bytes    = (unsigned int) mxGetM(prhs[2]) * (unsigned int) mxGetN(prhs[2]); //the byte number of the encoded sequence
	ptr_code     = (unsigned char *) mxGetPr(prhs[2]);
	//mexPrintf("num_symbols=%d, len_data=%d, num_bytes=%d\n", num_symbols, len_data, cnt_bytes);
	
	//assign memory for decoded sequence
	plhs[0] = mxCreateDoubleMatrix(1, len_data, mxREAL);
	ptr_decode = mxGetPr(plhs[0]);
	if (ptr_decode == NULL)
	{
		mexPrintf("Failure to allocate the memory!\n");
		exit(-1);
	}
	

	//create objects for arithmetic coding
	Arithmetic_Codec    codec(cnt_bytes+16); //extra 16 bytes, just in case 
	Adaptive_Data_Model adaptive_model(num_symbols);

	//copy the coded bytes to the buffer of Object codec
	memcpy(codec.buffer(), ptr_code, cnt_bytes);
	
	//perform the arithmetic coding
	codec.start_decoder();
	for (k=0; k<len_data; k++)
	{
		ptr_decode[k] = (double) codec.decode(adaptive_model);
	}
	codec.stop_decoder();
		
	return;
}