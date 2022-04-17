#include <iostream>
#include <vector>
#include <fstream>
#include <typeinfo>

#include "arithmetic_codec.h"
#define MAX_CODE_BYTES	1e7   //the predefined maximum coded bytes

using namespace std;

int main()
{
    int num_symbols = 256;
    FILE *code_file;
    code_file = fopen("code_file.bin", "r");

    Arithmetic_Codec codec((unsigned int)MAX_CODE_BYTES);
    Adaptive_Data_Model adaptive_model(num_symbols);
    codec.read_from_file(code_file);
    vector<short> data;
    ofstream fout("text_decode.txt");
    // codec.start_decoder();
    for (int i = 0; i < 120750; i++)
    {
        data.push_back(codec.decode(adaptive_model));
        fout << codec.decode(adaptive_model) << " ";
        // cout << codec.decode(adaptive_model) << " ";
        // if(i %50 == 0)
            // cout << endl;

    }
    codec.stop_decoder();

    
    /*for(int i = 0; i < data.size(); i++)
    {
        fout << data[i] << " ";
    }*/

    return 0;

}