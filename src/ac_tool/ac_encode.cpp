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
    int len_data = 0;
    ifstream error_file;
    vector<int> cmp_cnt;

    vector<int> data;
    string file_name = "text.txt";
    string file_dir = "../" + file_name;
    error_file.open(file_dir);
    int t = 0;
    int data_size = 0;
    //error_file >> num_symbols;
    while (!error_file.eof())
    {
        error_file >> t;

        data.push_back(t);
    }
    //num_symbols = 511;

    /*for(int i = 0; i < 120701; i++)
    {
        cout << data[i] << ' ';
        if(i % 30 == 0)
            cout << endl;
    }
    cout << endl;*/
    //cout << data.size() << endl;



    
    Arithmetic_Codec codec((unsigned int)MAX_CODE_BYTES);
    Adaptive_Data_Model adaptive_mode(num_symbols);
    codec.start_encoder(); 
    for(int j = 0; j < data.size()-1; j++)
    {
        codec.encode(data[j], adaptive_mode);
    }
    
    //int cnt_bytes = codec.stop_encoder();

    //write to file(bin)
    FILE *code_file;
    code_file = fopen("code_file.bin", "w");
    int cnt_bytes = codec.write_to_file(code_file);
    fclose(code_file);

    // int cnt_bytes = codec.stop_encoder();

    
    //cout << file_name << endl;
    cout << "AC:" << cnt_bytes << " Byte" << endl;
    cout << "bin size:" << data.size() << " Byte" << endl;
    cout << endl;
    error_file.close();
    
    
    


    return 0;
}
