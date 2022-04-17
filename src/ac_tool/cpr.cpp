#include <iostream>
#include <vector>
#include <fstream>
#include <typeinfo>
#include <sys/io.h>

#include "arithmetic_codec.h"
#define MAX_CODE_BYTES	1e7   //the predefined maximum coded bytes

using namespace std;

int compresse_error(string dataset)
{
    int len_data = 0;
    ifstream error_file;
    vector<int> cmp_cnt;
    int file_num = 0;
    if(dataset=="B100")
        file_num = 100;
    if(dataset=="Set5")
        file_num = 5;
    if(dataset=="Set14")
        file_num = 4;
    if(dataset=="Urban100")
        file_num = 100;
    
    for(int scale = 2; scale<=4; scale++)
    {
        cout << dataset + " scale=" + to_string(scale)<< endl;
        double sum = 0;
        for(int i = 0; i < file_num; i++)
        {
            vector<int> data;
            string file_name = to_string(i) + ".txt";
            string file_dir = "../error_data/"+dataset+"/X"+to_string(scale)+"/"+file_name;
            error_file.open(file_dir);
            int t = 0;
            int data_size = 0;
            int hr_size = 0;
            int num_symbols = 0;
            error_file >> hr_size >> num_symbols;
            while(!error_file.eof())
            {
                error_file >> t;
                data.push_back(t);
            }
            Arithmetic_Codec codec((unsigned int)MAX_CODE_BYTES);
            Adaptive_Data_Model adaptive_mode(num_symbols);
            codec.start_encoder();
            for(int j = 0; j < data.size(); j++)
            {
                if(data[j]==' ')
                    break;
                int temp_data = data[j];
                codec.encode(temp_data, adaptive_mode);
            }
            int cnt_bytes = codec.stop_encoder();
            double cmp = (hr_size*(1.0/scale/scale)+cnt_bytes)/hr_size;
            cout << cmp << endl;
            

            sum += cmp;
            error_file.close();
        }
        cout << sum/(file_num) << endl;
        

    }
    return 0;
}

int main()
{
    //compresse_error("B100");
    //compresse_error("Set5");
    compresse_error("Set14");
    //compresse_error("Urban100");
    return 0;
}