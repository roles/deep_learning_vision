#include<iostream>
#include<fstream>

#define KYOTO_DATA "/home/rolexye/project/data/image_db_kyoto/kyoto.txt"

using namespace std;

class Matrix{
private:
    int _nrow, _ncol;
    int _nelem;
    float *_data;
public:
    Matrix(int nrow, int ncol);
    int get_row_num();
    int get_col_num();
    int read_from_text(istream &fs);
    ~Matrix();
};
