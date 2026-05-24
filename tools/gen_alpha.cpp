#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

using namespace std;

// 读取 fvecs，只用来统计 query 数量
int count_fvecs(const string &filename)
{
    ifstream fin(filename, ios::binary);
    if (!fin)
    {
        cerr << "Cannot open " << filename << endl;
        exit(1);
    }

    int cnt = 0;
    while (true)
    {
        int dim;
        fin.read((char *)&dim, 4);
        if (!fin)
            break;
        fin.seekg(dim * 4, ios::cur);
        cnt++;
    }
    return cnt;
}

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        cerr << "Usage:\n"
             << "./gen_alpha query.fvecs alpha_min alpha_max out.fvecs out.txt\n";
        return 1;
    }

    string query_file = argv[1];
    float alpha_min = stof(argv[2]);
    float alpha_max = stof(argv[3]);
    string out_fvecs = argv[4];
    string out_txt = argv[5];

    int nq = count_fvecs(query_file);
    cout << "Number of queries: " << nq << endl;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(alpha_min, alpha_max);

    ofstream fout_bin(out_fvecs, ios::binary);
    ofstream fout_txt(out_txt);

    for (int i = 0; i < nq; i++)
    {
        float alpha = dist(gen);

        // fvecs 格式：dim=1 + value
        int dim = 1;
        fout_bin.write((char *)&dim, 4);
        fout_bin.write((char *)&alpha, 4);

        fout_txt << i << " " << alpha << "\n";
    }

    cout << "Alpha files written:\n"
         << "  Binary: " << out_fvecs << "\n"
         << "  Text:   " << out_txt << endl;

    return 0;
}
