#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <omp.h>

using namespace std;

//========== fvec reader ==========
vector<vector<float>> read_fvecs(const string &filename)
{
    ifstream fin(filename, ios::binary);
    if (!fin)
    {
        cerr << "Cannot open " << filename << endl;
        exit(1);
    }

    vector<vector<float>> data;
    while (true)
    {
        int dim;
        fin.read((char *)&dim, 4);
        if (!fin)
            break;

        vector<float> v(dim);
        fin.read((char *)v.data(), dim * 4);
        if (!fin)
            break;

        data.push_back(move(v));
    }
    return data;
}

//====== read alpha (same format: dim + value) ======
vector<float> read_alpha(const string &filename)
{
    ifstream fin(filename, ios::binary);
    if (!fin)
    {
        cerr << "Cannot open alpha file " << filename << endl;
        exit(1);
    }

    vector<float> alpha;
    while (true)
    {
        int dim;
        fin.read((char *)&dim, 4);
        if (!fin)
            break;

        float v;
        fin.read((char *)&v, 4);
        alpha.push_back(v);
    }
    return alpha;
}

//========== read trace - random delete ids ==========
vector<int> read_trace(const string &filename)
{
    ifstream fin(filename, ios::binary);
    if (!fin)
    {
        cerr << "Cannot open trace file " << filename << endl;
        exit(1);
    }

    int num;
    fin.read((char *)&num, 4);

    vector<int> ids(num);
    for (int i = 0; i < num; i++)
        fin.read((char *)&ids[i], 4);

    return ids;
}

//========== ivecs writer ==========
void write_ivecs(const string &filename, const vector<vector<int>> &data)
{
    ofstream fout(filename, ios::binary);
    for (auto &row : data)
    {
        int dim = row.size();
        fout.write((char *)&dim, 4);
        fout.write((char *)row.data(), dim * 4);
    }
}

//========== l2 distance ==========
inline float euclidean(const float *a, const float *b, int dim)
{
    float dist = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return sqrt(dist);
}

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        cout << "Usage:\n"
             << "./rand_del_gt  base_img.fvecs base_text.fvecs "
             << "query_img.fvecs query_text.fvecs alpha.fvecs trace.bin "
             << "\nOutput: top10.ivecs\n";
        return 1;
    }

    string base_img_file = argv[1];
    string base_text_file = argv[2];
    string query_img_file = argv[3];
    string query_text_file = argv[4];
    string alpha_file = argv[5];
    string trace_file = argv[6];

    // read vectors
    auto base_img = read_fvecs(base_img_file);
    auto base_text = read_fvecs(base_text_file);
    auto query_img = read_fvecs(query_img_file);
    auto query_text = read_fvecs(query_text_file);
    auto alpha = read_alpha(alpha_file);

    int N = base_img.size();
    int nq = query_img.size();
    int topk = 10;

    int base_dim = base_img[0].size();
    int text_dim = base_text[0].size();

    // read delete ids
    vector<int> del_ids = read_trace(trace_file);
    unordered_set<int> del_set(del_ids.begin(), del_ids.end());

    // generate alive index list
    vector<int> alive_ids;
    alive_ids.reserve(N);
    for (int i = 0; i < N; i++)
        if (!del_set.count(i))
            alive_ids.push_back(i);

    int alive_size = alive_ids.size();
    cout << "Original N = " << N << endl;
    cout << "Deleted = " << del_ids.size() << endl;
    cout << "Remaining = " << alive_size << endl;

    // allocate GT
    vector<vector<int>> topk_indices(nq, vector<int>(topk));

#pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < nq; qi++)
    {
        const float *q_img = query_img[qi].data();
        const float *q_txt = query_text[qi].data();
        float a = alpha[qi];

        vector<pair<float, int>> scores(alive_size);

        for (int j = 0; j < alive_size; j++)
        {
            int idx = alive_ids[j];
            float d_img = euclidean(q_img, base_img[idx].data(), base_dim);
            float d_txt = euclidean(q_txt, base_text[idx].data(), text_dim);
            float score = a * d_img + (1 - a) * d_txt;
            scores[j] = {score, idx};
        }

        nth_element(scores.begin(), scores.begin() + topk, scores.end());
        sort(scores.begin(), scores.begin() + topk);

        for (int k = 0; k < topk; k++)
            topk_indices[qi][k] = scores[k].second;
    }

    write_ivecs("top10.ivecs", topk_indices);
    cout << "Saved top10.ivecs\n";

    return 0;
}
