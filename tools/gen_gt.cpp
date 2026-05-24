// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <cmath>
// #include <algorithm>
// #include <string>

// #include <omp.h>

// using namespace std;

// // ================= 工具函数 =================

// vector<vector<float>> read_fvecs(const string &filename)
// {
//     ifstream fin(filename, ios::binary);
//     if (!fin)
//     {
//         cerr << "Cannot open " << filename << endl;
//         exit(1);
//     }

//     vector<vector<float>> data;
//     while (true)
//     {
//         int dim;
//         fin.read((char *)&dim, 4);
//         if (!fin)
//             break;
//         vector<float> v(dim);
//         fin.read((char *)v.data(), dim * 4);
//         if (!fin)
//             break;
//         data.push_back(move(v));
//     }
//     return data;
// }

// vector<float> read_alpha(const string &filename)
// {
//     ifstream fin(filename, ios::binary);
//     if (!fin)
//     {
//         cerr << "Cannot open alpha file " << filename << endl;
//         exit(1);
//     }

//     vector<float> alpha;
//     while (true)
//     {
//         int dim;
//         fin.read((char *)&dim, 4);
//         if (!fin)
//             break;
//         float v;
//         fin.read((char *)&v, 4);
//         alpha.push_back(v);
//     }
//     return alpha;
// }

// void write_ivecs(const string &filename, const vector<vector<int>> &data)
// {
//     ofstream fout(filename, ios::binary);
//     for (auto &row : data)
//     {
//         int dim = row.size();
//         fout.write((char *)&dim, 4);
//         fout.write((char *)row.data(), dim * 4);
//     }
// }

// inline float euclidean(const float *a, const float *b, int dim)
// {
//     float dist = 0.0f;
//     for (int i = 0; i < dim; i++)
//     {
//         float d = a[i] - b[i];
//         dist += d * d;
//     }
//     return sqrt(dist);
// }

// // ================= 主逻辑 =================

// int main(int argc, char **argv)
// {
//     if (argc != 12)
//     {
//         cerr << "Usage:\n"
//              << "./dynamic_gt \\\n"
//              << "  base_img.fvecs \\\n"
//              << "  base_text.fvecs \\\n"
//              << "  query_img.fvecs \\\n"
//              << "  query_text.fvecs \\\n"
//              << "  alpha_file.fvecs \\\n"
//              << "  output_dir \\\n"
//              << "  init_ratio \\\n"
//              << "  update_ratio \\\n"
//              << "  update_rounds \\\n"
//              << "  mode(insert|delete)\n";
//         return 1;
//     }

//     string base_img_file = argv[1];
//     string base_text_file = argv[2];
//     string query_img_file = argv[3];
//     string query_text_file = argv[4];
//     string alpha_file = argv[5];
//     string output_dir = argv[6];
//     float init_ratio = stof(argv[7]);
//     float update_ratio = stof(argv[8]);
//     int update_rounds = stoi(argv[9]);
//     string mode = argv[10];
//     int threads = stoi(argv[11]);
//     omp_set_num_threads(threads);
//     std::cout << "Threads: " << argv[11] << std::endl;
//     // == == = 读取数据 ==== =
//     std::cout << "Start read base_img base_text query_img query_text alpha" << std::endl;
//     auto base_img = read_fvecs(base_img_file);
//     auto base_text = read_fvecs(base_text_file);
//     auto query_img = read_fvecs(query_img_file);
//     auto query_text = read_fvecs(query_text_file);
//     auto alpha = read_alpha(alpha_file);

//     int N = base_img.size();
//     int nq = query_img.size();
//     int topk = 10;

//     if ((int)alpha.size() != nq)
//     {
//         cerr << "Alpha size mismatch with query size!" << endl;
//         return 1;
//     }

//     int init_N = max(1, (int)(N * init_ratio));
//     int step = max(1, (int)(N * update_ratio));

//     cout << "Total base: " << N << endl;
//     cout << "Initial active: " << init_N << endl;
//     cout << "Update step: " << step << endl;
//     cout << "Rounds: " << update_rounds << endl;
//     cout << "Mode: " << mode << endl;

//     int base_dim = base_img[0].size();
//     int text_dim = base_text[0].size();

//     // ===== active 区间（只用两个指针）=====
//     int active_start = 0;
//     int active_end = init_N; // [start, end)
//     string gt_output_dir = output_dir + "/build_" + to_string(int(init_ratio * 100)) + "/gt_" + to_string(int(update_ratio * 100));
//     std::cout << "gt_output_dir: " << gt_output_dir << std::endl;
//     // fs::create_directories(gt_output_dir);

//     // ================= 更新循环 =================
//     for (int round = 0; round <= update_rounds; round++)
//     {
//         cout << "\n=== Round " << round
//              << " | Active range: [" << active_start
//              << ", " << active_end - 1 << "] ===" << endl;

//         string round_dir = gt_output_dir + "/roundd_" +
//                            (round < 10 ? "0" + to_string(round) : to_string(round));
//         // fs::create_directories(round_dir);

//         int active_size = active_end - active_start;
//         vector<vector<int>> topk_indices(nq, vector<int>(topk));

// #pragma omp parallel for schedule(dynamic)
//         for (int i = 0; i < nq; i++)
//         {
//             const float *q_img = query_img[i].data();
//             const float *q_txt = query_text[i].data();
//             float a = alpha[i];

//             vector<pair<float, int>> scores(active_size);

//             for (int j = 0; j < active_size; j++)
//             {
//                 int idx = active_start + j;
//                 float d_img = euclidean(q_img, base_img[idx].data(), base_dim);
//                 float d_txt = euclidean(q_txt, base_text[idx].data(), text_dim);
//                 float score = a * d_img + (1.0f - a) * d_txt;
//                 scores[j] = {score, idx};
//             }

//             nth_element(scores.begin(), scores.begin() + topk, scores.end());
//             sort(scores.begin(), scores.begin() + topk);

//             for (int k = 0; k < topk; k++)
//                 topk_indices[i][k] = scores[k].second;
//         }

//         write_ivecs(round_dir + "_top10_results.ivecs", topk_indices);
//         cout << "Saved GT: " << round_dir << "_top10_results.ivecs" << endl;

//         // ===== 更新 active 集合 =====
//         if (round == update_rounds)
//             break;

//         if (mode == "insert")
//         {
//             active_end = min(active_end + step, N);
//         }
//         else if (mode == "delete")
//         {
//             active_start = min(active_start + step, active_end);
//         }
//         else
//         {
//             cerr << "Unknown mode: " << mode << endl;
//             return 1;
//         }
//     }

//     cout << "\nAll dynamic GT generation done.\n";
//     return 0;
// }

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <string>

using namespace std;

using idx_t = uint32_t;

// =====================================================
// read fvecs
// =====================================================
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
        fin.read((char *)v.data(), dim * sizeof(float));
        if (!fin)
            break;

        data.push_back(move(v));
    }

    return data;
}

// =====================================================
// read alpha (fvecs style: dim=1 + value)
// =====================================================
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
        if (!fin)
            break;

        alpha.push_back(v);
    }

    return alpha;
}

// =====================================================
// write ivecs
// =====================================================
void write_ivecs(const string &filename,
                 const vector<vector<idx_t>> &gt)
{
    ofstream out(filename, ios::binary);

    for (auto &row : gt)
    {
        int dim = row.size();
        out.write((char *)&dim, sizeof(int));
        out.write((char *)row.data(), dim * sizeof(idx_t));
    }
}

// =====================================================
// L2 (no sqrt !!!)
// =====================================================
inline float l2(const float *a,
                const float *b,
                int dim)
{
    float dist = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

// =====================================================
// main
// =====================================================
int main(int argc, char **argv)
{
    if (argc != 8)
    {
        cerr << "Usage:\n"
             << "./gen_gt "
             << "base_img.fvecs "
             << "query_img.fvecs "
             << "base_text.fvecs "
             << "query_text.fvecs "
             << "alpha.fvecs "
             << "output.ivecs "
             << "num_threads\n";
        return -1;
    }

    string base_img_path = argv[1];
    string query_img_path = argv[2];
    string base_text_path = argv[3];
    string query_text_path = argv[4];
    string alpha_path = argv[5];
    string output_path = argv[6];
    int num_threads = atoi(argv[7]);

    omp_set_num_threads(num_threads);

    // =====================================================
    // load data
    // =====================================================
    auto base_img = read_fvecs(base_img_path);
    auto query_img = read_fvecs(query_img_path);
    auto base_text = read_fvecs(base_text_path);
    auto query_text = read_fvecs(query_text_path);
    auto alpha = read_alpha(alpha_path);

    int nb = base_img.size();
    int nq = query_img.size();
    int K = 10;

    if ((int)alpha.size() != nq)
    {
        cerr << "alpha size mismatch\n";
        return -1;
    }

    int dim_img = base_img[0].size();
    int dim_text = base_text[0].size();

    cout << "nb = " << nb << " nq = " << nq << endl;

    // =====================================================
    // result
    // =====================================================
    vector<vector<idx_t>> gt_ids(nq, vector<idx_t>(K));

    // =====================================================
    // brute force
    // =====================================================
#pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < nq; qi++)
    {
        float a = alpha[qi];

        const float *q_img = query_img[qi].data();
        const float *q_text = query_text[qi].data();

        vector<pair<float, idx_t>> dist;
        dist.reserve(nb);

        for (int bi = 0; bi < nb; bi++)
        {
            float d_img = l2(q_img, base_img[bi].data(), dim_img);
            float d_text = l2(q_text, base_text[bi].data(), dim_text);

            float score = a * d_img + (1.0f - a) * d_text;

            dist.emplace_back(score, bi);
        }

        // ===== FIXED nth_element =====
        nth_element(dist.begin(),
                    dist.begin() + K,
                    dist.end());

        sort(dist.begin(), dist.begin() + K);

        for (int k = 0; k < K; k++)
            gt_ids[qi][k] = dist[k].second;
    }

    // =====================================================
    // save
    // =====================================================
    write_ivecs(output_path, gt_ids);

    cout << "saved: " << output_path << endl;

    return 0;
}