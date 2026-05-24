#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <string>
// #define threads 64
using namespace std;

// ------------------- 工具函数 -------------------

// 读取 fvecs
vector<vector<float>> read_fvecs(const string &filename)
{
    ifstream fin(filename, ios::binary);
    if (!fin)
    {
        cerr << "Error: cannot open " << filename << endl;
        exit(1);
    }

    vector<vector<float>> data;

    while (true)
    {
        int dim = 0;
        fin.read((char *)&dim, 4);

        if (!fin)
            break;

        vector<float> v(dim);
        fin.read((char *)v.data(), 4 * dim);

        if (!fin)
            break;

        data.push_back(move(v));
    }

    return data;
}

// 读取 alpha 文件
vector<float> read_alpha_file(const string &filename)
{
    ifstream fin(filename, ios::binary);

    if (!fin)
    {
        cerr << "Error: cannot open alpha file " << filename << endl;
        exit(1);
    }

    vector<float> alpha;

    while (true)
    {
        int dim = 0;
        fin.read((char *)&dim, 4);

        if (!fin)
            break;

        float val;
        fin.read((char *)&val, 4);

        alpha.push_back(val);
    }

    return alpha;
}

// 写 ivecs
void write_ivecs(const string &filename,
                 const vector<vector<int>> &data)
{
    ofstream fout(filename, ios::binary);

    for (auto &row : data)
    {
        int dim = row.size();

        fout.write((char *)&dim, 4);
        fout.write((char *)row.data(), 4 * dim);
    }
}

// 欧氏距离（去掉 sqrt 提升性能）
inline float euclidean_sq(const float *a,
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

// ------------------- 主逻辑 -------------------

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cerr << "Usage: ./ptopk_alpha_file "
             << "<dataset_path> "
             << "<alpha_file> "
             << "<update_ratio> "
             << "<rounds>"
             << endl;

        return 1;
    }
    // omp_set_num_threads(threads);
    string source = string(argv[1]) + "/";
    string alpha_file = string(argv[2]);

    int topk = 10;
    int rounds = atoi(argv[4]);

    float update_ratio = atof(argv[3]);
    float build_ratio = 1 - rounds * update_ratio;

    // ------------------- 读取数据 -------------------

    auto base_loc =
        read_fvecs(source + "base_text_emb.fvecs");

    auto base_emb =
        read_fvecs(source + "base_img_emb.fvecs");

    auto query_loc =
        read_fvecs(source + "query_text_emb.fvecs");

    auto query_emb =
        read_fvecs(source + "query_img_emb.fvecs");

    auto alpha =
        read_alpha_file(alpha_file);

    int total_base = base_loc.size();

    int window_size =
        (int)(total_base * build_ratio);

    int update_size =
        max(1, (int)(total_base * update_ratio));

    int nq = query_loc.size();

    if ((int)alpha.size() != nq)
    {
        cerr << "Alpha size mismatch." << endl;
        return 1;
    }

    int base_dim = base_loc[0].size();
    int emb_dim = base_emb[0].size();

    cout << "Total base: " << total_base << endl;
    cout << "Window size: " << window_size << endl;
    cout << "Update size: " << update_size << endl;

    // ------------------- 全局距离缓存 -------------------

    // 使用连续内存
    vector<float> coor_cache(
        (size_t)nq * total_base);

    vector<float> emb_cache(
        (size_t)nq * total_base);

    // 是否已经计算过
    vector<char> computed(
        (size_t)nq * total_base,
        0);

    auto IDX = [&](int qid, int bid)
    {
        return (size_t)qid * total_base + bid;
    };

    // ------------------- Sliding Window -------------------

    int start = 0;
    int end = window_size;

    string source_dir =
        source + "build_" +
        to_string(int(build_ratio * 100));

    for (int round = 0;
         round <= rounds && start < total_base;
         round++)
    {
        cout << "\n=== Round "
             << round
             << " ("
             << start
             << " ~ "
             << end - 1
             << ") ==="
             << endl;

        int window = end - start;

        // ----------------------------------------------------
        // (1) 仅计算新增部分距离
        // ----------------------------------------------------

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nq; i++)
        {
            const float *q1 =
                query_loc[i].data();

            const float *q2 =
                query_emb[i].data();

            for (int j = start; j < end; j++)
            {
                size_t idx = IDX(i, j);

                if (!computed[idx])
                {
                    coor_cache[idx] =
                        euclidean_sq(
                            q1,
                            base_loc[j].data(),
                            base_dim);

                    emb_cache[idx] =
                        euclidean_sq(
                            q2,
                            base_emb[j].data(),
                            emb_dim);

                    computed[idx] = 1;
                }
            }
        }

        // ----------------------------------------------------
        // (2) 生成 top-k
        // ----------------------------------------------------

        vector<vector<int>> topk_indices(
            nq,
            vector<int>(topk));

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nq; i++)
        {
            float a = alpha[i];

            vector<pair<float, int>> local_scores;
            local_scores.reserve(window);

            for (int j = start; j < end; j++)
            {
                size_t idx = IDX(i, j);

                float score =
                    a * emb_cache[idx] + (1.0f - a) * coor_cache[idx];

                local_scores.push_back(
                    {score, j});
            }

            nth_element(
                local_scores.begin(),
                local_scores.begin() + topk,
                local_scores.end());

            sort(
                local_scores.begin(),
                local_scores.begin() + topk);

            for (int k = 0; k < topk; k++)
            {
                topk_indices[i][k] =
                    local_scores[k].second;
            }
        }

        // ----------------------------------------------------
        // (3) 保存结果
        // ----------------------------------------------------

        string round_dir =
            source_dir +
            "/gt_sliding/round_" +
            (round < 10
                 ? "0" + to_string(round)
                 : to_string(round));

        string ivecs_name =
            round_dir +
            "_top10_results.ivecs";

        write_ivecs(
            ivecs_name,
            topk_indices);

        cout << "Saved: "
             << ivecs_name
             << endl;

        // ----------------------------------------------------
        // (4) 滑动窗口更新
        // ----------------------------------------------------

        if (end >= total_base)
            break;

        start += update_size;

        end = min(
            end + update_size,
            total_base);
    }

    cout << "\nAll done.\n";

    return 0;
}