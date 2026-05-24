#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <filesystem>

#include <omp.h>

using namespace std;
namespace fs = std::filesystem;

// ================= 工具函数 =================

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

        if (dim != 1)
        {
            cerr << "Alpha file dim should be 1, but got " << dim << endl;
            exit(1);
        }

        float v;
        fin.read((char *)&v, 4);
        if (!fin)
            break;

        alpha.push_back(v);
    }

    return alpha;
}

void write_ivecs(const string &filename, const vector<vector<int>> &data)
{
    ofstream fout(filename, ios::binary);
    if (!fout)
    {
        cerr << "Cannot write file " << filename << endl;
        exit(1);
    }

    for (const auto &row : data)
    {
        int dim = row.size();
        fout.write((char *)&dim, 4);
        fout.write((char *)row.data(), dim * 4);
    }
}

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

// ================= 输出当前 top-k =================

void save_current_topk(
    const vector<vector<pair<float, int>>> &current_topk,
    const string &output_file)
{
    int nq = current_topk.size();
    int topk = current_topk[0].size();

    vector<vector<int>> topk_indices(nq, vector<int>(topk));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < nq; i++)
    {
        for (int k = 0; k < topk; k++)
        {
            topk_indices[i][k] = current_topk[i][k].second;
        }
    }

    write_ivecs(output_file, topk_indices);
    cout << "Saved GT: " << output_file << endl;
}

// ================= 初始化计算前 50% GT =================

void compute_initial_topk(
    const vector<vector<float>> &base_img,
    const vector<vector<float>> &base_text,
    const vector<vector<float>> &query_img,
    const vector<vector<float>> &query_text,
    const vector<float> &alpha,
    int init_N,
    int topk,
    vector<vector<pair<float, int>>> &current_topk)
{
    int nq = query_img.size();
    int base_dim = base_img[0].size();
    int text_dim = base_text[0].size();

    current_topk.assign(nq, vector<pair<float, int>>(topk));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nq; i++)
    {
        const float *q_img = query_img[i].data();
        const float *q_txt = query_text[i].data();
        float a = alpha[i];

        vector<pair<float, int>> scores(init_N);

        for (int j = 0; j < init_N; j++)
        {
            float d_img = euclidean(q_img, base_img[j].data(), base_dim);
            float d_txt = euclidean(q_txt, base_text[j].data(), text_dim);
            float score = a * d_img + (1.0f - a) * d_txt;

            scores[j] = {score, j};
        }

        nth_element(scores.begin(), scores.begin() + topk, scores.end());
        sort(scores.begin(), scores.begin() + topk);

        for (int k = 0; k < topk; k++)
        {
            current_topk[i][k] = scores[k];
        }
    }
}

// ================= 插入新 block，并更新 top-k =================

void update_topk_by_insert_block(
    const vector<vector<float>> &base_img,
    const vector<vector<float>> &base_text,
    const vector<vector<float>> &query_img,
    const vector<vector<float>> &query_text,
    const vector<float> &alpha,
    int insert_start,
    int insert_end,
    int topk,
    vector<vector<pair<float, int>>> &current_topk)
{
    int nq = query_img.size();
    int base_dim = base_img[0].size();
    int text_dim = base_text[0].size();

    int insert_size = insert_end - insert_start;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nq; i++)
    {
        const float *q_img = query_img[i].data();
        const float *q_txt = query_text[i].data();
        float a = alpha[i];

        vector<pair<float, int>> candidates;
        candidates.reserve(topk + insert_size);

        // 旧数据中只保留当前 top-k 即可
        for (int k = 0; k < topk; k++)
        {
            candidates.push_back(current_topk[i][k]);
        }

        // 只计算新插入的 1%
        for (int idx = insert_start; idx < insert_end; idx++)
        {
            float d_img = euclidean(q_img, base_img[idx].data(), base_dim);
            float d_txt = euclidean(q_txt, base_text[idx].data(), text_dim);
            float score = a * d_img + (1.0f - a) * d_txt;

            candidates.push_back({score, idx});
        }

        nth_element(candidates.begin(), candidates.begin() + topk, candidates.end());
        sort(candidates.begin(), candidates.begin() + topk);

        for (int k = 0; k < topk; k++)
        {
            current_topk[i][k] = candidates[k];
        }
    }
}

// ================= 主逻辑 =================

int main(int argc, char **argv)
{
    if (argc != 11)
    {
        cerr << "Usage:\n"
             << "./insert_gt_opt \\\n"
             << "  base_img.fvecs \\\n"
             << "  base_text.fvecs \\\n"
             << "  query_img.fvecs \\\n"
             << "  query_text.fvecs \\\n"
             << "  alpha_file.fvecs \\\n"
             << "  output_dir \\\n"
             << "  init_ratio \\\n"
             << "  insert_ratio \\\n"
             << "  insert_rounds \\\n"
             << "  threads\n";
        return 1;
    }

    string base_img_file = argv[1];
    string base_text_file = argv[2];
    string query_img_file = argv[3];
    string query_text_file = argv[4];
    string alpha_file = argv[5];
    string output_dir = argv[6];

    float init_ratio = stof(argv[7]);
    float insert_ratio = stof(argv[8]);
    int insert_rounds = stoi(argv[9]);
    int threads = stoi(argv[10]);

    omp_set_num_threads(threads);

    cout << "Threads: " << threads << endl;
    cout << "Start reading data..." << endl;

    auto base_img = read_fvecs(base_img_file);
    auto base_text = read_fvecs(base_text_file);
    auto query_img = read_fvecs(query_img_file);
    auto query_text = read_fvecs(query_text_file);
    auto alpha = read_alpha(alpha_file);

    int N = base_img.size();
    int nq = query_img.size();
    int topk = 10;

    if ((int)base_text.size() != N)
    {
        cerr << "Base img/text size mismatch!" << endl;
        return 1;
    }

    if ((int)query_text.size() != nq)
    {
        cerr << "Query img/text size mismatch!" << endl;
        return 1;
    }

    if ((int)alpha.size() != nq)
    {
        cerr << "Alpha size mismatch with query size!" << endl;
        return 1;
    }

    int init_N = max(topk, (int)(N * init_ratio));
    int step = max(1, (int)(N * insert_ratio));

    cout << "Total base: " << N << endl;
    cout << "Query size: " << nq << endl;
    cout << "Initial active: " << init_N << endl;
    cout << "Insert step: " << step << endl;
    cout << "Insert rounds: " << insert_rounds << endl;

    string gt_output_dir =
        output_dir + "/build_" + to_string(int(init_ratio * 100)) +
        "/gt_insert_" + to_string(int(insert_ratio * 100));

    fs::create_directories(gt_output_dir);

    cout << "gt_output_dir: " << gt_output_dir << endl;

    vector<vector<pair<float, int>>> current_topk;

    // ===== 初始化计算 50% GT =====
    cout << "\n=== Initial GT computation: active range [0, "
         << init_N - 1 << "] ===" << endl;

    compute_initial_topk(
        base_img,
        base_text,
        query_img,
        query_text,
        alpha,
        init_N,
        topk,
        current_topk);

    int active_end = init_N;

    // 保存初始 GT
    save_current_topk(
        current_topk,
        gt_output_dir + "/round_00_before_top10_results.ivecs");

    // ===== 插入循环 =====
    for (int round = 0; round < insert_rounds; round++)
    {
        string round_str = round < 10 ? "0" + to_string(round) : to_string(round);

        // before 文件：直接保存当前 top-k，不重新计算距离
        string before_file =
            gt_output_dir + "/round_" + round_str + "_before_top10_results.ivecs";

        if (round != 0)
        {
            save_current_topk(current_topk, before_file);
        }

        int insert_start = active_end;
        int insert_end = min(active_end + step, N);

        if (insert_start >= insert_end)
        {
            cout << "No more vectors to insert. Stop." << endl;
            break;
        }

        cout << "\n=== Round " << round
             << " | Insert range: [" << insert_start
             << ", " << insert_end - 1 << "]"
             << " | Insert size: " << insert_end - insert_start
             << " ===" << endl;

        // 只计算新插入 1% 的距离，并更新 current_topk
        update_topk_by_insert_block(
            base_img,
            base_text,
            query_img,
            query_text,
            alpha,
            insert_start,
            insert_end,
            topk,
            current_topk);

        active_end = insert_end;

        string after_file =
            gt_output_dir + "/round_" + round_str + "_after_top10_results.ivecs";

        save_current_topk(current_topk, after_file);

        if (active_end >= N)
        {
            cout << "All base vectors have been inserted. Stop early." << endl;
            break;
        }
    }

    cout << "\nAll optimized insert GT generation done.\n";

    return 0;
}