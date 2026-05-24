#include <iostream>
#include <fstream>
// #include <filesystem>
#include <vector>
using namespace std;

// namespace fs = std::filesystem;

/*
Usage:
    ./make_build <dataset_path> <total_vectors> <img_dim> <text_dim>
Example:
    ./make_build ./data 1000000 768 512
*/

void copy_partial(const string &infile, const string &outfile, size_t total_bytes)
{
    ifstream fin(infile, ios::binary);
    ofstream fout(outfile, ios::binary);
    if (!fin)
    {
        cerr << "Error: cannot open input file " << infile << endl;
        exit(1);
    }
    if (!fout)
    {
        cerr << "Error: cannot open output file " << outfile << endl;
        exit(1);
    }

    const size_t BUF_SIZE = 1 << 30; // 1MB
    vector<char> buffer(BUF_SIZE);
    size_t remaining = total_bytes;

    while (remaining > 0 && fin)
    {
        size_t to_read = min(remaining, BUF_SIZE);
        fin.read(buffer.data(), to_read);
        size_t read_bytes = fin.gcount();
        fout.write(buffer.data(), read_bytes);
        remaining -= read_bytes;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        cerr << "Usage: ./make_build <dataset_path> <total_vectors> <img_dim> <text_dim> <update_ratio> " << endl;
        return 1;
    }

    string source = string(argv[1]);
    if (source.back() != '/')
        source += "/";

    int total_vectors = stoi(argv[2]);
    int img_dim = stoi(argv[3]);
    int text_dim = stoi(argv[4]);
    float ratio = stof(argv[5]);

    int build_vectors = static_cast<int>(total_vectors * ratio);

    // 每个向量大小（字节）：4（维度） + 4 * dim
    size_t img_bytes_per_vec = 4 + 4ull * img_dim;
    size_t text_bytes_per_vec = 4 + 4ull * text_dim;

    size_t img_total_bytes = img_bytes_per_vec * build_vectors;
    size_t text_total_bytes = text_bytes_per_vec * build_vectors;

    string img_path = source + "base_img_emb.fvecs";
    string text_path = source + "base_text_emb.fvecs";
    int ratio_int = static_cast<int>(ratio * 100);
    string build_dir = source + "build_" + to_string(ratio_int);
    // fs::create_directories(build_dir);

    string img_out = build_dir + "/base_img_emb.fvecs";
    string text_out = build_dir + "/base_text_emb.fvecs";

    cout << "Total vectors: " << total_vectors << endl;
    cout << "Build vectors: " << build_vectors << " (" << ratio * 100 << "%)" << endl;
    cout << "Image dim: " << img_dim << " -> bytes per vec: " << img_bytes_per_vec << endl;
    cout << "Text  dim: " << text_dim << " -> bytes per vec: " << text_bytes_per_vec << endl;
    cout << "Copying first " << build_vectors << " vectors..." << endl;

    copy_partial(img_path, img_out, img_total_bytes);
    copy_partial(text_path, text_out, text_total_bytes);

    cout << "✅ Build files saved to " << build_dir << endl;
    return 0;
}
