// make_traces.cpp
// Usage: ./make_traces <total_base> <update_size> <rounds> <out_path> [mode]
// mode: insert | delete | both   (default: both)
//
// Binary format:
//   both   : [int32 N][del_ids N*int32][ins_ids N*int32]
//            (N is #deletes AND also #inserts; insert=delete=N)
//   delete : [int32 N][del_ids N*int32]
//            (N is #deletes)
//   insert : [int32 N][ins_ids N*int32]
//            (N is #inserts)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cstdint>
#include <algorithm>
#include <sstream>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <total_base> <update_ratio> <rounds> <out_path> [mode]\n";
        cerr << "  mode: insert | delete | both (default: both)\n";
        return 1;
    }

    int64_t total_base = atoll(argv[1]);
    float update_ratio = atof(argv[2]);
    int32_t update_size = int32_t(update_ratio * total_base);
    int32_t rounds = atoi(argv[3]);
    string out_path = string(argv[4]);
    if (!out_path.empty() && out_path.back() != '/' && out_path.back() != '\\')
        out_path += "/";

    string mode = "both";
    if (argc >= 6)
        mode = string(argv[5]);

    if (mode != "insert" && mode != "delete" && mode != "both")
    {
        cerr << "Error: invalid mode: " << mode << "\n";
        cerr << "  mode must be one of: insert | delete | both\n";
        return 1;
    }

    // create output trace directory
    string trace_dir = out_path + "trace_" + to_string(int(update_ratio * 100)) + "/";
    // fs::create_directories(trace_dir);

    int64_t window_size = (int64_t)(total_base * (1 - update_ratio * rounds)); // initial 90%
    if (window_size < 0)
        window_size = 0;

    cout << "mode=" << mode << ", total_base=" << total_base << ", window_size=" << window_size
         << ", update_size=" << update_size << ", rounds=" << rounds << "\n";
    cout << "Trace files will be written to: " << trace_dir << "\n";

    for (int32_t r = 0; r < rounds; ++r)
    {
        // compute delete ids: starting from r*update_size (simulate sliding)
        int64_t del_start = (int64_t)r * update_size;
        vector<int32_t> del_ids;
        int64_t del_end = -1;
        for (int64_t i = del_start; i < del_start + update_size && i < total_base; ++i)
        {
            del_ids.push_back((int32_t)i);
            del_end = i;
        }

        // compute insert ids: from window_size + r*update_size
        int64_t ins_start = window_size + (int64_t)r * update_size;
        vector<int32_t> ins_ids;
        int64_t ins_end = -1;
        for (int64_t i = ins_start; i < ins_start + update_size && i < total_base; ++i)
        {
            ins_ids.push_back((int32_t)i);
            ins_end = i;
        }

        // N is the count for ONE operation list:
        // - both: N = #deletes = #inserts (we enforce equality by taking min)
        // - delete: N = #deletes
        // - insert: N = #inserts
        int32_t N = 0;
        if (mode == "both")
            N = (int32_t)min<int64_t>(del_ids.size(), ins_ids.size());
        else if (mode == "delete")
            N = (int32_t)del_ids.size();
        else // insert
            N = (int32_t)ins_ids.size();

        // file name
        ostringstream oss;
        oss << trace_dir << "round_" << (r < 10 ? "0" + to_string(r) : to_string(r)) << "_trace.bin";
        string fname = oss.str();

        ofstream fout(fname, ios::binary);
        if (!fout)
        {
            cerr << "Error: cannot open " << fname << " for write\n";
            return 1;
        }

        // write N first (as requested: in both mode, it's the insert OR delete count)
        fout.write(reinterpret_cast<const char *>(&N), sizeof(int32_t));

        if (N > 0)
        {
            if (mode == "both")
            {
                std::cout << mode << std::endl;
                fout.write(reinterpret_cast<const char *>(del_ids.data()), sizeof(int32_t) * N);
                fout.write(reinterpret_cast<const char *>(ins_ids.data()), sizeof(int32_t) * N);
            }
            else if (mode == "delete")
            {
                std::cout << mode << std::endl;
                fout.write(reinterpret_cast<const char *>(del_ids.data()), sizeof(int32_t) * N);
            }
            else // insert
            {
                std::cout << mode << std::endl;
                fout.write(reinterpret_cast<const char *>(ins_ids.data()), sizeof(int32_t) * N);
            }
        }

        fout.close();

        cout << "Wrote " << fname << " (N=" << N
             << ", dels=" << del_ids.size() << ", ins=" << ins_ids.size() << ")\n";
        cout << "del: " << del_start << " " << del_end
             << " ins: " << ins_start << " " << ins_end << "\n";
    }

    cout << "All traces generated.\n";
    return 0;
}
