#include <builder.h>
#include <set_para.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "parameters.h"
#include "policy.h"

//////////////////////////////////////////////////////////////
// split helper
//////////////////////////////////////////////////////////////

static std::vector<std::string>
split(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim))
        result.emplace_back(std::move(item));

    return result;
}

//////////////////////////////////////////////////////////////
// Multi-round DEG update
//////////////////////////////////////////////////////////////

void DEG(stkq::Parameters &parameters)
{
    unsigned num_threads = parameters.get<unsigned>("n_threads");

    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string graph_file = parameters.get<std::string>("graph_file");

    //////////////////////////////////////////////////////////
    // 解析 trace CSV
    //////////////////////////////////////////////////////////

    auto trace_csv =
        parameters.get<std::string>("trace_files");

    auto gt_csv =
        parameters.get<std::string>("gt_files");
    auto gt_files =
        split(gt_csv, ',');

    auto trace_files =
        split(trace_csv, ',');

    unsigned id_flag =
        parameters.get<unsigned>("id_flag");

    //////////////////////////////////////////////////////////
    // RAII Builder
    //////////////////////////////////////////////////////////

    stkq::IndexBuilder builder(
        num_threads,
        parameters.get<float>("max_emb_distance"),
        parameters.get<float>("max_spatial_distance"));

    //////////////////////////////////////////////////////////
    // LOAD ONCE
    //////////////////////////////////////////////////////////

    builder.load(
        const_cast<char *>(base_emb_path.c_str()),
        const_cast<char *>(base_loc_path.c_str()),
        nullptr, nullptr, nullptr, nullptr,
        parameters,
        false,
        true);

    builder.peak_memory_footprint();

    builder.load_graph(
        stkq::TYPE::INDEX_DEG,
        const_cast<char *>(graph_file.c_str()),
        parameters);

    builder.peak_memory_footprint();

    std::cout << "\n====================================\n"
                 " Multi-Round Streaming "
              << id_flag << " Start\n"
                            " Total Rounds: "
              << trace_files.size()
              << "\n====================================\n";

    //////////////////////////////////////////////////////////
    // MULTI ROUND UPDATE
    //////////////////////////////////////////////////////////

    for (size_t r = 0; r < trace_files.size(); ++r)
    {
        std::cout << "\n=========== ROUND "
                  << r
                  << " ===========" << std::endl;

        parameters.set<std::string>(
            "trace_path",
            trace_files[r]);
        std::string trace_path = parameters.get<std::string>("trace_path");
        builder.load_trace_(&trace_path[0], id_flag, parameters);

        parameters.set<std::string>(
            "query_gt",
            gt_files[r]);
        std::string query_gt = parameters.get<std::string>("query_gt");
        std::cout << "Start Loading Ground Truth:"
                  << query_gt << std::endl;
        builder.load_gt_(&query_gt[0], parameters);

        parameters.set<unsigned>("current_rounds", r);

        std::cout << "Start Search( " << r << " iter)" << std::endl;
        builder.search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, stkq::TYPE::L_UPDATE_ASCEND, parameters);
        std::cout << std::endl;

        builder.update(
            stkq::UPDATE_DEG,
            id_flag,
            parameters);

        // builder.update_graph(parameters);
        auto fut = builder.update_graph_async(parameters);
        while (!builder.final_index_->is_search_graph_finished.load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "search graph update over!" << std::endl;
        int an = builder.average_neighbor_size;
        int acn = builder.active_nodes;
        if (acn != 0)
            std::cout << "average_neighbor_size: " << an / acn << std::endl;
        std::cout << "active_nodes: " << acn << std::endl;
    }

    parameters.set<std::string>(
        "query_gt",
        gt_files[trace_files.size()]);
    std::string query_gt = parameters.get<std::string>("query_gt");
    std::cout << "Start Loading Ground Truth:"
              << query_gt << std::endl;
    builder.load_gt_(&query_gt[0], parameters);

    std::cout << "Start Search( " << trace_files.size() << " iter)" << std::endl;
    builder.search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, stkq::TYPE::L_UPDATE_ASCEND, parameters);

    // builder.search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, stkq::TYPE::L_SEARCH_ASCEND, parameters);
    std::cout << std::endl;

    //////////////////////////////////////////////////////////
    // SAVE ONCE
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

    // const std::string save_path =
    //     graph_file + "_final";

    // std::cout << "\nSaving final graph:\n"
    //           << save_path << std::endl;

    // builder.save_graph(
    //     stkq::TYPE::INDEX_DEG,
    //     const_cast<char *>(save_path.c_str()));

    // builder.peak_memory_footprint();
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    /**
     * ./update_multi
     * graph
     * base_emb
     * base_loc
     * trace1 trace2 ...
     * ef
     * max_edges
     * threads
     * id_flag
     * delete_mode
     */

    if (argc < 10)
    {
        std::cout << "\nUsage:\n"
                     "./update_multi graph base_emb base_loc "
                     "trace1 [trace2 ...] "
                     "ef max_edges threads id_flag delete_mode\n";

        return -1;
    }
    std::cout << "Argc Num: " << argc << std::endl;

    constexpr int tail = 11;

    int num_traces = (argc - 4 - tail) / 2;
    int num_gt = num_traces + 1;

    if (num_traces <= 0)
    {
        std::cerr << "ERROR: No trace files provided.\n";
        return -1;
    }

    //////////////////////////////////////////////////////////
    // 拼接 CSV（关键！！）
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

    std::string trace_csv, gt_csv;

    for (int i = 4; i < 4 + num_traces; ++i)
    {
        if (i > 4)
            trace_csv += ",";

        trace_csv += argv[i];
    }
    for (int i = 4 + num_traces; i < 4 + num_traces + num_gt; ++i)
    {
        if (i > 4 + num_traces)
            gt_csv += ",";

        gt_csv += argv[i];
    }
    int offset = 4 + num_traces + num_gt;

    unsigned ef = std::stoul(argv[offset]);
    unsigned max_edges = std::stoul(argv[offset + 1]);
    unsigned threads = std::stoul(argv[offset + 2]);
    unsigned id_flag = std::stoul(argv[offset + 3]);
    unsigned delete_mode = std::stoul(argv[offset + 4]);
    unsigned L = std::stoul(argv[offset + 5]);
    unsigned K = std::stoul(argv[offset + 6]);
    int angle = std::stoul(argv[offset + 7]);
    std::string query_alpha = argv[offset + 8];
    std::string query_emb = argv[offset + 9];
    std::string query_loc = argv[offset + 10];

    std::cout << "Init Angle: " << angle << std::endl;
    //////////////////////////////////////////////////////////
    // Parameters
    //////////////////////////////////////////////////////////

    stkq::Parameters parameters;

    parameters.set<std::string>("exc_type", "update");
    parameters.set<unsigned>("update_rounds", num_traces);
    parameters.set<int>("angle", angle);
    // std::cout << "Total update rounds: " << num_traces << std::endl;
    parameters.set<float>("max_spatial_distance", 1);
    parameters.set<float>("max_emb_distance", 1);

    parameters.set<std::string>("graph_file", argv[1]);
    parameters.set<std::string>("base_emb_path", argv[2]);
    parameters.set<std::string>("base_loc_path", argv[3]);

    parameters.set<unsigned>("n_threads", threads);
    parameters.set<unsigned>("max_m", max_edges);
    parameters.set<unsigned>("ef_construction", ef);
    parameters.set<unsigned>("id_flag", id_flag);
    parameters.set<unsigned>("delete_mode", delete_mode);
    parameters.set<unsigned>("argc", argc);
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("K", K);
    parameters.set<std::string>("query_alpha", query_alpha);
    parameters.set<std::string>("query_emb", query_emb);
    parameters.set<std::string>("query_loc", query_loc);

    // ⭐⭐⭐⭐⭐ 核心修复
    parameters.set<std::string>(
        "trace_files",
        trace_csv);

    parameters.set<std::string>(
        "gt_files",
        gt_csv);

    //////////////////////////////////////////////////////////
    // RUN
    //////////////////////////////////////////////////////////

    DEG(parameters);

    return 0;
}