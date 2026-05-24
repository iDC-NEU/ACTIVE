#include <builder.h>
#include <set_para.h>
#include <iostream>
#include <string>
#include "parameters.h"
#include "policy.h"
#include "json.hpp"
#include <regex>
using json = nlohmann::json;

void DEG(stkq::Parameters& parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    std::string query_emb_path = "";
    std::string query_loc_path = "";
    std::string query_alpha_path = "";
    std::string ground_path = "";

    unsigned id_flag = parameters.get<unsigned>("id_flag");
    auto* builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));

    builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters, false, true);
    builder->peak_memory_footprint();
    builder->load_graph(stkq::TYPE::INDEX_DEG, &graph_file[0], parameters);
    builder->peak_memory_footprint();
    // ========================
    // 生成 save_path
    // ========================
    std::string save_path;

    // 获取 trace_path 和 round_id
    std::string trace_path = parameters.get<std::string>("trace_path");
    std::regex round_pattern(R"(round_(\d+)_trace)");
    std::smatch match;
    std::string round_id = "00"; // 默认值
    if (std::regex_search(trace_path, match, round_pattern))
    {
        round_id = match[1]; // e.g., "00", "01"...
    }
    unsigned r = std::stoi(round_id);
    parameters.set<unsigned>("current_rounds", r);
    // 获取原始 graph_file
    std::string base_graph_index = graph_file;

    // 循环去掉末尾所有 "_数字" 后缀
    while (true)
    {
        size_t pos = base_graph_index.rfind('_');
        if (pos == std::string::npos)
            break;

        std::string suffix = base_graph_index.substr(pos + 1);
        // 判断是否全是数字
        bool all_digit = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
        if (!all_digit)
            break;

        // 去掉末尾 "_数字"
        base_graph_index = base_graph_index.substr(0, pos);
    }

    // 拼接新的 round_id
    save_path = base_graph_index + "_" + round_id + "_cmp";
    int rid = std::stoi(round_id);
    std::cout << "Output_graph_index: " << save_path << std::endl;
    unsigned argc = parameters.get<unsigned>("argc");
    if (argc > 9)
    {

        // std::cout<<parameters.get<unsigned>("K")<<" "<<parameters.get<unsigned>("L")<<std::endl;
        std::cout << "Start Search( " << rid - 1 << " iter)" << std::endl;
        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, stkq::TYPE::L_UPDATE_ASCEND, parameters);
        std::cout << std::endl;
    }
    std::cout << "Start Update" << std::endl;
    builder->update(stkq::UPDATE_DEG, id_flag, parameters)->save_graph(stkq::TYPE::INDEX_DEG, &save_path[0]);
    builder->peak_memory_footprint();
}

int set(json data, stkq::Parameters& parameters)
{
    try
    {
        data.value("id_flag", 0);
        parameters.set<unsigned>("max_m", data.at("max_edges").get<unsigned>());
        parameters.set<unsigned>("ef_construction", data.at("ef_spatial").get<unsigned>());
        parameters.set<std::string>("graph_path", data.at("graph_path").get<std::string>());
        parameters.set<unsigned>("n_threads", 8);

        parameters.set<float>("max_spatial_distance", 1);
        parameters.set<float>("max_emb_distance", 1);

        parameters.set<std::string>("base_emb_path", data.at("base_emb_path").get<std::string>());
        parameters.set<std::string>("base_loc_path", data.at("base_loc_path").get<std::string>());
        parameters.set<std::string>("query_emb_path", data.at("query_emb_path").get<std::string>());
        parameters.set<std::string>("query_loc_path", data.at("query_loc_path").get<std::string>());
        parameters.set<std::string>("query_alpha_path", data.at("query_alpha_path").get<std::string>());
        parameters.set<std::string>("ground_path", data.at("ground_path").get<std::string>());
        parameters.set<std::string>("trace_path", data.at("trace_path").get<std::string>());
        parameters.set<unsigned>("id_flag", data.value("id_flag", 0));
    }
    catch (json::type_error& e)
    {
        std::cerr << "type error: " << e.what() << std::endl;
        return -1;
    }
    catch (json::out_of_range& e)
    {
        std::cerr << "Necessary configuration items are missing: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 9 && argc != 16)
    {
        std::cout << argc << " " << *argv << std::endl;
        std::cout << "./update grapg_path base_emb_path base_loc_path trace_path ef(L) max_edges(R) thread id_flag"
            << std::endl;
        exit(-1);
    }
    std::string graph_file(argv[1]);
    std::string base_emb_path(argv[2]);
    std::string base_loc_path(argv[3]);
    std::string trace_file(argv[4]);
    std::string ef_construction(argv[5]);
    std::string max_m(argv[6]);
    std::string thread(argv[7]);
    std::string id_flag(argv[8]);
    stkq::Parameters parameters;
    parameters.set<unsigned>("argc", argc);
    if (argc > 9)
    {
        unsigned int L = static_cast<unsigned int>(std::stoul(argv[9]));
        unsigned int K = static_cast<unsigned int>(std::stoul(argv[10]));

        parameters.set<unsigned>("L", L);
        parameters.set<unsigned>("K", K);
        parameters.set<std::string>("query_alpha", argv[11]);
        parameters.set<std::string>("query_emb", argv[12]);
        parameters.set<std::string>("query_loc", argv[13]);
        parameters.set<std::string>("query_gt", argv[14]);
    }

    parameters.set<std::string>("exc_type", "update");
    parameters.set<float>("max_spatial_distance", 1);
    parameters.set<float>("max_emb_distance", 1);

    parameters.set<std::string>("graph_file", graph_file);
    parameters.set<std::string>("base_emb_path", base_emb_path);
    parameters.set<std::string>("base_loc_path", base_loc_path);
    parameters.set<std::string>("trace_path", trace_file);
    parameters.set<unsigned>("n_threads", std::stoi(thread));
    parameters.set<unsigned>("max_m", std::stoi(max_m));
    parameters.set<unsigned>("ef_construction", std::stoi(ef_construction));
    parameters.set<unsigned>("id_flag", std::stoi(id_flag));
    parameters.set<unsigned>("delete_mode", std::stoi(argv[15]));
    parameters.set<unsigned>("update_rounds", 1);

    DEG(parameters);
    return 0;
}