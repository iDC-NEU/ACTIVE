#include <builder.h>
#include <set_para.h>
#include <iostream>

void DEG(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb");
    std::string base_loc_path = parameters.get<std::string>("base_loc");
    std::string query_emb_path = "";
    std::string query_loc_path = "";
    std::string query_alpha_path = "";
    std::string ground_path = "";

    std::string graph_file = parameters.get<std::string>("graph_index");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_dis"), parameters.get<float>("max_loc_dis"));
    if (parameters.get<std::string>("exc_type") == "search")
    {
        query_emb_path = parameters.get<std::string>("query_emb");
        query_loc_path = parameters.get<std::string>("query_loc");
        query_alpha_path = parameters.get<std::string>("query_alpha");
        ground_path = parameters.get<std::string>("query_gt");
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_DEG, &graph_file[0], parameters);
        builder->peak_memory_footprint();
        int search_type = parameters.get<unsigned>("search_type");
        // 1-one times 0-from k to max_L
        stkq::TYPE s_type = (search_type == 1) ? stkq::TYPE::L_UPDATE_ASCEND : stkq::TYPE::L_SEARCH_ASCEND;
        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, s_type, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

int main(int argc, char **argv)
{

    if (argc != 14)
    {
        std::cout << "./search vec_base_emb vec_base_loc graph_index L K alpha_file query_base_emb query_base_loc query_gt threads max_emb_dis max_loc_dis"
                  << std::endl;
        exit(-1);
    }

    stkq::Parameters parameters;
    parameters.set<std::string>("exc_type", "search");
    parameters.set<std::string>("base_emb", argv[1]);
    parameters.set<std::string>("base_loc", argv[2]);
    parameters.set<std::string>("graph_index", argv[3]);
    std::string exc_type = parameters.get<std::string>("exc_type");
    std::cout << "exc_type: " << parameters.get<std::string>("exc_type")
              << ", base_emb: " << parameters.get<std::string>("base_emb")
              << ", base_loc: " << parameters.get<std::string>("base_loc")
              << ", graph_index: " << parameters.get<std::string>("graph_index");
    if (exc_type == "search")
    {
        unsigned int L = static_cast<unsigned int>(std::stoul(argv[4]));
        unsigned int K = static_cast<unsigned int>(std::stoul(argv[5]));

        parameters.set<unsigned>("L", L);
        parameters.set<unsigned>("K", K);
        parameters.set<std::string>("query_alpha", argv[6]);
        parameters.set<std::string>("query_emb", argv[7]);
        parameters.set<std::string>("query_loc", argv[8]);
        parameters.set<std::string>("query_gt", argv[9]);
        parameters.set<unsigned>("id_flag", 1);
        unsigned int threads = static_cast<unsigned int>(std::stoul(argv[10]));
        parameters.set<unsigned>("n_threads", threads);

        unsigned int max_emb_dis = static_cast<unsigned int>(std::stoul(argv[11]));
        unsigned int max_loc_dis = static_cast<unsigned int>(std::stoul(argv[12]));
        unsigned int search_type = static_cast<unsigned int>(std::stoul(argv[13]));
        parameters.set<unsigned>("search_type", search_type);
        parameters.set<unsigned>("max_emb_dis", max_emb_dis);
        parameters.set<unsigned>("max_loc_dis", max_loc_dis);
        parameters.set<unsigned>("max_m", 0);
        std::cout << ", L: " << parameters.get<unsigned>("L")
                  << ", K: " << parameters.get<unsigned>("K")
                  << ", query_alpha: " << parameters.get<std::string>("query_alpha")
                  << ", query_emb: " << parameters.get<std::string>("query_emb")
                  << ", query_loc: " << parameters.get<std::string>("query_loc")
                  << ", query_gt: " << parameters.get<std::string>("query_gt")
                  << ", threads: " << parameters.get<unsigned>("n_threads")
                  << ", search_type: " << parameters.get<unsigned>("search_type");
    }
    parameters.set<int>("mult", -1);
    DEG(parameters);

    return 0;
}