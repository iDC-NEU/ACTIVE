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
    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        builder->load(&base_emb_path[0], &base_loc_path[0], "", "", "", "", parameters)
            ->init(stkq::INIT_DEG)
            ->save_graph(stkq::TYPE::INDEX_DEG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }

}

int main(int argc, char **argv)
{

    if (argc != 9)
    {
        std::cout << "./build vec_base_emb vec_base_loc graph_index ef(L) max_edges(R) threads max_emb_dis max_loc_dis"
                    << std::endl;
        exit(-1);
    }

    stkq::Parameters parameters;
    parameters.set<std::string>("exc_type", "build");
    parameters.set<std::string>("base_emb", argv[1]);
    parameters.set<std::string>("base_loc", argv[2]);
    parameters.set<std::string>("graph_index", argv[3]);
    std::string exc_type = parameters.get<std::string>("exc_type");
    std::cout << "exc_type: " << parameters.get<std::string>("exc_type")
              << ", base_emb: " << parameters.get<std::string>("base_emb")
              << ", base_loc: " << parameters.get<std::string>("base_loc")
              << ", graph_index: " << parameters.get<std::string>("graph_index");
    if (exc_type == "build")
    {
        unsigned int ef = static_cast<unsigned int>(std::stoul(argv[4]));
        unsigned int max_edges = static_cast<unsigned int>(std::stoul(argv[5]));
        unsigned int threads = static_cast<unsigned int>(std::stoul(argv[6]));
        unsigned int max_emb_dis = static_cast<unsigned int>(std::stoul(argv[7]));
        unsigned int max_loc_dis = static_cast<unsigned int>(std::stoul(argv[8]));
        parameters.set<unsigned>("max_m", max_edges);
        parameters.set<unsigned>("ef_construction", ef);
        parameters.set<unsigned>("n_threads", threads);
        parameters.set<unsigned>("max_emb_dis", max_emb_dis);
        parameters.set<unsigned>("max_loc_dis", max_loc_dis);
        std::cout << ", max_m: " << parameters.get<unsigned>("max_m")
                  << ", ef_construction: " << parameters.get<unsigned>("ef_construction")
                  << ", threads: " << parameters.get<unsigned>("n_threads");
    }
    parameters.set<int>("mult", -1);
    DEG(parameters);

    return 0;
}