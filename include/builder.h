
#ifndef STKQ_BUILDER_H
#define STKQ_BUILDER_H

#include "index.h"
#include "parameters.h"
#include <future>
namespace stkq
{
    class IndexBuilder
    {
    public:
        explicit IndexBuilder(const unsigned num_threads, const float max_emb_dist, const float max_spatial_dist, bool dual_index = false)
        {
            if (dual_index == false)
            {
                final_index_ = new Index(max_emb_dist, max_spatial_dist);
                omp_set_num_threads(num_threads);
            }
            else
            {
                final_index_ = new Index(max_emb_dist, max_spatial_dist);
                final_index_1 = new Index(max_emb_dist, max_spatial_dist);
                final_index_2 = new Index(max_emb_dist, max_spatial_dist);
                omp_set_num_threads(num_threads);
            }
        }

        virtual ~IndexBuilder()
        {
            if (final_index_ != nullptr)
                delete final_index_;
            if (final_index_1 != nullptr)
                delete final_index_1;
            if (final_index_2 != nullptr)
                delete final_index_2;
        }

        IndexBuilder *load(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file, Parameters &parameters, bool dual = false, bool update = false);

        IndexBuilder *init(TYPE type, bool debug = false);

        IndexBuilder *update(TYPE type, unsigned id_flag, Parameters &parameters);

        IndexBuilder *update2search();

        IndexBuilder *save_graph(TYPE type, char *graph_file);

        inline std::future<void> update_graph_async(const Parameters &parame)
        {
            return std::async(std::launch::async,
                              [this, parame]()
                              {
                                  this->update_graph(parame);
                              });
        }
        void update_graph(const Parameters &parame);
        // void update_graph(stkq::Parameters parame);
        int average_neighbor_size = 0;
        int active_nodes = 0;
        IndexBuilder *load_graph(TYPE type, char *graph_file_1, char *graph_file_2);
        IndexBuilder *load_graph(TYPE type, char *graph_file, Parameters &parame);

        IndexBuilder *refine(TYPE type, bool debug);

        IndexBuilder *search(TYPE entry_type, TYPE route_type, TYPE L_type, Parameters para_);

        IndexBuilder *insert(TYPE entry_type, TYPE route_type, TYPE L_type, Parameters para_, bool delete_ = false);

        void print_graph();

        void load_trace_(char *trace_path, unsigned id_flag, Parameters &parameters);
        void load_gt_(char *gt_path, Parameters &parameters);

        void degree_info(std::unordered_map<unsigned, unsigned> &in_degree, std::unordered_map<unsigned, unsigned> &out_degree, TYPE type);

        void conn_info(TYPE type);

        void graph_quality(TYPE type);

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt, TYPE type);

        void findRoot(boost::dynamic_bitset<> &flag, std::vector<unsigned> &root);

        void set_begin_time()
        {
            s = std::chrono::high_resolution_clock::now();
        }

        void set_end_time()
        {
            e = std::chrono::high_resolution_clock::now();
        }

        IndexBuilder *draw();

        std::chrono::duration<double> GetBuildTime() { return e - s; }

        void peak_memory_footprint();
        Index *final_index_ = nullptr;

    private:
        Index *final_index_1 = nullptr;
        Index *final_index_2 = nullptr;

        std::chrono::high_resolution_clock::time_point s;
        std::chrono::high_resolution_clock::time_point e;
    };
}

#endif