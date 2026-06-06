//
// Created by MurphySL on 2020/10/23.
//

#include "component.h"
#include "new_set.h"


namespace stkq
{
    void ComponentPruneHeuristic::PruneInner(unsigned query, unsigned int range, boost::dynamic_bitset<> flags,
                                             std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_)
    {
        std::vector<Index::SimpleNeighbor> picked;
        // 创建一个向量 picked, 用于存储选定的邻居
        // if (pool.size() > range)
        // {
        // 如果候选邻居的数量超过了设定的范围 range 则需要进行剪枝
        // Index::MinHeap<float, Index::SimpleNeighbor> skipped;
        for (int i = 0; i < pool.size(); i++)
        {
            bool skip = false;
            float cur_dist = pool[i].distance;
            for (size_t j = 0; j < picked.size(); j++)
            {

                float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)picked[j].id * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + (size_t)pool[i].id * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

                float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)picked[j].id * index->getBaseLocDim(),
                                                         index->getBaseLocData() + (size_t)pool[i].id * index->getBaseLocDim(),
                                                         index->getBaseLocDim());

                float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                if (dist < cur_dist)
                {
                    skip = true;
                    break;
                }
            }

            if (!skip)
            {
                picked.push_back(pool[i]);
            }
            // else
            // {
            //     skipped.push(cur_dist, pool[i]);
            // }

            if (picked.size() == range)
                break;
        }

        // while (picked.size() < range && skipped.size())
        // {
        //     picked.push_back(skipped.top().data);
        //     skipped.pop();
        // }

        // }
        // else
        // {
        //     for (int i = 0; i < pool.size(); i++)
        //     {
        //         picked.push_back(pool[i]);
        //     }
        // }

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range; // 定位到 cut_graph_ 中对应查询节点的部分
        for (size_t t = 0; t < picked.size(); t++)
        {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
            // std::cout << picked[t].id << "|" << picked[t].distance << " ";
        }
        // std::cout << std::endl;
        // 将选定的邻居复制到 cut_graph_ 数组

        if (picked.size() < range)
        {
            des_pool[picked.size()].distance = -1;
            // 如果 picked 的大小小于 range, 在 des_pool 的相应位置设置距离为 -1，表示没有足够的邻居
        }

        std::vector<Index::SimpleNeighbor>().swap(picked);
    }

    void ComponentDEGPruneHeuristic::PruneInner(std::vector<DEGNNDescentNeighbor> &pool, unsigned int range,
                                                std::vector<Index::DEGNeighbor> &cut_graph_, bool new_set,
                                                std::vector<NEWSimpleNeighborL> &remain_nodes)
    {
        std::vector<Index::DEGNeighbor> picked;
        // pool 按照layer排序 在同层内按照geo_distance排序
        Index::skyline_queue queue;
        sort(pool.begin(), pool.end());
        queue.init_queue(pool);
        pool.swap(queue.pool);
        remain_nodes.reserve(remain_nodes.size() + pool.size());
        int iter = 0;
        int visited_layer = 0;
        while (picked.size() < range && iter < pool.size())
        {
            std::vector<DEGNNDescentNeighbor> candidate;
            while (iter < pool.size())
            {
                if (pool[iter].layer_ == visited_layer)
                {
                    candidate.emplace_back(pool[iter]);
                }
                else
                {
                    break;
                }
                iter++;
            }
            std::vector<Index::DEGNeighbor> tempres_picked;
            for (int i = 0; i < candidate.size(); i++)
            {
                // 这里先初始化useful range 根据斜率算出来
                std::vector<std::pair<float, float>> prune_range;
                float cur_geo_dist = candidate[i].geo_distance_; // s_pq
                float cur_emb_dist = candidate[i].emb_distance_; // e_pq
                for (size_t j = 0; j < picked.size(); j++)
                {
                    const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
                    // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                    float xq_e_dist = index->get_E_Dist()->compare(
                        index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbDim());
                    // E(x,q)

                    float xq_s_dist = index->get_S_Dist()->compare(
                        index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                        index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                        index->getBaseLocDim());
                    // S(x,q)

                    float exist_e_dist = picked[j].emb_distance_; // e_xp

                    float exist_s_dist = picked[j].geo_distance_; // s_xp

                    // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                    // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                    // if alpha holds on for the two equation at the same time, the edge will be pruned
                    // now for equation 1
                    float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                    float diff2 = cur_geo_dist - exist_s_dist;
                    /*
                    diff1 > 0 && diff2 > 0
                    equation 1 holds on when alpha < = diff2 / diff1
                    diff1 < 0 && diff2 < 0
                    equation 1 holds on when alpha > = diff2 / diff1
                    diff1 < 0 && diff2 > 0
                    the equation hold forever
                    diff1 > 0 && diff2 < 0
                    equation never hold which means this edge will not be pruned by this strategy
                    */
                    // float eq1_prune_upper_alpha = 1;
                    // float eq1_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_1;
                    if (diff1 > 0 && diff2 > 0)
                    {
                        // equation 1 holds on when alpha < = diff2 / diff1
                        tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                        // eq1_prune_lower_alpha = diff2 / diff1 ;
                    }
                    else if (diff1 < 0 && diff2 < 0)
                    {
                        // equation 1 holds on when alpha > = diff2 / diff1
                        // eq1_prune_upper_alpha = diff2 / diff1 ;
                        tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                    }
                    else if (diff1 < 0 && diff2 > 0)
                    {
                        tmp_prune_range_1 = {0.0f, 1.0f};
                        // the equation hold forever
                    }
                    else if (diff1 > 0 && diff2 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_1 = {0.0f, 0.0f};
                    }
                    // now for equation 2
                    float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                    float diff4 = cur_geo_dist - xq_s_dist;
                    /*
                    similar to previous
                    */
                    // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                    // float eq2_prune_upper_alpha = 1;
                    // float eq2_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_2;
                    if (diff3 > 0 && diff4 > 0)
                    {
                        // equation 2 holds on when alpha < = diff4 / diff3
                        // eq2_prune_upper_alpha = diff4 / diff3;
                        // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                    }
                    else if (diff3 < 0 && diff4 < 0)
                    {
                        // equation 2 holds on when alpha > = diff4 / diff3
                        // eq2_prune_lower_alpha = diff4 / diff3;
                        // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                    }
                    else if (diff3 < 0 && diff4 > 0)
                    {
                        // the equation hold forever
                        // then we do not change the previous range
                        tmp_prune_range_2 = {0.0f, 1.0f};
                    }
                    else if (diff3 > 0 && diff4 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_2 = {0.0f, 0.0f};
                    }

                    std::pair<float, float> tmp_prune_range;
                    tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                    tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                    if (tmp_prune_range.second > tmp_prune_range.first)
                    {
                        // now we consider whether this range is useful range, that is (second > first)
                        // now we check its intersection range with shared_use_range
                        intersection(picked_use_range, tmp_prune_range, prune_range);
                    }
                    else
                    {
                        continue;
                        // this range is not useful, so this edge will not be pruned by this selected edge
                    }
                }
                prune_range = mergeIntervals(prune_range);
                std::vector<std::pair<float, float>> after_pruned_use_range;
                get_use_range(prune_range, after_pruned_use_range);
                float threshold = 0.1;
                float use_size = 0;
                for (int j = 0; j < after_pruned_use_range.size(); j++)
                {
                    use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                }
                if (use_size >= threshold)
                {
                    picked.push_back(Index::DEGNeighbor(candidate[i].id_, candidate[i].emb_distance_,
                                                        candidate[i].geo_distance_, after_pruned_use_range, visited_layer));
                }
                else
                {
                    remain_nodes.emplace_back(candidate[i].id_, candidate[i].emb_distance_, candidate[i].geo_distance_);
                }
            }
            visited_layer++;
            if (picked.size() >= range)
                break;
        }
        while (iter < pool.size())
        {
            remain_nodes.emplace_back(pool[iter].id_, pool[iter].emb_distance_, pool[iter].geo_distance_);
            iter++;
        }
        cut_graph_.swap(picked);
    }

    void ComponentDEGPruneHeuristic::PruneInner(std::vector<DEGNNDescentNeighbor> &pool, unsigned int range,
                                                std::vector<Index::DEGNeighbor> &cut_graph_, int tid)
    {
        std::vector<Index::DEGNeighbor> picked;
        // pool 按照layer排序 在同层内按照geo_distance排序
        Index::skyline_queue queue;
        sort(pool.begin(), pool.end());
        queue.init_queue(pool);
        pool.swap(queue.pool);
        int iter = 0;
        int visited_layer = 0;
        while (picked.size() < range && iter < pool.size())
        {
            std::vector<DEGNNDescentNeighbor> candidate;
            while (iter < pool.size())
            {
                if (pool[iter].layer_ == visited_layer)
                {
                    candidate.emplace_back(pool[iter]);
                }
                else
                {
                    break;
                }
                iter++;
            }
            std::vector<Index::DEGNeighbor> tempres_picked;
            for (int i = 0; i < candidate.size(); i++)
            {
                // 这里先初始化useful range 根据斜率算出来
                std::vector<std::pair<float, float>> prune_range;
                float cur_geo_dist = candidate[i].geo_distance_; // s_pq
                float cur_emb_dist = candidate[i].emb_distance_; // e_pq
                for (size_t j = 0; j < picked.size(); j++)
                {
                    const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
                    // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                    float xq_e_dist = index->get_E_Dist()->compare(
                        index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbDim());
                    // E(x,q)

                    float xq_s_dist = index->get_S_Dist()->compare(
                        index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                        index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                        index->getBaseLocDim());
                    // S(x,q)
                    if (tid != -1)
                        index->cmp_counts[tid + 1] += 2;
                    float exist_e_dist = picked[j].emb_distance_; // e_xp

                    float exist_s_dist = picked[j].geo_distance_; // s_xp

                    // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                    // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                    // if alpha holds on for the two equation at the same time, the edge will be pruned
                    // now for equation 1
                    float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                    float diff2 = cur_geo_dist - exist_s_dist;
                    /*
                    diff1 > 0 && diff2 > 0
                    equation 1 holds on when alpha < = diff2 / diff1
                    diff1 < 0 && diff2 < 0
                    equation 1 holds on when alpha > = diff2 / diff1
                    diff1 < 0 && diff2 > 0
                    the equation hold forever
                    diff1 > 0 && diff2 < 0
                    equation never hold which means this edge will not be pruned by this strategy
                    */
                    // float eq1_prune_upper_alpha = 1;
                    // float eq1_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_1;
                    if (diff1 > 0 && diff2 > 0)
                    {
                        // equation 1 holds on when alpha < = diff2 / diff1
                        tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                        // eq1_prune_lower_alpha = diff2 / diff1 ;
                    }
                    else if (diff1 < 0 && diff2 < 0)
                    {
                        // equation 1 holds on when alpha > = diff2 / diff1
                        // eq1_prune_upper_alpha = diff2 / diff1 ;
                        tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                    }
                    else if (diff1 < 0 && diff2 > 0)
                    {
                        tmp_prune_range_1 = {0.0f, 1.0f};
                        // the equation hold forever
                    }
                    else if (diff1 > 0 && diff2 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_1 = {0.0f, 0.0f};
                    }
                    // now for equation 2
                    float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                    float diff4 = cur_geo_dist - xq_s_dist;
                    /*
                    similar to previous
                    */
                    // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                    // float eq2_prune_upper_alpha = 1;
                    // float eq2_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_2;
                    if (diff3 > 0 && diff4 > 0)
                    {
                        // equation 2 holds on when alpha < = diff4 / diff3
                        // eq2_prune_upper_alpha = diff4 / diff3;
                        // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                    }
                    else if (diff3 < 0 && diff4 < 0)
                    {
                        // equation 2 holds on when alpha > = diff4 / diff3
                        // eq2_prune_lower_alpha = diff4 / diff3;
                        // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                    }
                    else if (diff3 < 0 && diff4 > 0)
                    {
                        // the equation hold forever
                        // then we do not change the previous range
                        tmp_prune_range_2 = {0.0f, 1.0f};
                    }
                    else if (diff3 > 0 && diff4 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_2 = {0.0f, 0.0f};
                    }

                    std::pair<float, float> tmp_prune_range;
                    tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                    tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                    if (tmp_prune_range.second > tmp_prune_range.first)
                    {
                        // now we consider whether this range is useful range, that is (second > first)
                        // now we check its intersection range with shared_use_range
                        intersection(picked_use_range, tmp_prune_range, prune_range);
                    }
                    else
                    {
                        continue;
                        // this range is not useful, so this edge will not be pruned by this selected edge
                    }
                }
                prune_range = mergeIntervals(prune_range);
                std::vector<std::pair<float, float>> after_pruned_use_range;
                get_use_range(prune_range, after_pruned_use_range);
                float threshold = 0.1;
                float use_size = 0;
                for (int j = 0; j < after_pruned_use_range.size(); j++)
                {
                    use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                }
                if (use_size >= threshold)
                {
                    picked.push_back(Index::DEGNeighbor(candidate[i].id_, candidate[i].emb_distance_,
                                                        candidate[i].geo_distance_, after_pruned_use_range, visited_layer, candidate[i].flag));
                }
            }
            visited_layer++;
            if (picked.size() >= range)
                break;
        }
        cut_graph_.swap(picked);
    }

    void ComponentDEGPruneHeuristic::PruneInnerOPT(std::vector<DEGNNDescentNeighbor> &pool, unsigned int range,
                                                   std::vector<Index::DEGNeighbor> &cut_graph_, int angle, int tid)
    {
        {
            constexpr float COS_10 = 0.9848077530122080f;
            constexpr float COS_15 = 0.9659258262890683f;
            constexpr float COS_20 = 0.9396926207859084f;
            constexpr float COS_30 = 0.8660254037844386f;
            constexpr float COS_40 = 0.7660444431189780f;
            constexpr float COS_50 = 0.6427876096865394f;
            constexpr float COS_60 = 0.5000000000000000f;
            constexpr float COS_70 = 0.3420201433256687f;
            constexpr float COS_80 = 0.1736481776669304f;
            constexpr float COS_90 = 0.0000000000000000f;
            float cos2th = 0.0f;
            std::unordered_map<uint64_t, std::pair<float, float>> dual_dis_map;

            switch (angle)
            {
            case 10:
                cos2th = COS_10*COS_10;
                break;
            case 15:
                cos2th = COS_15*COS_15;
                break;
            case 20:
                cos2th = COS_20*COS_20;
                break;
            case 30:
                cos2th = COS_30*COS_30;
                break;
            case 40:
                cos2th = COS_40*COS_40;
                break;
            case 50:
                cos2th = COS_50*COS_50;
                break;
            case 60:
                cos2th = COS_60*COS_60;
                break;
            case 70:
                cos2th = COS_70*COS_70;
                break;
            case 80:
                cos2th = COS_80*COS_80;
                break;
            case 90:
                cos2th = COS_90*COS_90;
                break;
            default:
                std::cerr << "Unsupported angle: " << angle << std::endl;
                std::exit(EXIT_FAILURE);
            }
            auto get_dual_dist =
                [&](int32_t a, int32_t b)
                -> std::pair<float, float>
            {
                // auto key = std::make_pair(
                //     std::min(a, b),
                //     std::max(a, b));

                uint64_t key =
                    ((uint64_t)std::min(a, b) << 32) |
                    (uint32_t)std::max(a, b);

                auto it = dual_dis_map.find(key);

                if (it != dual_dis_map.end())
                    return it->second;

                float e_d = index->get_E_Dist()->compare(
                    index->getBaseEmbData() + (size_t)a * index->getBaseEmbDim(),
                    index->getBaseEmbData() + (size_t)b * index->getBaseEmbDim(),
                    index->getBaseEmbDim());

                float s_d = index->get_S_Dist()->compare(
                    index->getBaseLocData() + (size_t)a * index->getBaseLocDim(),
                    index->getBaseLocData() + (size_t)b * index->getBaseLocDim(),
                    index->getBaseLocDim());

                if (tid != -1)
                    index->cmp_counts[tid + 1] += 2;

                auto result = std::make_pair(e_d, s_d);
                dual_dis_map.emplace(key, result);

                return result;
            };
            auto judge_dominate =
                [](const std::pair<float, float> &a,
                const std::pair<float, float> &b,
                const float cos2th) -> bool
            {
                float ax = a.first;
                float ay = a.second;
                float bx = b.first;
                float by = b.second;

                float dot = ax * bx + ay * by;

                float na2 = ax * ax + ay * ay;
                float nb2 = bx * bx + by * by;

                if (na2 == 0.0f || nb2 == 0.0f)
                    return false;

                // 防止夹角 > 90°
                if (dot <= 0.0f)
                    return false;

                return dot * dot >= cos2th * na2 * nb2;
            };

            std::vector<Index::DEGNeighbor> picked;
            // pool 按照layer排序 在同层内按照geo_distance排序
            Index::skyline_queue queue;
            sort(pool.begin(), pool.end());
            queue.init_queue(pool);
            pool.swap(queue.pool);
            int iter = 0;
            int visited_layer = 0;
            while (picked.size() < range && iter < pool.size())
            {
                std::vector<DEGNNDescentNeighbor> candidate;
                while (iter < pool.size())
                {
                    if (pool[iter].layer_ == visited_layer)
                    {
                        candidate.emplace_back(pool[iter]);
                    }
                    else
                    {
                        break;
                    }
                    iter++;
                }
                std::vector<Index::DEGNeighbor> tempres_picked;

                if (visited_layer == 0)
                {
                    // candidate = spread_from_center(candidate);
                    for (int i = 0; i < candidate.size(); i++)
                    {
                        // 这里先初始化useful range 根据斜率算出来
                        std::vector<std::pair<float, float>> prune_range;
                        float cur_geo_dist = candidate[i].geo_distance_; // s_pq
                        float cur_emb_dist = candidate[i].emb_distance_; // e_pq
                        for (size_t j = 0; j < tempres_picked.size(); j++)
                        {
                            const std::vector<std::pair<float, float>> &picked_use_range = tempres_picked[j].available_range;
                            // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                            // float xq_e_dist = 0, xq_s_dist = 0;
                            auto [xq_e_dist, xq_s_dist] = get_dual_dist(tempres_picked[j].id_, candidate[i].id_);
                            // float xq_e_dist = index->get_E_Dist()->compare(
                            //     index->getBaseEmbData() + (size_t)tempres_picked[j].id_ * index->getBaseEmbDim(),
                            //     index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                            //     index->getBaseEmbDim());
                            // E(x,q)

                            // float xq_s_dist = index->get_S_Dist()->compare(
                            //     index->getBaseLocData() + (size_t)tempres_picked[j].id_ * index->getBaseLocDim(),
                            //     index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                            //     index->getBaseLocDim());
                            // S(x,q)
                            // if (tid != -1)
                            //     index->cmp_counts[tid + 1] += 2;
                            float exist_e_dist = tempres_picked[j].emb_distance_; // e_xp

                            float exist_s_dist = tempres_picked[j].geo_distance_; // s_xp

                            // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                            // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                            // if alpha holds on for the two equation at the same time, the edge will be pruned
                            // now for equation 1
                            float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                            float diff2 = cur_geo_dist - exist_s_dist;
                            /*
                            diff1 > 0 && diff2 > 0
                            equation 1 holds on when alpha < = diff2 / diff1
                            diff1 < 0 && diff2 < 0
                            equation 1 holds on when alpha > = diff2 / diff1
                            diff1 < 0 && diff2 > 0
                            the equation hold forever
                            diff1 > 0 && diff2 < 0
                            equation never hold which means this edge will not be pruned by this strategy
                            */
                            // float eq1_prune_upper_alpha = 1;
                            // float eq1_prune_lower_alpha = 0;
                            std::pair<float, float> tmp_prune_range_1;
                            if (diff1 > 0 && diff2 > 0)
                            {
                                // equation 1 holds on when alpha < = diff2 / diff1
                                tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                                // eq1_prune_lower_alpha = diff2 / diff1 ;
                            }
                            else if (diff1 < 0 && diff2 < 0)
                            {
                                // equation 1 holds on when alpha > = diff2 / diff1
                                // eq1_prune_upper_alpha = diff2 / diff1 ;
                                tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                            }
                            else if (diff1 < 0 && diff2 > 0)
                            {
                                tmp_prune_range_1 = {0.0f, 1.0f};
                                // the equation hold forever
                            }
                            else if (diff1 > 0 && diff2 < 0)
                            {
                                // equation never hold
                                // break;
                                tmp_prune_range_1 = {0.0f, 0.0f};
                            }
                            // now for equation 2
                            float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                            float diff4 = cur_geo_dist - xq_s_dist;
                            /*
                            similar to previous
                            */
                            // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                            // float eq2_prune_upper_alpha = 1;
                            // float eq2_prune_lower_alpha = 0;
                            std::pair<float, float> tmp_prune_range_2;
                            if (diff3 > 0 && diff4 > 0)
                            {
                                // equation 2 holds on when alpha < = diff4 / diff3
                                // eq2_prune_upper_alpha = diff4 / diff3;
                                // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                                tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                            }
                            else if (diff3 < 0 && diff4 < 0)
                            {
                                // equation 2 holds on when alpha > = diff4 / diff3
                                // eq2_prune_lower_alpha = diff4 / diff3;
                                // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                                tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                            }
                            else if (diff3 < 0 && diff4 > 0)
                            {
                                // the equation hold forever
                                // then we do not change the previous range
                                tmp_prune_range_2 = {0.0f, 1.0f};
                            }
                            else if (diff3 > 0 && diff4 < 0)
                            {
                                // equation never hold
                                // break;
                                tmp_prune_range_2 = {0.0f, 0.0f};
                            }

                            std::pair<float, float> tmp_prune_range;
                            tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                            tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                            if (tmp_prune_range.second > tmp_prune_range.first)
                            {
                                // now we consider whether this range is useful range, that is (second > first)
                                // now we check its intersection range with shared_use_range
                                intersection(picked_use_range, tmp_prune_range, prune_range);
                            }
                            else
                            {
                                continue;
                                // this range is not useful, so this edge will not be pruned by this selected edge
                            }
                        }
                        prune_range = mergeIntervals(prune_range);
                        std::vector<std::pair<float, float>> after_pruned_use_range;
                        get_use_range(prune_range, after_pruned_use_range);
                        float threshold = 0.1;
                        float use_size = 0;
                        for (int j = 0; j < after_pruned_use_range.size(); j++)
                        {
                            use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                        }
                        if (use_size >= threshold)
                        {
                            tempres_picked.push_back(Index::DEGNeighbor(candidate[i].id_, candidate[i].emb_distance_,
                                                                        candidate[i].geo_distance_, after_pruned_use_range, visited_layer));
                        }
                    }
                    Index::DEGNeighbor lastnode = tempres_picked[tempres_picked.size() - 1];
                    tempres_picked.clear();
                    tempres_picked.push_back(lastnode);
                    int last_i = 0;
                    for (int i = candidate.size() - 1; i >= 0; i--)
                    {
                        if (lastnode.id_ == candidate[i].id_)
                        {
                            last_i = i;
                            break;
                        }
                    }
                    for (int i = last_i - 1; i >= 0; i--)
                    {
                        // 这里先初始化useful range 根据斜率算出来
                        std::vector<std::pair<float, float>> prune_range;
                        float cur_geo_dist = candidate[i].geo_distance_; // s_pq
                        float cur_emb_dist = candidate[i].emb_distance_; // e_pq
                        for (size_t j = 0; j < tempres_picked.size(); j++)
                        {
                            const std::vector<std::pair<float, float>> &picked_use_range = tempres_picked[j].available_range;
                            // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                            // float xq_e_dist = index->get_E_Dist()->compare(
                            //     index->getBaseEmbData() + (size_t)tempres_picked[j].id_ * index->getBaseEmbDim(),
                            //     index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                            //     index->getBaseEmbDim());
                            // // E(x,q)

                            // float xq_s_dist = index->get_S_Dist()->compare(
                            //     index->getBaseLocData() + (size_t)tempres_picked[j].id_ * index->getBaseLocDim(),
                            //     index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                            //     index->getBaseLocDim());
                            // // S(x,q)
                            // if (tid != -1)
                            //     index->cmp_counts[tid + 1] += 2;
                            // float xq_e_dist = 0, xq_s_dist = 0;
                            auto [xq_e_dist, xq_s_dist] = get_dual_dist(tempres_picked[j].id_, candidate[i].id_);

                            float exist_e_dist = tempres_picked[j].emb_distance_; // e_xp

                            float exist_s_dist = tempres_picked[j].geo_distance_; // s_xp

                            // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                            // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                            // if alpha holds on for the two equation at the same time, the edge will be pruned
                            // now for equation 1
                            float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                            float diff2 = cur_geo_dist - exist_s_dist;
                            /*
                            diff1 > 0 && diff2 > 0
                            equation 1 holds on when alpha < = diff2 / diff1
                            diff1 < 0 && diff2 < 0
                            equation 1 holds on when alpha > = diff2 / diff1
                            diff1 < 0 && diff2 > 0
                            the equation hold forever
                            diff1 > 0 && diff2 < 0
                            equation never hold which means this edge will not be pruned by this strategy
                            */
                            // float eq1_prune_upper_alpha = 1;
                            // float eq1_prune_lower_alpha = 0;
                            std::pair<float, float> tmp_prune_range_1;
                            if (diff1 > 0 && diff2 > 0)
                            {
                                // equation 1 holds on when alpha < = diff2 / diff1
                                tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                                // eq1_prune_lower_alpha = diff2 / diff1 ;
                            }
                            else if (diff1 < 0 && diff2 < 0)
                            {
                                // equation 1 holds on when alpha > = diff2 / diff1
                                // eq1_prune_upper_alpha = diff2 / diff1 ;
                                tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                            }
                            else if (diff1 < 0 && diff2 > 0)
                            {
                                tmp_prune_range_1 = {0.0f, 1.0f};
                                // the equation hold forever
                            }
                            else if (diff1 > 0 && diff2 < 0)
                            {
                                // equation never hold
                                // break;
                                tmp_prune_range_1 = {0.0f, 0.0f};
                            }
                            // now for equation 2
                            float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                            float diff4 = cur_geo_dist - xq_s_dist;
                            /*
                            similar to previous
                            */
                            // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                            // float eq2_prune_upper_alpha = 1;
                            // float eq2_prune_lower_alpha = 0;
                            std::pair<float, float> tmp_prune_range_2;
                            if (diff3 > 0 && diff4 > 0)
                            {
                                // equation 2 holds on when alpha < = diff4 / diff3
                                // eq2_prune_upper_alpha = diff4 / diff3;
                                // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                                tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                            }
                            else if (diff3 < 0 && diff4 < 0)
                            {
                                // equation 2 holds on when alpha > = diff4 / diff3
                                // eq2_prune_lower_alpha = diff4 / diff3;
                                // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                                tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                            }
                            else if (diff3 < 0 && diff4 > 0)
                            {
                                // the equation hold forever
                                // then we do not change the previous range
                                tmp_prune_range_2 = {0.0f, 1.0f};
                            }
                            else if (diff3 > 0 && diff4 < 0)
                            {
                                // equation never hold
                                // break;
                                tmp_prune_range_2 = {0.0f, 0.0f};
                            }

                            std::pair<float, float> tmp_prune_range;
                            tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                            tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                            if (tmp_prune_range.second > tmp_prune_range.first)
                            {
                                // now we consider whether this range is useful range, that is (second > first)
                                // now we check its intersection range with shared_use_range
                                intersection(picked_use_range, tmp_prune_range, prune_range);
                            }
                            else
                            {
                                continue;
                                // this range is not useful, so this edge will not be pruned by this selected edge
                            }
                        }
                        prune_range = mergeIntervals(prune_range);
                        std::vector<std::pair<float, float>> after_pruned_use_range;
                        get_use_range(prune_range, after_pruned_use_range);
                        float threshold = 0.1;
                        float use_size = 0;
                        for (int j = 0; j < after_pruned_use_range.size(); j++)
                        {
                            use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                        }
                        if (use_size >= threshold)
                        {
                            tempres_picked.push_back(Index::DEGNeighbor(candidate[i].id_, candidate[i].emb_distance_,
                                                                        candidate[i].geo_distance_, after_pruned_use_range, visited_layer));
                        }
                    }
                    picked.insert(picked.end(), tempres_picked.begin(), tempres_picked.end());
                }
                else
                {
                    for (int i = 0; i < candidate.size(); i++)
                    {
                        // 这里先初始化useful range 根据斜率算出来
                        std::vector<std::pair<float, float>> prune_range;
                        float cur_geo_dist = candidate[i].geo_distance_; // s_pq
                        float cur_emb_dist = candidate[i].emb_distance_; // e_pq

                        for (size_t j = 0; j < picked.size(); j++)
                        {
                            float exist_e_dist = picked[j].emb_distance_; // e_xp

                            float exist_s_dist = picked[j].geo_distance_; // s_xp
                            if (!judge_dominate(std::make_pair(exist_s_dist, exist_e_dist), std::make_pair(cur_geo_dist, cur_emb_dist), cos2th))
                            {
                                continue;
                            }
                            const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
                            // we want to find out if this edge can prune the candidate within its picked_avaiable_range
                            // float xq_e_dist = 0, xq_s_dist = 0;
                            auto [xq_e_dist, xq_s_dist] = get_dual_dist(picked[j].id_, candidate[i].id_);

                            // float xq_e_dist = index->get_E_Dist()->compare(
                            //     index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                            //     index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                            //     index->getBaseEmbDim());
                            // // E(x,q)

                            // float xq_s_dist = index->get_S_Dist()->compare(
                            //     index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                            //     index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                            //     index->getBaseLocDim());
                            // S(x,q)
                            // if (tid != -1)
                            //     index->cmp_counts[tid + 1] += 2;
                            // alpha * (E(p,x) - S(p,x) - E(p,q) + S(p,q)) <= S(p,q) - S(p,x)
                            // alpha * (E(q,x) - S(q,x) - E(p,q) + S(p,q)) <= S(p,q) - S(q,x)
                            // if alpha holds on for the two equation at the same time, the edge will be pruned
                            // now for equation 1
                            float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                            float diff2 = cur_geo_dist - exist_s_dist;
                            /*
                            diff1 > 0 && diff2 > 0
                            equation 1 holds on when alpha < = diff2 / diff1
                            diff1 < 0 && diff2 < 0
                            equation 1 holds on when alpha > = diff2 / diff1
                            diff1 < 0 && diff2 > 0
                            the equation hold forever
                            diff1 > 0 && diff2 < 0
                            equation never hold which means this edge will not be pruned by this strategy
                            */
                            // float eq1_prune_upper_alpha = 1;
                            // float eq1_prune_lower_alpha = 0;
                            std::pair<float, float> tmp_prune_range_1;
                            if (diff1 > 0 && diff2 > 0)
                            {
                                // equation 1 holds on when alpha < = diff2 / diff1
                                tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                                // eq1_prune_lower_alpha = diff2 / diff1 ;
                            }
                            else if (diff1 < 0 && diff2 < 0)
                            {
                                // equation 1 holds on when alpha > = diff2 / diff1
                                // eq1_prune_upper_alpha = diff2 / diff1 ;
                                tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                            }
                            else if (diff1 < 0 && diff2 > 0)
                            {
                                tmp_prune_range_1 = {0.0f, 1.0f};
                                // the equation hold forever
                            }
                            else if (diff1 > 0 && diff2 < 0)
                            {
                                // equation never hold
                                // break;
                                tmp_prune_range_1 = {0.0f, 0.0f};
                            }
                            // now for equation 2
                            float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                            float diff4 = cur_geo_dist - xq_s_dist;
                            /*
                            similar to previous
                            */
                            // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                            // float eq2_prune_upper_alpha = 1;
                            // float eq2_prune_lower_alpha = 0;
                            std::pair<float, float> tmp_prune_range_2;
                            if (diff3 > 0 && diff4 > 0)
                            {
                                // equation 2 holds on when alpha < = diff4 / diff3
                                // eq2_prune_upper_alpha = diff4 / diff3;
                                // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                                tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                            }
                            else if (diff3 < 0 && diff4 < 0)
                            {
                                // equation 2 holds on when alpha > = diff4 / diff3
                                // eq2_prune_lower_alpha = diff4 / diff3;
                                // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                                tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                            }
                            else if (diff3 < 0 && diff4 > 0)
                            {
                                // the equation hold forever
                                // then we do not change the previous range
                                tmp_prune_range_2 = {0.0f, 1.0f};
                            }
                            else if (diff3 > 0 && diff4 < 0)
                            {
                                // equation never hold
                                // break;
                                tmp_prune_range_2 = {0.0f, 0.0f};
                            }

                            std::pair<float, float> tmp_prune_range;
                            tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                            tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                            if (tmp_prune_range.second > tmp_prune_range.first)
                            {
                                // now we consider whether this range is useful range, that is (second > first)
                                // now we check its intersection range with shared_use_range
                                intersection(picked_use_range, tmp_prune_range, prune_range);
                            }
                            else
                            {
                                continue;
                                // this range is not useful, so this edge will not be pruned by this selected edge
                            }
                        }

                        prune_range = mergeIntervals(prune_range);
                        std::vector<std::pair<float, float>> after_pruned_use_range;
                        get_use_range(prune_range, after_pruned_use_range);
                        float threshold = 0.1;
                        float use_size = 0;
                        for (int j = 0; j < after_pruned_use_range.size(); j++)
                        {
                            use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                        }
                        if (use_size >= threshold)
                        {
                            picked.push_back(Index::DEGNeighbor(candidate[i].id_, candidate[i].emb_distance_,
                                                                candidate[i].geo_distance_, after_pruned_use_range, visited_layer));
                        }
                    }
                }
                visited_layer++;
                if (picked.size() >= range)
                    break;
            }
            if (picked.size() > range)
                picked.resize(range);
            cut_graph_.swap(picked);
        }
        
    }

}