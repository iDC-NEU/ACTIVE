#ifndef NEIHBOR_H
#define NEIHBOR_H

namespace stkq
{
    struct DEGNNDescentNeighbor
    {
        unsigned id_;
        float emb_distance_;
        float geo_distance_;
        bool flag;
        int layer_;
        bool delete_ = false;
        // std::vector<std::pair<float, float>> available_range_;

        DEGNNDescentNeighbor() = default;
        DEGNNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(f), layer_(layer)
        {
        }
        // DEGNNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer, std::vector<std::pair<float, float>> &available_range) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(f), layer_(layer), available_range_(available_range)
        // {
        // }
        DEGNNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer, bool is_expanded) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(f), layer_(layer)
        {
        }
        inline bool operator<(const DEGNNDescentNeighbor &other) const
        {
            // return geo_distance_ < other.geo_distance_;
            return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
            // 较小的 geo_distance_ 值会被排序到较前的位置
        }
    };
}
#endif