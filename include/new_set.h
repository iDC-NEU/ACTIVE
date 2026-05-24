#ifndef NEW_SET_H
#define NEW_SET_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <vector>
#include "neighbor.h"
#include "skyline_tree.h"

namespace stkq {
struct NEWNeighbor {
    unsigned id_;           // 较少访问
    float geo_distance_;     // 用于比较的主要字段
    float emb_distance_;     // 用于比较的次要字段
    int layer_;             // 频繁访问
    bool flag_;            // 较少访问
    std::vector<float> min_y_list; // 大对象放最后
    explicit NEWNeighbor() = default;
    explicit NEWNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer)
        : id_{id}, geo_distance_(geo_distance), emb_distance_{emb_distance}, layer_(layer), flag_(f)
    { }
    explicit NEWNeighbor(unsigned id, float emb_distance, float geo_distance)
        : id_{id}, geo_distance_(geo_distance), emb_distance_{emb_distance}
    { }
    NEWNeighbor(const NEWNeighbor &other)
        : id_{other.id_}, geo_distance_(other.geo_distance_), emb_distance_{other.emb_distance_}, layer_(other.layer_), flag_(other.flag_), min_y_list(other.min_y_list)
    { }
    NEWNeighbor(NEWNeighbor &&other)
        : id_{other.id_}, geo_distance_(other.geo_distance_), emb_distance_{other.emb_distance_}, layer_(other.layer_), flag_(other.flag_), min_y_list(std::move(other.min_y_list))
    { }
    NEWNeighbor& operator=(NEWNeighbor& other) noexcept {
        if (this != &other) {
            id_ = other.id_;
            geo_distance_ = other.geo_distance_;
            emb_distance_ = other.emb_distance_;
            layer_ = other.layer_;
            flag_ = other.flag_;
            min_y_list = other.min_y_list;
        }
        return *this;
    }
    // 移动赋值
    NEWNeighbor& operator=(NEWNeighbor&& other) noexcept {
        if (this != &other) {
            id_ = other.id_;
            geo_distance_ = other.geo_distance_;
            emb_distance_ = other.emb_distance_;
            layer_ = other.layer_;
            flag_ = other.flag_;
            min_y_list = std::move(other.min_y_list);
        }
        return *this;
    }
    inline bool operator<(const NEWNeighbor &other) const {
        return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
    }
};

struct NEWSimpleNeighbor {
    unsigned id_;           // 较少访问
    float geo_distance_;     // 用于比较的主要字段
    float emb_distance_;     // 用于比较的次要字段
    int layer_;
    explicit NEWSimpleNeighbor() = default;
    NEWSimpleNeighbor(unsigned id, float emb_distance, float geo_distance, int layer)
        : id_{id}, geo_distance_(geo_distance), emb_distance_{emb_distance}, layer_(layer)
    { }
    inline bool operator<(const NEWSimpleNeighbor &other) const {
        return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
    }
};

struct NEWSimpleNeighborL {
    unsigned id_;           // 较少访问
    float geo_distance_;     // 用于比较的主要字段
    float emb_distance_;     // 用于比较的次要字段
    explicit NEWSimpleNeighborL() = default;
    NEWSimpleNeighborL(unsigned id, float emb_distance, float geo_distance)
        : id_{id}, geo_distance_(geo_distance), emb_distance_{emb_distance}
    { }
};

class NEWSkyLine {
private:
    std::vector<NEWNeighbor> pool_;
    unsigned M_; // 记录pool的最大大小, 也就是candidate边数
    std::vector<std::vector<float>> update_nodes_;
    
public:
    NEWSkyLine() = default;

    NEWSkyLine(unsigned l) : M_(l) {
        pool_.reserve(M_);
    }

    NEWSkyLine(NEWSkyLine &&other) : pool_(std::move(other.pool_)), M_(other.M_) {}
    
    void operator=(NEWSkyLine &&other) {
        pool_ = std::move(other.pool_);
        M_ = other.M_;
    }

    void construct(std::vector<NEWSimpleNeighbor>& pool){
        if (pool.empty()) {
            pool_.clear();
            return;
        }
        std::sort(pool.begin(), pool.end());
        pool_.clear();
        pool_.reserve(pool.size());
        M_ = 200;
        // 预分配第一个元素，避免特殊处理
        const auto& first = pool[0];
        if (first.id_ > 900000) {
            SKY_ASSERT(false, "error id");
        }
        pool_.emplace_back(first.id_, first.emb_distance_, first.geo_distance_, false, first.layer_);
        pool_[0].min_y_list.reserve(first.layer_ + 1);
        pool_[0].min_y_list.push_back(first.emb_distance_);

        // 从第二个元素开始批量处理
        for (size_t i = 1; i < pool.size(); ++ i) {
            const auto& cur = pool[i];
            const auto& prev = pool_[i - 1];
            
            // 添加新元素
            if (cur.id_ > 900000) {
                SKY_ASSERT(false, "error id");
            }
            pool_.emplace_back(cur.id_, cur.emb_distance_, cur.geo_distance_, false, cur.layer_);
            auto& current_neighbor = pool_.back();
            auto& min_y_list = current_neighbor.min_y_list;
            
            size_t required_size = prev.min_y_list.size() + 1;
            min_y_list.reserve(required_size);
            min_y_list.assign(prev.min_y_list.begin(), prev.min_y_list.end());
            
            // 层数不对，被修剪了
            bool update = false;
            for (size_t i = 0; i < min_y_list.size(); i ++) {
                if (current_neighbor.emb_distance_ < min_y_list.at(i) && !update) {
                    min_y_list[i] = current_neighbor.emb_distance_;
                    current_neighbor.layer_ = i;
                    update = true;
                    break;
                }
            }
            if (!update) {
                min_y_list.emplace_back(current_neighbor.emb_distance_);
                current_neighbor.layer_ = min_y_list.size()-1;
            }
            
            validate_min_y_list(current_neighbor, "construct");
        }
    }

    void construct(std::vector<DEGNNDescentNeighbor>& pool){
        if (pool.empty()) {
            pool_.clear();
            return;
        }
        std::sort(pool.begin(), pool.end());
        pool_.clear();
        pool_.reserve(pool.size());
        M_ = 200;
        // 预分配第一个元素，避免特殊处理
        const auto& first = pool[0];
        pool_.emplace_back(first.id_, first.emb_distance_, first.geo_distance_, false, first.layer_);
        pool_[0].min_y_list.reserve(first.layer_ + 1);
        pool_[0].min_y_list.push_back(first.emb_distance_);

        // 从第二个元素开始批量处理
        for (size_t i = 1; i < pool.size(); ++ i) {
            const auto& cur = pool[i];
            const auto& prev = pool_[i - 1];
            
            // 添加新元素
            pool_.emplace_back(cur.id_, cur.emb_distance_, cur.geo_distance_, false, cur.layer_);
            auto& current_neighbor = pool_.back();
            auto& min_y_list = current_neighbor.min_y_list;
            
            size_t required_size = prev.min_y_list.size() + 1;
            min_y_list.reserve(required_size);
            min_y_list.assign(prev.min_y_list.begin(), prev.min_y_list.end());
            
            // 层数不对，被修剪了
            bool update = false;
            for (size_t i = 0; i < min_y_list.size(); i ++) {
                if (current_neighbor.emb_distance_ < min_y_list.at(i) && !update) {
                    min_y_list[i] = current_neighbor.emb_distance_;
                    current_neighbor.layer_ = i;
                    update = true;
                    break;
                }
            }
            if (!update) {
                min_y_list.emplace_back(current_neighbor.emb_distance_);
                current_neighbor.layer_ = min_y_list.size()-1;
            }
            
            validate_min_y_list(current_neighbor, "construct");
        }
    }

    void insert(unsigned id, float e_d, float s_d) {
        NEWNeighbor new_neighbor{id, e_d, s_d, true, -1};
        auto insert_pos = std::upper_bound(pool_.begin(), pool_.end(), new_neighbor);
        size_t insert_idx = insert_pos - pool_.begin();
        int insert_node_layer = 0;
        update_nodes_.clear();
        update_nodes_.reserve(10);

        if (id > 900000) {
            SKY_ASSERT(false, "error id");
        }

        if (insert_idx > 0) {
            const auto& prev = pool_[insert_idx - 1];

            size_t required_size = prev.min_y_list.size() + 1;
            new_neighbor.min_y_list.reserve(required_size);
            new_neighbor.min_y_list.assign(prev.min_y_list.begin(), prev.min_y_list.end());
            bool update = false;

            for (size_t i = 0; i < new_neighbor.min_y_list.size(); i ++) {
                if (e_d < new_neighbor.min_y_list[i] && !update) {
                    insert_node_layer = i;
                    new_neighbor.min_y_list[i] = e_d;
                    update = true;
                    break;
                }
            }
            if (!update) {
                new_neighbor.min_y_list.emplace_back(e_d);
                insert_node_layer = new_neighbor.min_y_list.size()-1;
            }
        } else {
            new_neighbor.min_y_list.push_back(e_d);
        }

        new_neighbor.layer_ = insert_node_layer;

        update_nodes_.emplace_back();
        update_nodes_[0].reserve(4);
        update_nodes_.at(0).push_back(e_d);

        validate_min_y_list(new_neighbor, "insert");
        // if (new_neighbor.min_y_list.size() < new_neighbor.layer_
        //     || new_neighbor.emb_distance_ != new_neighbor.min_y_list[new_neighbor.layer_]) {
        //     SKY_ASSERT(false, "min_y_list size inconsistent");
        // }
        pool_.insert(pool_.begin() + insert_idx, std::move(new_neighbor));

        for (size_t i = insert_idx + 1; i < pool_.size(); ++i) {
            auto& current_neighbor = pool_[i];
            auto& prev_neighbor = pool_[i-1];

            if (current_neighbor.min_y_list.size() < prev_neighbor.min_y_list.size()) {
                // std::cout<<"new layer" << current_neighbor.min_y_list.size() << " " << prev_neighbor.min_y_list.size() << std::endl;
                current_neighbor.min_y_list.emplace_back(prev_neighbor.min_y_list.back());
                if (current_neighbor.min_y_list.size() < prev_neighbor.min_y_list.size()) {
                    SKY_ASSERT(false, "error min y list");
                }
            }

            if (current_neighbor.layer_ < insert_node_layer) continue;
            // 受影响
            size_t layer_diff = current_neighbor.layer_ - insert_node_layer;
            if (layer_diff < update_nodes_.size()) {
                const auto& update_list = update_nodes_[layer_diff];
                // 优化10: 使用范围for循环，编译器更容易优化
                for (float e_distance : update_list) {
                    if (e_distance < current_neighbor.emb_distance_) {
                        auto& prev_neighbor = pool_[i-1];
                        if (prev_neighbor.min_y_list.size() < current_neighbor.min_y_list.size()) {
                            current_neighbor.min_y_list[current_neighbor.layer_] = 
                                std::numeric_limits<float>::max();
                        } else {
                            current_neighbor.min_y_list[current_neighbor.layer_] = 
                                prev_neighbor.min_y_list[current_neighbor.layer_];
                        }
                        current_neighbor.layer_++;
                        size_t new_layer_diff = current_neighbor.layer_ - insert_node_layer;
                        
                        // 确保update_nodes_有足够空间
                        while (update_nodes_.size() <= new_layer_diff) {
                            update_nodes_.emplace_back();
                            update_nodes_.back().reserve(4);
                        }
                        
                        update_nodes_[new_layer_diff].push_back(current_neighbor.emb_distance_);
                        break;
                    }
                }
            }
        }

        
        if (pool_.size() > M_) {
            int max_layer = -1;
            size_t remove_idx = 0;
            for (size_t i = pool_.size(); i > 0; --i) {
                size_t idx = i - 1;
                if (pool_[idx].layer_ > max_layer) {
                    max_layer = pool_[idx].layer_;
                    remove_idx = idx;
                }
            }
            pool_.erase(pool_.begin()+remove_idx);
        }

        // for (auto &p : pool_) {
        //     validate_min_y_list(p, "insert all");
        // }

        return ;
    }

    std::vector<DEGNNDescentNeighbor> traverse_all_sort() {
        std::vector<DEGNNDescentNeighbor> res;

        res.reserve(pool_.size());
        for (const auto& neighbor : pool_) {
            res.emplace_back(neighbor.id_, neighbor.emb_distance_, neighbor.geo_distance_, false, neighbor.layer_);
        }
        // std::sort(res.begin(), res.end(), 
        //       [](const auto& a, const auto& b) {
        //           return a.layer_ < b.layer_ || 
        //                  (a.layer_ == b.layer_ && a < b);
        //       });

        std::sort(res.begin(), res.end(), 
        [](const DEGNNDescentNeighbor& a, const DEGNNDescentNeighbor& b) noexcept {
            // 先比较layer_（更可能不同）
            if (a.layer_ != b.layer_) {
                return a.layer_ < b.layer_;
            }
            // layer_相同时才比较距离
            return a.geo_distance_ < b.geo_distance_ || 
                   (a.geo_distance_ == b.geo_distance_ && a.emb_distance_ < b.emb_distance_);
        });

        return res;
    }
    std::vector<DEGNNDescentNeighbor> traverse_all_layer() {
        std::vector<DEGNNDescentNeighbor> res;

        res.reserve(pool_.size());
        // 优化2: 动态计算最大layer，避免固定大小数组
        int max_layer = 0;
        for (const auto& neighbor : pool_) {
            max_layer = std::max(max_layer, neighbor.layer_);
        }
        
        if (max_layer < 0) {
            return res;
        }
        
        // 优化3: 只分配需要的layer数量
        std::vector<std::vector<NEWNeighbor>> layer_neighbor(max_layer + 1);
        
        // 优化4: 预估每层容量，减少vector扩容
        std::vector<size_t> layer_counts(max_layer + 1, 0);
        for (const auto& neighbor : pool_) {
            if (neighbor.layer_ >= 0) {
                layer_counts[neighbor.layer_]++;
            }
        }
        
        for (int i = 0; i <= max_layer; ++i) {
            layer_neighbor[i].reserve(layer_counts[i]);
        }
        
        // 分组到各层
        for (const auto& neighbor : pool_) {
            if (neighbor.layer_ >= 0) {
                layer_neighbor[neighbor.layer_].emplace_back(neighbor);
            }
        }
        // std::vector<std::vector<NEWNeighbor>> layer_neighbor(20);
        // for (auto &p: pool_) {
        //     layer_neighbor.at(p.layer_).emplace_back(p);
        // }

        for (auto &layer : layer_neighbor) {
            if (layer.empty()) {
                continue;
            }
            // if (res.size() > 80) {
            //     break;
            // }
            for (auto &nei:layer) {
                res.emplace_back(nei.id_, nei.emb_distance_, nei.geo_distance_, false, nei.layer_);
                // if (res.size() > 80) {
                //     break;
                // }
            }
        }
        return res;
    }

    std::vector<DEGNNDescentNeighbor> traverse() {
        std::vector<DEGNNDescentNeighbor> res;

        res.reserve(pool_.size());
        for (auto &nei:pool_) {
            res.emplace_back(nei.id_, nei.emb_distance_, nei.geo_distance_, false, nei.layer_);
        }
        return res;
    }


    std::vector<size_t> traverse_layer() {
        std::vector<size_t> res;
        res.reserve(20);
        int layer = -1;
        for (size_t i = 0; i < pool_.size(); i++) {
            auto &p = pool_.at(i);
            if (layer == -1 && p.flag_) {
                res.emplace_back(i);
                layer = p.layer_;
            } else if (p.layer_ == layer && p.flag_) {
                res.emplace_back(i);
            }
        }

        return res;
    }

    std::vector<unsigned> traverse_layer_n() {
        std::vector<unsigned> res;
        res.reserve(20);
        int layer = -1;
        for (size_t i = 0; i < pool_.size(); i++) {
            auto &p = pool_.at(i);
            if (layer == -1 && p.flag_) {
                res.emplace_back(p.id_);
                layer = p.layer_;
                p.flag_ = false;
            } else if (p.layer_ == layer && p.flag_) {
                res.emplace_back(p.id_);
                p.flag_ = false;
            }
        }

        return res;
    }

    NEWNeighbor& at(size_t idx) {
        return pool_.at(idx);
    }

    const NEWNeighbor& at(size_t idx) const {
        return pool_.at(idx);
    }
 
    void remove(unsigned id, std::vector<NEWSimpleNeighbor> &nodes) {
        auto remove_pos = std::find_if(pool_.begin(), pool_.end(), 
        [id](const NEWNeighbor& neighbor) noexcept {
            return neighbor.id_ == id;
        });
    
        // 提前检查元素是否存在
        if (remove_pos == pool_.end()) {
            return; // 元素不存在，直接返回
        }
        const size_t remove_idx = remove_pos - pool_.begin();
        const float remove_e_d = remove_pos->emb_distance_;
        const int remove_layer = pool_[remove_idx].layer_;
        float update_min_y = std::numeric_limits<float>::max();

        // 修复1: 添加层级上限防护
        constexpr int MAX_LAYER = 100;  // 设置合理的最大层级
        if (remove_layer > MAX_LAYER) {
            SKY_ASSERT(false, "Remove layer exceeds maximum allowed");
            return;
        }

        if (remove_idx > 0) {
            const auto& prev_neighbor = pool_[remove_idx - 1];
            if (remove_layer < prev_neighbor.min_y_list.size()) {
                update_min_y = prev_neighbor.min_y_list.at(remove_layer);
            } else {
                // 移除的是新开的一层，需要从前面的邻居重新计算
                update_min_y = 0;
            }
        }


        // 批量处理受影响的节点
        auto max_layer = 0;
        std::vector<size_t> affected_indices;
        affected_indices.reserve(pool_.size() - remove_idx - 1);
        
        // 第一遍：识别所有受影响的节点
        for (size_t i = remove_idx + 1; i < pool_.size(); ++i) {
            auto& current_neighbor = pool_[i];
            max_layer = std::max(max_layer, current_neighbor.layer_);
            
            if (std::abs(current_neighbor.min_y_list[remove_layer]-remove_e_d) < 1e-6f) {
                current_neighbor.min_y_list[remove_layer] = update_min_y;
            }
            if (current_neighbor.layer_ < remove_layer) continue;
            
            // 如果这个节点在被移除层受到影响
            // if (current_neighbor.min_y_list[remove_layer] == remove_e_d) {
            if (std::abs(current_neighbor.min_y_list[remove_layer]-remove_e_d) < 1e-6f) {
                affected_indices.push_back(i);
            }
        }

        update_nodes_.clear();
        update_nodes_.resize(max_layer+1);
        update_nodes_.emplace_back();
        
        nodes.reserve(nodes.size() + affected_indices.size());
        // 第二遍：处理受影响的节点
        for (size_t idx : affected_indices) {
            auto& affected_neighbor = pool_[idx];
            
            // 更新min_y值
            affected_neighbor.min_y_list[remove_layer] = update_min_y;
            
            // 检查是否需要降层
            if (affected_neighbor.emb_distance_ < update_min_y) {
                auto& prev_neighbor = pool_[idx-1];
                if (prev_neighbor.min_y_list.size() < affected_neighbor.min_y_list.size()) {
                    // std::cout<< prev_neighbor.min_y_list.size() << " " << affected_neighbor.min_y_list.size() << std::endl;
                    auto size_t_layer = static_cast<size_t>(affected_neighbor.layer_);
                    if (size_t_layer >= affected_neighbor.min_y_list.size()) {
                        std::cout<< size_t_layer << " " << affected_neighbor.min_y_list.size() << std::endl;
                        SKY_ASSERT(false, "error layer");
                    }
                    affected_neighbor.min_y_list[affected_neighbor.layer_] = 
                        std::numeric_limits<float>::max();
                } else {
                    affected_neighbor.min_y_list[affected_neighbor.layer_] = 
                        prev_neighbor.min_y_list[affected_neighbor.layer_];
                }
                // 节点需要降层
                affected_neighbor.layer_--;
                auto size_t_layer = static_cast<size_t>(affected_neighbor.layer_);
                if (size_t_layer >= affected_neighbor.min_y_list.size()) {
                    std::cout<< size_t_layer << " " << affected_neighbor.min_y_list.size() << std::endl;
                    SKY_ASSERT(false, "error layer");
                }
                affected_neighbor.min_y_list[affected_neighbor.layer_] = affected_neighbor.emb_distance_;
                
                // 添加到更新列表
                nodes.emplace_back(affected_neighbor.id_, 
                                affected_neighbor.emb_distance_,
                                affected_neighbor.geo_distance_,
                                affected_neighbor.layer_);
                
                // 更新update_nodes_结构
                size_t new_layer_diff = affected_neighbor.layer_ - remove_layer;
                // while (update_nodes_.size() <= new_layer_diff) {
                //     update_nodes_.emplace_back();
                //     update_nodes_.back().reserve(4);
                // }
                
                if (new_layer_diff < update_nodes_.size()) {
                    update_nodes_[new_layer_diff].push_back(affected_neighbor.emb_distance_);
                }
            }
        }
        
        // 优化7: 第三遍：处理层级传播效应
        for (size_t i = remove_idx + 1; i < pool_.size(); ++i) {
            auto& current_neighbor = pool_[i];
            
            if (current_neighbor.layer_ < remove_layer) continue;
            
            size_t layer_diff = current_neighbor.layer_ - remove_layer;
            if (layer_diff < update_nodes_.size()) {
                const auto& update_list = update_nodes_[layer_diff];
                
                for (float e_distance : update_list) {
                    if (e_distance < current_neighbor.emb_distance_) {
                        current_neighbor.layer_++;
                        size_t new_layer_diff = current_neighbor.layer_ - remove_layer;
                        
                        // 确保update_nodes_有足够空间
                        // while (update_nodes_.size() <= new_layer_diff) {
                        //     update_nodes_.emplace_back();
                        //     update_nodes_.back().reserve(4);
                        // }
                        
                        update_nodes_[new_layer_diff].push_back(current_neighbor.emb_distance_);
                        break;
                    }
                }
            }
        }
        
        // 优化8: 最后移除元素
        pool_.erase(remove_pos);
        // for (auto &p : pool_) {
        //     validate_min_y_list(p, "remove");
        // }
        return ;
    }

    size_t size() {
        return pool_.size();
    }

private:
    void validate_min_y_list(const NEWNeighbor& neighbor, std::string s) {
        // 优化7: 更严格和清晰的验证逻辑
        if (neighbor.layer_ < 0) {
            std::string t = "Invalid layer: negative value ";
            t+=s;
            SKY_ASSERT(false,  t.c_str());
            return;
        }
        
        size_t layer_idx = static_cast<size_t>(neighbor.layer_);
        
        if (layer_idx > neighbor.min_y_list.size()) {
            std::string t = "Layer index out of bounds in min_y_list ";
            t+=s;
            SKY_ASSERT(false,  t.c_str());
            return;
        }
        
        // 验证当前层的值是否正确
        float expected_value = neighbor.emb_distance_;
        float actual_value = neighbor.min_y_list.at(layer_idx);
        
        if (std::abs(expected_value - actual_value) >= 1e-6f) {
            std::cout<<layer_idx<<std::endl;
            for (auto& e_d : neighbor.min_y_list) {
                std::cout << e_d << std::endl;
            }
            std::cout<<expected_value << " " <<actual_value << std::endl;
            std::string t = "min_y_list value inconsistent with emb_distance ";
            t+=s;
            SKY_ASSERT(false,  t.c_str());
        }
    }

};
}

#endif