#include "neighbor.h"
#include "new_set.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_unordered_set.h"
#include "tbb/concurrent_vector.h"
#include "update.h"
#include <cfloat>
#include <cstddef>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cassert>

#define OnlyInNBRS true

namespace stkq
{

  void ComponentUpdateDEG::EntryInner()
  {
    update_in_one.store(0);
    update_in_two.store(0);
    update_reinsert.store(0);
    update_in_zero.store(0);
    index->get_E_Dist()
        ->init_count();
    index->get_S_Dist()->init_count();
    index->angle = index->getParam().get<int>("angle");
    index->max_m_ = index->getParam().get<unsigned>("max_m");
    index->m_ = index->max_m_;
    index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
    index->n_threads_ = index->getParam().get<unsigned>("n_threads");

    std::cout << "Update Angle: " << index->angle << ", OMP: " << index->n_threads_ << std::endl;
    index->cmp_counts.assign(index->n_threads_, 0);
    index->emb_center = new float[index->getBaseEmbDim()];
    index->loc_center = new float[index->getBaseLocDim()];
    for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
      index->emb_center[j] = 0;

    for (unsigned i = 0; i < index->getActiveIndexLen(); i++)
    {
      for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
      {
        index->emb_center[j] +=
            index->getBaseEmbData()[static_cast<size_t>(i) *
                                        index->getBaseEmbDim() +
                                    j];
      }
    }

    for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
    {
      index->emb_center[j] /= index->getActiveIndexLen();
    }

    for (unsigned j = 0; j < index->getBaseLocDim(); j++)
      index->loc_center[j] = 0;

    for (unsigned i = 0; i < index->getActiveIndexLen(); i++)
    {
      for (unsigned j = 0; j < index->getBaseLocDim(); j++)
      {
        index->loc_center[j] +=
            index->getBaseLocData()[static_cast<size_t>(i) *
                                        index->getBaseLocDim() +
                                    j];
      }
    }

    for (unsigned j = 0; j < index->getBaseLocDim(); j++)
    {
      index->loc_center[j] /= index->getActiveIndexLen();
    }
    if (index->current_rounds == 0)
    {
#pragma omp parallel for schedule(dynamic, 128)
      for (size_t i = 0; i < index->getActiveIndexLen(); i++)
      {
        auto *node = index->DEG_nodes_[i];
        auto &simple_firends = node->GetSearchFriends();
        thread_local std::vector<Index::DEGNeighbor> firends;
        // thread_local std::vector<DEGNNDescentNeighbor> tempres;
        // thread_local std::vector<DEGNNDescentNeighbor> layered_tempres;

        // tempres.clear();
        // layered_tempres.clear();
        // layered_tempres.reserve(simple_firends.size());
        // tempres.reserve(simple_firends.size());
        firends.clear();
        firends.reserve(simple_firends.size());
        for (auto &si_firend : simple_firends)
        {
          auto id = si_firend.id_;
          float ed = index->get_E_Dist()->compare(
              index->getBaseEmbData() +
                  (size_t)node->GetId() * index->getBaseEmbDim(),
              index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
              index->getBaseEmbDim());
          float sd = index->get_S_Dist()->compare(
              index->getBaseLocData() +
                  (size_t)node->GetId() * index->getBaseLocDim(),
              index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
              index->getBaseLocDim());
          std::vector<std::pair<float, float>> range;
          for (auto &[x, y] : si_firend.active_range)
          {
            auto x_f = static_cast<float>(x) / 100.0f;
            auto y_f = static_cast<float>(y) / 100.0f;
            range.emplace_back(x_f, y_f);
          }
          // tempres.emplace_back(id, ed, sd, true, -1, range);
          firends.emplace_back(id, ed, sd, range);
        }
        // update_layer(tempres, layered_tempres, max_layer);
        // {
        //   firends.emplace_back(temp.id_, temp.emb_distance_, temp.geo_distance_, temp.available_range_, temp.layer_);
        // }
        node->SetFriends(firends);
      }

      for (size_t i = 0; i < index->enterpoint_set.size(); i++)
      {
        index->DEG_enterpoints.emplace_back(
            index->DEG_nodes_[index->enterpoint_set[i]]);
      }
    }
  }

  void ComponentUpdateDEG::UpdateIndex()
  {
    for (size_t i = 0; i < index->getActiveIndexLen(); i++)
    {
      auto *node = index->DEG_nodes_[i];
      auto &firends = node->GetFriends();
      std::vector<Index::DEGSimpleNeighbor> search_firends;
      search_firends.reserve(firends.size());
      for (auto &firend : firends)
      {
        auto id = firend.id_;
        std::vector<std::pair<int8_t, int8_t>> range;
        for (auto &[x, y] : firend.available_range)
        {
          int8_t x8 = static_cast<int8_t>(x * 100);
          int8_t y8 = static_cast<int8_t>(y * 100);
          range.emplace_back(x8, y8);
        }
        search_firends.emplace_back(id, range);
      }
      firends.clear();
      node->SetSearchFriends(search_firends);
    }
  }

  void ComponentUpdateDEG::Update()
  {
    EntryInner();

    std::chrono::high_resolution_clock::time_point s;
    std::chrono::high_resolution_clock::time_point e;
    std::chrono::duration<double> time;
    s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
      auto *visited_list = new Index::VisitedList(index->getBaseLen());
      auto insert_data = index->getInsertData();
#pragma omp for schedule(dynamic, 128)
      for (size_t i = 0; i < index->getUpdateLen(); i++)
      {
        auto id = static_cast<int>(insert_data[i]);
        auto *qnode = new Index::DEGNode(id, index->max_m_);
        index->DEG_nodes_[id] = qnode;
        InsertNode(qnode, visited_list, -1);
      }
      delete visited_list;
    }
    e = std::chrono::high_resolution_clock::now();
    time = e - s;
    std::cout << "insert " << index->getUpdateLen() << " entries "
              << "TOTAL INSERT TIME: " << time.count() << "s INSERT LATENCY "
              << 1000 * time.count() / index->getUpdateLen() << "ms" << std::endl;
    s = std::chrono::high_resolution_clock::now();
    DeleteComputeInNeighbor();
    e = std::chrono::high_resolution_clock::now();
    time = e - s;
    std::cout << "delete " << index->getUpdateLen() << " entries "
              << "TOTAL DELETE TIME: " << time.count() << "s DELETE LATENCY "
              << 1000 * time.count() / index->getUpdateLen() << "ms" << std::endl;
  }

  void ComponentUpdateDEG::InsertNode(Index::DEGNode *insert_node,
                                      Index::VisitedList *visited_list, int tid)
  {
    std::vector<DEGNNDescentNeighbor> pool;
    SearchAtLayer(insert_node, visited_list, pool, tid);
    std::vector<Index::DEGNeighbor> result;
    a->DEG2NeighborOPT(insert_node->GetId(), insert_node->GetMaxM(), pool,
                       result, index->angle, tid);

    for (size_t j = 0; j < result.size(); j++)
    {
      auto *neighbor = index->DEG_nodes_[result[j].id_];
      Link(neighbor, insert_node, 0, result[j].emb_distance_,
           result[j].geo_distance_, tid);
    }
    insert_node->SetFriends(result);
    UpdateEnterpointSet(insert_node);
  }

  void ComponentUpdateDEG::Insert()
  {
    EntryInner();
    std::cout << "Start Nodes= " << index->getInsertData()[0] << " ~ " << index->getInsertData()[index->getUpdateLen() - 1] << " from " << index->getActiveIndexLen() << std::endl;
    std::chrono::high_resolution_clock::time_point s;
    std::chrono::high_resolution_clock::time_point e;
    std::chrono::duration<double> time;
    s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto *visited_list = new Index::VisitedList(index->getBaseLen());
      auto insert_data = index->getInsertData();
#pragma omp for schedule(dynamic, 128)
      for (size_t i = 0; i < index->getUpdateLen(); i++)
      {

        auto id = static_cast<int>(insert_data[i]);
        auto *qnode = new Index::DEGNode(id, index->max_m_);
        index->DEG_nodes_[id] = qnode;
        InsertNode(qnode, visited_list, -1);
        if (i % 100000 == 0)
        {
          std::cout << "inserting " << i << " / " << index->getUpdateLen()
                    << std::endl;
        }
      }
      delete visited_list;
    }
    index->setActiveIndexLen(index->getActiveIndexLen() + index->getUpdateLen());
    e = std::chrono::high_resolution_clock::now();
    time = e - s;
    long long int cmp_count = 0;
    for (int i = 0; i < index->cmp_counts.size(); i++)
      cmp_count += index->cmp_counts[i];
    long long e_times = index->get_E_Dist()->get_count();
    long long s_times = index->get_S_Dist()->get_count();
    std::cout << "delete cmp counts: " << cmp_count << " " << e_times << " " << s_times << " " << e_times + s_times << std::endl;

    std::cout << "insert " << index->getUpdateLen() << " entries "
              << "TOTAL INSERT TIME: " << time.count() << "s DELETE LATENCY "
              << 1000 * time.count() / index->getUpdateLen() << "ms " << "OPS " << index->getUpdateLen() / time.count()
              << std::endl;
    {
      delete index->emb_center;
      delete index->loc_center;
      index->enterpoint_set.clear();
      for (auto &es : index->DEG_enterpoints)
      {
        index->enterpoint_set.push_back(es->GetId());
      }
    }
  }

  void ComponentUpdateDEG::Delete()
  {
    EntryInner();
    std::chrono::high_resolution_clock::time_point s;
    std::chrono::high_resolution_clock::time_point e;
    std::chrono::duration<double> time;

    s = std::chrono::high_resolution_clock::now();
    std::cout << "Delete_multi_pf Start!" << std::endl;
    Delete_multirepair_pf();
    std::cout << "Delete_multi_pf End!" << std::endl;
    double total = update_in_one.load() + update_in_two.load() + update_reinsert.load();
    if (total == 0)
      total = 1;

    auto percent = [&](std::atomic<unsigned> &v)
    {
      return v.load() * 100.0 / total;
    };
    std::cout << "Update In_Nbrs_Size: " << total
              << " one_in_size: " << update_in_one.load()
              << " (" << percent(update_in_one) << "%)"
              << " two_in_size: " << update_in_two.load()
              << " (" << percent(update_in_two) << "%)"
              << " reinsert_size: " << update_reinsert.load()
              << " (" << percent(update_reinsert) << "%)"
              << std::endl;

    e = std::chrono::high_resolution_clock::now();
    time = e - s;
    long long cmp_count = 0;
    // long long e_times = index->get_E_Dist()->get_count();
    // long long s_times = index->get_S_Dist()->get_count();
    for (int i = 0; i < index->cmp_counts.size(); i++)
      cmp_count += index->cmp_counts[i];
    std::cout << "delete cmp counts: " << cmp_count << std::endl;
    std::cout << "delete " << index->getUpdateLen() << " entries "
              << "TOTAL DELETE TIME: " << time.count() << "s DELETE LATENCY "
              << 1000.0 * time.count() / index->getUpdateLen() << "ms "
              << "OPS " << index->getUpdateLen() / time.count()
              << std::endl;
    {
      delete index->emb_center;
      delete index->loc_center;
      index->enterpoint_set.clear();
      for (auto &es : index->DEG_enterpoints)
      {
        index->enterpoint_set.push_back(es->GetId());
      }
    }
  }


  // freshdiskann




  void ComponentUpdateDEG::Delete_multirepair_pf()
  {

    // set in graph
    std::chrono::high_resolution_clock::time_point s;
    std::chrono::high_resolution_clock::time_point e;
    std::chrono::duration<double> time;
    // init reverse graph
    s = std::chrono::high_resolution_clock::now();
    if (!index->current_rounds)
    {
      SetInNBR();
      std::cout << "First init in_graph!" << std::endl;
    }
    else
    {
      while (!index->is_in_graph_finished.load(std::memory_order_acquire))
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
      std::cout << "In graph update!" << std::endl;
    }
    e = std::chrono::high_resolution_clock::now();

    time = e - s;
    std::cout << "SetINGraph Time: " << time.count() << std::endl;

    // init delete set
    s = std::chrono::high_resolution_clock::now();
    std::vector<int> delete_ids(index->getUpdateLen());
    auto update_data = index->getDeleteData();
    std::unordered_set<unsigned> update_in_id_set;

    for (size_t i = 0; i < index->getUpdateLen(); i++)
    {

      int id = update_data[i];
      delete_ids[i] = id;
      index->DEG_nodes_[id]->SetDelete(true);
      auto &in_nbrs = index->DEG_nodes_[id]->GetInNeighbor();
      for (auto &in : in_nbrs)
      {
        update_in_id_set.insert(in.id_); // 包含 delete
      }
      RemoveFromEntryPoints(delete_ids[i]);
    }
    e = std::chrono::high_resolution_clock::now();

    time = e - s;
    std::cout << "Predone Time: " << time.count() << std::endl;

    // repair graph
    std::vector<unsigned> update_in_id_vec{update_in_id_set.begin(),
                                           update_in_id_set.end()};
#pragma omp parallel
    {
      // 每个线程一份，在线程内部复用
      int tid = omp_get_thread_num();
      auto *visited_list = new Index::VisitedList(index->getBaseLen());
      std::vector<DEGNNDescentNeighbor> tempres;
      std::unordered_set<unsigned> deleted_nbrs;
      std::unordered_set<unsigned> id_delete_set;
      std::vector<DEGNNDescentNeighbor> neighbors_tmp;
      std::vector<DEGNNDescentNeighbor> layered_tempres;
      std::vector<Index::DEGNeighbor> result;
      std::vector<Index::DEGNeighbor> result_filter;
      std::unordered_set<unsigned> id_set;
#pragma omp for schedule(dynamic, 128)
      for (size_t i = 0; i < update_in_id_vec.size(); i++)
      {
        auto &qnode = index->DEG_nodes_[update_in_id_vec[i]];
        if (qnode->GetDelete())
        {
          continue;
        }
        auto &neighbors = qnode->GetFriends();

        // 清空，而不是重新分配
        deleted_nbrs.clear();
        id_delete_set.clear();
        tempres.clear();
        result.clear();
        result_filter.clear();
        id_set.clear();
        layered_tempres.clear();
        visited_list->Reset();
        neighbors_tmp.clear();

        for (auto &nbr : neighbors)
        {
          if (index->DEG_nodes_[nbr.id_]->GetDelete())
          {
            auto ret = id_delete_set.insert(nbr.id_);
            if (ret.second)
              deleted_nbrs.insert(nbr.id_);
          }
          else
          {
            if (visited_list->NotVisited(nbr.id_) && nbr.id_ != qnode->GetId())
            {
              tempres.emplace_back(nbr.id_, nbr.emb_distance_, nbr.geo_distance_, true,
                                   -1);
              visited_list->MarkAsVisited(nbr.id_);
            }
          }
        }
        int nbrs_size = tempres.size();
        if (qnode->GetInNeighbor().size() != 0 && nbrs_size >= int(qnode->GetMaxM() * 2 / 3))
        {
          // one_in
          // get active in_nbrs
          for (const auto &in_n : qnode->GetInNeighbor())
          {
            if (!index->DEG_nodes_[in_n.id_]->GetDelete())
            {
              // not delete
              if (visited_list->NotVisited(in_n.id_) && in_n.id_ != qnode->GetId())
              {
                tempres.emplace_back(in_n.id_, in_n.emb_distance_, in_n.geo_distance_,
                                     true, -1);
                visited_list->MarkAsVisited(in_n.id_);
              }
            }
          }
        }
        else if (qnode->GetInNeighbor().size() != 0 && nbrs_size >= int(qnode->GetMaxM() / 2) && nbrs_size < int(qnode->GetMaxM() * 2 / 3))
        {
          // double_fresh
          // get active in_nbrs
          for (const auto &in_n : qnode->GetInNeighbor())
          {
            if (!index->DEG_nodes_[in_n.id_]->GetDelete())
            {
              // not delete
              if (visited_list->NotVisited(in_n.id_) && in_n.id_ != qnode->GetId())
              {
                tempres.emplace_back(in_n.id_, in_n.emb_distance_, in_n.geo_distance_,
                                     true, -1);
                visited_list->MarkAsVisited(in_n.id_);
              }
            }
          }
          for (auto &nbr : neighbors)
          {
            neighbors_tmp.emplace_back(nbr.id_, nbr.emb_distance_, nbr.geo_distance_, true,
                                       -1);
          }
          int max_layer = 0;
          update_layer(neighbors_tmp, layered_tempres, max_layer);
          // 哪些层包含删除点
          std::unordered_set<int> deleted_layers;

          // 每一层在 layered_tempres 中的起始下标
          std::unordered_map<int, int> layer_start_loc;

          int current_layer = -1;

          for (int j = 0; j < static_cast<int>(layered_tempres.size()); ++j)
          {
            auto &lnode = layered_tempres[j];

            if (lnode.layer_ > current_layer)
            {
              current_layer = lnode.layer_;
              layer_start_loc[current_layer] = j;
            }

            if (index->DEG_nodes_[lnode.id_]->GetDelete())
            {
              deleted_layers.insert(current_layer);
            }
          }

          for (auto &dlayer : deleted_layers)
          {
            for (int j = layer_start_loc[dlayer]; j < layered_tempres.size(); j++)
            {
              if (layered_tempres[j].layer_ > dlayer)
                break;

              auto &onbrs = index->DEG_nodes_[layered_tempres[j].id_]->GetFriends();
              for (auto &onbr : onbrs)
              {
                if (index->DEG_nodes_[onbr.id_]->GetDelete())
                  continue;
                // not delete
                if (visited_list->NotVisited(onbr.id_) && onbr.id_ != qnode->GetId())
                {
                  float emb_d = index->get_E_Dist()->compare(
                      index->getBaseEmbData() + (size_t)onbr.id_ * index->getBaseEmbDim(),
                      index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                      index->getBaseEmbDim());
                  float geo_d = index->get_S_Dist()->compare(
                      index->getBaseLocData() + (size_t)onbr.id_ * index->getBaseLocDim(),
                      index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                      index->getBaseLocDim());
                  index->cmp_counts[tid] += 2;
                  tempres.emplace_back(onbr.id_, emb_d, geo_d, true, -1);
                  visited_list->MarkAsVisited(onbr.id_);
                }
              }
            }
          }
        }
        else
        {
          // all
          // get active in_nbrs
          for (const auto &in_n : qnode->GetInNeighbor())
          {
            if (!index->DEG_nodes_[in_n.id_]->GetDelete())
            {
              // not delete
              if (visited_list->NotVisited(in_n.id_) && in_n.id_ != qnode->GetId())
              {
                tempres.emplace_back(in_n.id_, in_n.emb_distance_, in_n.geo_distance_,
                                     true, -1);
                visited_list->MarkAsVisited(in_n.id_);
              }
            }
          }
          for (auto &nbr : neighbors)
          {
            neighbors_tmp.emplace_back(nbr.id_, nbr.emb_distance_, nbr.geo_distance_, true,
                                       -1);
          }
          int max_layer = 0;
          update_layer(neighbors_tmp, layered_tempres, max_layer);
          // 哪些层包含删除点
          int now_layer_node_loc = -1;

          int current_layer = -1;

          for (int j = 0; j < static_cast<int>(layered_tempres.size()); ++j)
          {
            auto &lnode = layered_tempres[j];

            if (lnode.layer_ > current_layer)
            {
              current_layer = lnode.layer_;
              now_layer_node_loc = j;
            }

            if (index->DEG_nodes_[lnode.id_]->GetDelete())
            {
              break;
            }
          }
          for (int j = now_layer_node_loc; j < layered_tempres.size(); j++)
          {
            auto &onbrs = index->DEG_nodes_[layered_tempres[j].id_]->GetFriends();
            for (auto &onbr : onbrs)
            {
              if (index->DEG_nodes_[onbr.id_]->GetDelete())
                continue;
              // not delete
              if (visited_list->NotVisited(onbr.id_) && onbr.id_ != qnode->GetId())
              {
                float emb_d = index->get_E_Dist()->compare(
                    index->getBaseEmbData() + (size_t)onbr.id_ * index->getBaseEmbDim(),
                    index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                    index->getBaseEmbDim());
                float geo_d = index->get_S_Dist()->compare(
                    index->getBaseLocData() + (size_t)onbr.id_ * index->getBaseLocDim(),
                    index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                    index->getBaseLocDim());
                index->cmp_counts[tid] += 2;
                tempres.emplace_back(onbr.id_, emb_d, geo_d, true, -1);
                visited_list->MarkAsVisited(onbr.id_);
                if (tempres.size() >= index->ef_construction_ * 1.5)
                  break;
              }
            }
          }
        }

        a->DEG2NeighborOPT(qnode->GetId(), qnode->GetMaxM(), tempres, result, index->angle, tid);
        for (auto &res : result)
        {
          if (index->DEG_nodes_[res.id_]->GetDelete())
          {
            continue;
          }
          result_filter.emplace_back(res.id_, res.emb_distance_,
                                     res.geo_distance_, res.available_range,
                                     res.layer_, res.flag_);
        }
        qnode->SetFriends(result_filter);

        UpdateEnterpointSet(qnode);
        if (i % 100000 == 0)
        {
          std::cout << "process in_nbrs: " << i << " / " << update_in_id_vec.size()
                    << std::endl;
        }
      }
      delete visited_list;
    }
  }

  void ComponentUpdateDEG::DeleteComputeInNeighbor()
  {
    if (0)
    {
      std::chrono::high_resolution_clock::time_point s;
      std::vector<int> delete_ids;
      delete_ids.reserve(index->getUpdateLen());
      auto update_data = index->getDeleteData();
      for (size_t i = 0; i < index->getUpdateLen(); i++)
      {
        auto id = static_cast<int>(update_data[i]);
        delete_ids.emplace_back(id);
      }
      std::chrono::high_resolution_clock::time_point e;

      std::chrono::duration<double> time;

      // s = std::chrono::high_resolution_clock::now();
      std::unordered_set<unsigned> delete_set{delete_ids.begin(), delete_ids.end()};
      tbb::concurrent_unordered_set<unsigned> update_id_set;
      std::unordered_set<unsigned> update_out_id_set;
#pragma omp parallel
      {
#pragma omp for schedule(dynamic, 128)
        for (size_t i = 0; i < delete_ids.size(); ++i)
        {
          index->DEG_nodes_[delete_ids.at(i)]->SetDelete(true);
        }
      }
#pragma omp parallel for schedule(dynamic, 128)
      for (size_t i = 0; i < index->getActiveIndexLen(); i++)
      {
        auto node = index->DEG_nodes_[i];
        if (node->GetDelete())
        {
          continue;
        }
        auto neighbors = index->DEG_nodes_[i]->GetFriends();
        for (auto &n : neighbors)
        {
          if (delete_set.find(n.id_) != delete_set.end())
          {
            update_id_set.insert(i);
            node->Setis_in(true);
            break;
          }
        }
      }

      if (!OnlyInNBRS)
        for (size_t i = 0; i < delete_ids.size(); ++i)
        {
          auto &nbr = index->DEG_nodes_[delete_ids.at(i)]->GetFriends();
          for (auto &n : nbr)
          {
            if (index->DEG_nodes_[n.id_]->Getis_in() != true &&
                index->DEG_nodes_[n.id_]->GetDelete() != true)
            {
              update_out_id_set.insert(n.id_);
            }
          }
        }
      // e = std::chrono::high_resolution_clock::now();
      // time = e - s;
      for (size_t i = 0; i < delete_ids.size(); ++i)
      {
        RemoveFromEntryPoints(delete_ids[i]);
      }
      std::vector<unsigned> update_ids{update_id_set.begin(), update_id_set.end()};
      for (size_t i = 0; i < index->getActiveIndexLen(); i++)
      {
        auto node = index->DEG_nodes_[i];
        if (node->GetDelete())
        {
          continue;
        }
        auto neighbors = index->DEG_nodes_[i]->GetFriends();
        for (auto &n : neighbors)
        {
          if (update_id_set.find(n.id_) != update_id_set.end())
          {
            auto &in_neighbor = index->DEG_nodes_[n.id_]->GetInNeighbor();
            in_neighbor.emplace_back(node->GetId(), n.emb_distance_,
                                     n.geo_distance_, true, -1);
          }
        }
      }

      s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
      {
#pragma omp for schedule(dynamic, 128)
        for (size_t i = 0; i < update_ids.size(); i++)
        {
          auto *qnode = index->DEG_nodes_[update_ids.at(i)];
          if (qnode->GetDelete())
          {
            continue;
          }
          UpdateNode(qnode);
          if (i % 100000 == 0)
            std::cout << "update in_nbrs " << i << " / " << update_ids.size()
                      << std::endl;
        }
      }
      e = std::chrono::high_resolution_clock::now();
      time = e - s;
      std::cout << "update in time: " << time.count() << std::endl;

      if (!OnlyInNBRS)
      {
        s = std::chrono::high_resolution_clock::now();
        std::vector<unsigned> update_out_ids{update_out_id_set.begin(),
                                             update_out_id_set.end()};
        tbb::concurrent_unordered_set<unsigned> link_ids;
        tbb::concurrent_unordered_map<unsigned,
                                      tbb::concurrent_vector<DEGNNDescentNeighbor>>
            link_graph;
#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 128)
          for (size_t i = 0; i < update_out_ids.size(); i++)
          {
            auto *qnode = index->DEG_nodes_[update_out_ids.at(i)];
            if (qnode->GetDelete())
            {
              continue;
            }

            auto neighbors = qnode->GetFriends();
            for (auto &n : neighbors)
            {
              link_ids.insert(n.id_);
              link_graph[n.id_].emplace_back(qnode->GetId(), n.emb_distance_,
                                             n.geo_distance_, true, -1);
            }
            // UpdateOutNode(qnode);
            //     update_out_ids.size() << std::endl;
          }
        }
        std::vector<unsigned> link_ids_vec(link_ids.begin(), link_ids.end());
#pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < link_ids_vec.size(); i++)
        {
          std::vector<DEGNNDescentNeighbor> tempres;
          std::vector<Index::DEGNeighbor> result;

          unsigned id = link_ids_vec[i];
          auto *node = index->DEG_nodes_[id];
          if (node->GetDelete())
            continue;
          auto neighbors = node->GetFriends();
          std::unordered_set<unsigned> res;
          for (auto &n : neighbors)
          {

            auto *nnode = index->DEG_nodes_[n.id_];
            if (nnode->GetDelete())
            {
              continue;
            }
            auto ret = res.insert(n.id_);
            if (ret.second)
            {
              tempres.emplace_back(n.id_, n.emb_distance_, n.geo_distance_, true, -1);
            }
          }
          auto &new_nbrs = link_graph[id];
          for (auto &n : new_nbrs)
          {
            auto *nnode = index->DEG_nodes_[n.id_];
            if (nnode->GetDelete())
            {
              continue;
            }
            auto ret = res.insert(n.id_);
            if (ret.second)
            {
              tempres.emplace_back(n.id_, n.emb_distance_, n.geo_distance_, true, -1);
            }
          }

          a->DEG2NeighborOPT(id, node->GetMaxM(), tempres, result, index->angle);
          node->SetFriends(result);
        }
        e = std::chrono::high_resolution_clock::now();
        time = e - s;
        std::cout << "update out time: " << time.count() << std::endl;
      }
    }
    else
    {
      std::chrono::high_resolution_clock::time_point s;
      std::vector<int> delete_ids;
      delete_ids.reserve(index->getUpdateLen());
      auto update_data = index->getDeleteData();
      for (size_t i = 0; i < index->getUpdateLen(); i++)
      {
        auto id = static_cast<int>(update_data[i]);
        delete_ids.emplace_back(id);
        index->DEG_nodes_[delete_ids.at(i)]->SetDelete(true);
      }
      std::chrono::high_resolution_clock::time_point e;

      std::chrono::duration<double> time;

      s = std::chrono::high_resolution_clock::now();
      std::unordered_set<unsigned> delete_set{delete_ids.begin(), delete_ids.end()};
      tbb::concurrent_unordered_set<unsigned> update_id_set;

#pragma omp parallel for schedule(dynamic, 128)
      for (size_t i = 0; i < index->getActiveIndexLen(); i++)
      {
        auto node = index->DEG_nodes_[i];
        if (node->GetDelete())
        {
          continue;
        }
        const auto &neighbors = index->DEG_nodes_[i]->GetFriends();
        for (auto &n : neighbors)
        {
          if (index->DEG_nodes_[n.id_]->GetDelete())
          {
            update_id_set.insert(i);
            node->Setis_in(true);
            break;
          }
        }
      }
      e = std::chrono::high_resolution_clock::now();
      time = e - s;
      std::cout << "PreDone_1: " << time.count() << std::endl;
      s = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < delete_ids.size(); ++i)
      {
        RemoveFromEntryPoints(delete_ids[i]);
      }
      e = std::chrono::high_resolution_clock::now();
      time = e - s;
      std::cout << "PreDone_2: " << time.count() << std::endl;
      s = std::chrono::high_resolution_clock::now();
      std::vector<unsigned> update_ids{update_id_set.begin(), update_id_set.end()};
#pragma omp parallel for schedule(dynamic, 128)
      for (size_t i = 0; i < index->getActiveIndexLen(); i++)
      {
        auto node = index->DEG_nodes_[i];
        if (node->GetDelete())
        {
          continue;
        }
        const auto &neighbors = index->DEG_nodes_[i]->GetFriends();
        for (auto &n : neighbors)
        {
          if (index->DEG_nodes_[n.id_]->Getis_in())
          {
            std::unique_lock<std::mutex> lock(index->DEG_nodes_[n.id_]->GetAccessGuard());
            auto &in_neighbor = index->DEG_nodes_[n.id_]->GetInNeighbor();
            in_neighbor.emplace_back(node->GetId(), n.emb_distance_,
                                     n.geo_distance_, true, -1);
          }
        }
      }
      e = std::chrono::high_resolution_clock::now();
      time = e - s;
      std::cout << "PreDone_3: " << time.count() << std::endl;
      s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
      {
#pragma omp for schedule(dynamic, 128)
        for (size_t i = 0; i < update_ids.size(); i++)
        {
          auto *qnode = index->DEG_nodes_[update_ids.at(i)];
          if (qnode->GetDelete())
          {
            continue;
          }
          UpdateNode(qnode);
          if (i % 100000 == 0)
            std::cout << "update in_nbrs " << i << " / " << update_ids.size()
                      << std::endl;
        }
      }
      e = std::chrono::high_resolution_clock::now();
      time = e - s;
      std::cout << "update in time: " << time.count() << std::endl;
    }
  }

  void ComponentUpdateDEG::UpdateNode(Index::DEGNode *update_node)
  {

    // ComponentDEGPruneHeuristic *a = new ComponentDEGPruneHeuristic(index);
    std::vector<DEGNNDescentNeighbor> tempres;
    std::vector<Index::DEGNeighbor> result;
    std::vector<Index::DEGNeighbor> result_filter;
    std::unordered_set<unsigned> id_set;
    std::vector<DEGNNDescentNeighbor> tmp;

    auto *node1 = update_node;

    UpdateEnterpointSet(node1);
    std::vector<Index::DEGNeighbor> &node1_friends = node1->GetFriends();

    for (const auto &f : node1_friends)
    {
      auto *fnode = index->DEG_nodes_[f.id_];
      if (fnode->GetDelete())
      {
        continue;
      }
      auto ret = id_set.insert(f.id_);
      if (ret.second)
      {
        tempres.emplace_back(f.id_, f.emb_distance_, f.geo_distance_, true, -1);
      }
    }
    for (const auto &in_n : node1->GetInNeighbor())
    {
      auto ret = id_set.insert(in_n.id_);
      if (ret.second)
      {
        tempres.emplace_back(in_n.id_, in_n.emb_distance_, in_n.geo_distance_,
                             true, -1);
      }
    }

    a->DEG2NeighborOPT(update_node->GetId(), update_node->GetMaxM(), tempres,
                       result, index->angle);
    for (auto &res : result)
    {
      if (index->DEG_nodes_[res.id_]->GetDelete())
      {
        continue;
      }
      result_filter.emplace_back(res.id_, res.emb_distance_, res.geo_distance_,
                                 res.available_range, res.layer_, res.flag_);
      // LinkUpdate(index->DEG_nodes_[res.id_], node1, 0, res.emb_distance_, res.geo_distance_);
    }

    {
      node1->SetFriends(result_filter);
    }

    // add link_edge
    // {
    //   auto* cand = index->DEG_nodes_[t.id_];
    //   {
    //     continue;
    //   }
    //   LinkUpdate(cand, node1, 0, t.emb_distance_, t.geo_distance_);
    // }
  }

  void ComponentUpdateDEG::UpdateOutNode(Index::DEGNode *update_node)
  {
    // ComponentDEGPruneHeuristic *a = new ComponentDEGPruneHeuristic(index);
    std::vector<DEGNNDescentNeighbor> tempres;
    std::unordered_set<unsigned> id_set;

    auto *node1 = update_node;
    {
      std::unique_lock<std::mutex> lock(node1->GetAccessGuard());
      UpdateEnterpointSet(node1);
      std::vector<Index::DEGNeighbor> &node1_friends = node1->GetFriends();

      for (const auto &f : node1_friends)
      {
        auto *fnode = index->DEG_nodes_[f.id_];
        if (fnode->GetDelete())
        {
          continue;
        }
        auto ret = id_set.insert(f.id_);
        if (ret.second)
        {
          tempres.emplace_back(f.id_, f.emb_distance_, f.geo_distance_, true, -1);
        }
      }
    }

    for (auto &t : tempres)
    {
      auto *cand = index->DEG_nodes_[t.id_];
      if (cand->GetDelete())
      {
        continue;
      }
      LinkUpdate(cand, node1, 0, t.emb_distance_, t.geo_distance_);
    }
  }

  void ComponentUpdateDEG::RecomputeDistance()
  {
    for (size_t i = 0; i < index->getActiveIndexLen(); i++)
    {
      auto *unode = index->DEG_nodes_[i];
      auto unode_id = unode->GetId();
      auto &search_friends = unode->GetSearchFriends();
      std::vector<Index::DEGNeighbor> friends;
      friends.reserve(search_friends.size());

      for (size_t i = 0; i < search_friends.size(); i++)
      {
        auto friends_id = search_friends[i].id_;
        float e_d = index->get_E_Dist()->compare(
            index->getBaseEmbData() + (size_t)unode_id * index->getBaseEmbDim(),
            index->getBaseEmbData() + (size_t)friends_id * index->getBaseEmbDim(),
            index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(
            index->getBaseLocData() + (size_t)unode_id * index->getBaseLocDim(),
            index->getBaseLocData() + (size_t)friends_id * index->getBaseLocDim(),
            index->getBaseLocDim());
        // TODO active range
        // search_friends[i].active_range
        friends.emplace_back(friends_id, e_d, s_d);
      }
    }
  }

  void ComponentUpdateDEG::RemoveFromEntryPoints(unsigned delete_id)
  {
    // 从DEG_enterpoints_skyeline中移除
    auto it_skyline =
        std::remove_if(index->DEG_enterpoints_skyeline.begin(),
                       index->DEG_enterpoints_skyeline.end(),
                       [delete_id](const DEGNNDescentNeighbor &neighbor)
                       {
                         return neighbor.id_ == delete_id;
                       });
    index->DEG_enterpoints_skyeline.erase(it_skyline,
                                          index->DEG_enterpoints_skyeline.end());

    index->DEG_enterpoints.erase(
        std::remove_if(index->DEG_enterpoints.begin(),
                       index->DEG_enterpoints.end(),
                       [delete_id](Index::DEGNode *node)
                       { return node->GetId() == delete_id; }),
        index->DEG_enterpoints.end());
  }
  void ComponentUpdateDEG::RemoveFromEntryPoints()
  {
    // 从DEG_enterpoints_skyeline中移除
    auto it_skyline =
        std::remove_if(index->DEG_enterpoints_skyeline.begin(),
                       index->DEG_enterpoints_skyeline.end(),
                       [this](const DEGNNDescentNeighbor &neighbor)
                       {
                         return index->DEG_nodes_[neighbor.id_]->GetDelete();
                       });
    index->DEG_enterpoints_skyeline.erase(it_skyline,
                                          index->DEG_enterpoints_skyeline.end());

    index->DEG_enterpoints.erase(
        std::remove_if(index->DEG_enterpoints.begin(),
                       index->DEG_enterpoints.end(),
                       [](Index::DEGNode *node)
                       { return node->GetDelete(); }),
        index->DEG_enterpoints.end());
  }
  void ComponentUpdateDEG::UpdateEnterpointSet(Index::DEGNode *qnode)
  {
    float e_d = index->get_E_Dist()->compare(
        index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
        index->emb_center, index->getBaseEmbDim());

    float s_d = index->get_S_Dist()->compare(
        index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
        index->loc_center, index->getBaseLocDim());

    {
      std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex);

      index->DEG_enterpoints_skyeline.push_back(
          DEGNNDescentNeighbor(qnode->GetId(), e_d, s_d, true, 0));

      sort(index->DEG_enterpoints_skyeline.begin(),
           index->DEG_enterpoints_skyeline.end());

      float max_emb_dis = 0;
      float min_emb_dis = 1e9;

      std::vector<DEGNNDescentNeighbor> skyline;

      for (auto it = index->DEG_enterpoints_skyeline.rbegin();
           it != index->DEG_enterpoints_skyeline.rend(); ++it)
      {
        if (it->emb_distance_ > max_emb_dis)
        {
          skyline.push_back(*it);
          max_emb_dis = it->emb_distance_;
        }
      }

      index->DEG_enterpoints_skyeline.swap(skyline);

      index->DEG_enterpoints.clear();

      for (int i = 0; i < index->DEG_enterpoints_skyeline.size(); i++)
      {
        index->DEG_enterpoints.push_back(
            index->DEG_nodes_[index->DEG_enterpoints_skyeline[i].id_]);
      }
    }
  }

  void ComponentUpdateDEG::SearchAtLayer(
      Index::DEGNode *qnode, Index::VisitedList *visited_list,
      std::vector<DEGNNDescentNeighbor> &pool, int tid)
  {
    visited_list->Reset();
    unsigned ef_construction = index->ef_construction_;
    unsigned query = qnode->GetId();

    std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex,
                                                 std::defer_lock);

    enterpoint_lock.lock();

    for (int i = 0; i < index->DEG_enterpoints.size(); i++)
    {
      auto &enterpoint = index->DEG_enterpoints[i];

      unsigned enterpoint_id = enterpoint->GetId();

      float e_d = index->get_E_Dist()->compare(
          index->getBaseEmbData() + (size_t)query * index->getBaseEmbDim(),
          index->getBaseEmbData() +
              (size_t)enterpoint_id * index->getBaseEmbDim(),
          index->getBaseEmbDim());

      float s_d = index->get_S_Dist()->compare(
          index->getBaseLocData() + (size_t)query * index->getBaseLocDim(),
          index->getBaseLocData() +
              (size_t)enterpoint_id * index->getBaseLocDim(),
          index->getBaseLocDim());
      if (tid != -1)
        index->cmp_counts[tid] += 2;
      pool.emplace_back(enterpoint_id, e_d, s_d, true, 0);

      visited_list->MarkAsVisited(enterpoint_id);
    }

    enterpoint_lock.unlock();

    sort(pool.begin(), pool.end());
    auto queue = Index::skyline_queue(ef_construction);

    queue.init_queue(pool);

    int k = 0;
    int l = 0;

    while (k < queue.pool.size())
    {
      while (queue.pool[k].layer_ == l)
      {
        if (queue.pool[k].flag)
        {
          queue.pool[k].flag = false;
          unsigned n = queue.pool[k].id_;
          Index::DEGNode *candidate_node = index->DEG_nodes_[n];
          std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
          const std::vector<Index::DEGNeighbor> &neighbors =
              candidate_node->GetFriends();
          for (unsigned m = 0; m < neighbors.size(); ++m)
          {
            unsigned id = neighbors[m].id_;
            if (visited_list->NotVisited(id))
            {
              visited_list->MarkAsVisited(id);

              float e_d = index->get_E_Dist()->compare(
                  index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                  index->getBaseEmbData() +
                      (size_t)query * index->getBaseEmbDim(),
                  index->getBaseEmbDim());

              float s_d = index->get_S_Dist()->compare(
                  index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                  index->getBaseLocData() +
                      (size_t)query * index->getBaseLocDim(),
                  index->getBaseLocDim());
              if (tid != -1)
                index->cmp_counts[tid] += 2;
              queue.pool.emplace_back(id, e_d, s_d, true, -1);
            }
          }
        }
        k++;
        if (k >= queue.pool.size())
        {
          break;
        }
      }
      int nk = 0;
      queue.updateNeighbor(nk);
      k = nk;
      if (k < queue.pool.size())
      {
        l = queue.pool[k].layer_;
      }
    }

    pool.swap(queue.pool);
  }



  void ComponentUpdateDEG::Link(Index::DEGNode *source, Index::DEGNode *target,
                                int level, float e_dist, float s_dist, int tid)
  {
    std::unique_lock<std::mutex> lock(source->GetAccessGuard());
    std::vector<Index::DEGNeighbor> &neighbors = source->GetFriends();
    std::vector<DEGNNDescentNeighbor> tempres;
    std::vector<Index::DEGNeighbor> result;
    tempres.emplace_back(
        DEGNNDescentNeighbor(target->GetId(), e_dist, s_dist, true, -1));
    for (const auto &neighbor : neighbors)
    {
      tempres.emplace_back(
          DEGNNDescentNeighbor(neighbor.id_, neighbor.emb_distance_,
                               neighbor.geo_distance_, true, -1));
    }
    a->DEG2NeighborOPT(source->GetId(), source->GetMaxM(), tempres, result, index->angle, tid);
    source->SetFriends(result);
  }

  void ComponentUpdateDEG::LinkUpdate(Index::DEGNode *source,
                                      Index::DEGNode *target, int level,
                                      float e_dist, float s_dist)
  {
    std::unique_lock<std::mutex> lock(source->GetAccessGuard());
    std::vector<Index::DEGNeighbor> &neighbors = source->GetFriends();
    std::vector<DEGNNDescentNeighbor> tempres;
    std::vector<Index::DEGNeighbor> result;
    tempres.emplace_back(
        DEGNNDescentNeighbor(target->GetId(), e_dist, s_dist, true, -1));
    for (const auto &neighbor : neighbors)
    {
      if (neighbor.id_ == target->GetId())
      {
        return;
      }
      if (index->DEG_nodes_[neighbor.id_]->GetDelete())
      {
        continue;
      }
      tempres.emplace_back(
          DEGNNDescentNeighbor(neighbor.id_, neighbor.emb_distance_,
                               neighbor.geo_distance_, true, -1));
    }
    neighbors.clear();
    // ComponentDEGPruneHeuristic *a = new ComponentDEGPruneHeuristic(index);
    a->DEG2NeighborOPT(source->GetId(), source->GetMaxM(), tempres, result, index->angle);
    source->SetFriends(result);
    std::vector<DEGNNDescentNeighbor>().swap(tempres);
    std::vector<Index::DEGNeighbor>().swap(result);
  }
} // namespace stkq
