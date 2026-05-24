# 与sliding不同的是，这个脚本是用来生成仅批插入或批删除的gt文件

./dynamic_gt \
  base_img.fvecs \
  base_text.fvecs \
  query_img.fvecs \
  query_text.fvecs \
  alpha_file.fvecs \
  output_dir \
  init_ratio \
  update_ratio \
  update_rounds \
  mode


✅ 示例一：批插入（insert）
场景说明

base 向量总数：假设 100,000

初始构建索引：前 50%

每轮插入：1%

插入轮数：10 轮

每一轮都会重新生成 GT

命令示例
./dynamic_gt \
  ./data/base_img.fvecs \
  ./data/base_text.fvecs \
  ./data/query_img.fvecs \
  ./data/query_text.fvecs \
  ./data/query_alpha.fvecs \
  ./output/dynamic_gt_insert \
  0.5 \
  0.01 \
  10 \
  insert

含义逐项解释
参数	含义
0.5	初始索引包含 base 的 50%（ID: 0 ～ 49999）
0.01	每轮插入 1%（1000 条）
10	插入 10 次
insert	插入模式
实际 active 变化
Round 0: [0 .. 49999]
Round 1: [0 .. 50999]
Round 2: [0 .. 51999]
...
Round 10: [0 .. 59999]

✅ 示例二：批删除（delete）
场景说明

初始索引：前 50%

每轮删除：1%

删除 10 轮

命令示例
./dynamic_gt \
  ./data/base_img.fvecs \
  ./data/base_text.fvecs \
  ./data/query_img.fvecs \
  ./data/query_text.fvecs \
  ./data/query_alpha.fvecs \
  ./output/dynamic_gt_delete \
  0.5 \
  0.01 \
  10 \
  delete

实际 active 变化
Round 0: [0 .. 49999]
Round 1: [1000 .. 49999]
Round 2: [2000 .. 49999]
...
Round 10: [10000 .. 49999]

📂 输出目录结构示例
output/
└── dynamic_gt_insert/
    ├── round_00/
    │   └── top10_results.ivecs
    ├── round_01/
    │   └── top10_results.ivecs
    ├── round_02/
    │   └── top10_results.ivecs
    └── ...

./gen_gt /data/linsy/HVS/dataset/openimg_1w/base_img_emb.fvecs\ 
     /data/linsy/HVS/dataset/openimg_1w/base_text_emb.fvecs\
     /data/linsy/HVS/dataset/openimg_1w/query_img_emb.fvecs\
     /data/linsy/HVS/dataset/openimg_1w/query_text_emb.fvecs\ 
     /data/linsy/HVS/dataset/openimg_1w/range_3_query_alpha.fvecs\ 
     /data/linsy/HVS/dataset/openimg_1w/gt_\ 
     0.9 0.01 10 delete


./gen_gt /data/linsy/EPDG/DEG/dataset/Openimage/base_img_emb.fvecs /data/linsy/EPDG/DEG/dataset/Openimage/base_text_emb.fvecs /data/linsy/EPDG/DEG/dataset/Openimage/query_img_emb.fvecs /data/linsy/EPDG/DEG/dataset/Openimage/query_text_emb.fvecs /data/linsy/EPDG/DEG/dataset/Openimage/alpha.fvecs /data/linsy/EPDG/DEG/dataset/Openimage/build_90/gt 0.9 0.01 10 delete