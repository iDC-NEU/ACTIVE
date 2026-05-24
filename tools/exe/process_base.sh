# 从原始向量中提取出90%当作初始索引


./data/
├── base_img_emb.fvecs
└── base_text_emb.fvecs


# 提取的向量在 ./data/build/


# 提取前90%的向量出来，如果要修改初始构建的数量，需要修改cpp文件本身
./make_build ./data 1000000 768 512
