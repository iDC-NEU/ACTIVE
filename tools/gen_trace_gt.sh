#!/bin/bash

root_path=/mnt/nvme3/linsy/EPDG/DEG
cpp_path=${root_path}/tools
exec_path=${cpp_path}/exe

# 数据集列表
datasets=("LAION")

# update_ratio 和 rounds 绑定（相同索引对应一组）
update_ratios=(0.01)
rounds_list=(99)

# 编译工具
echo "========================Compiling Tools========================"
g++ ${cpp_path}/gen_alpha.cpp -o ${exec_path}/gen_alpha
g++ ${cpp_path}/process_base.cpp -o ${exec_path}/process_base
g++ ${cpp_path}/gen_trace.cpp -o ${exec_path}/gen_trace
g++ ${cpp_path}/gen_gt.cpp -o ${exec_path}/gen_gt -fopenmp

# 循环数据集
for dataset_name in "${datasets[@]}"; do
    echo "========================Processing dataset: $dataset_name ========================"
    dataset_path=${root_path}/dataset/${dataset_name}

    # 设置 dataset 相关参数
    case "$dataset_name" in
      OpenImage)
        base_num=507444
        img_dim=768
        text_dim=768
        ;;
      CC3M)
        base_num=3131153
        img_dim=768
        text_dim=768
        ;;
      Howto100M)
        base_num=1238875
        img_dim=1024
        text_dim=768
        ;;
      LAION)
        base_num=10642155
        img_dim=512
        text_dim=512
        ;;
      *)
        echo "Unknown dataset_name: $dataset_name"
        exit 1
        ;;
    esac

    base_ratio=1 # 1-update_rounds*update_ratio
    start_range=0.0
    end_range=1.0

    file_img=${dataset_path}/query_img_emb.fvecs
    alpha_file=${dataset_path}/alpha_${start_range}_${end_range}.fvecs
    alpha_file_txt=${dataset_path}/alpha_${start_range}_${end_range}.txt

    # 循环 update_ratio 和 rounds（绑定）
    for i in "${!update_ratios[@]}"; do
        update_ratio=${update_ratios[$i]}
        rounds=${rounds_list[$i]}

        echo "------------------------ dataset: $dataset_name | update_ratio: $update_ratio | rounds: $rounds ------------------------"

        # gen_alpha
        # ${exec_path}/gen_alpha ${file_img} ${start_range} ${end_range} ${alpha_file} ${alpha_file_txt}

        # process_base
        # ${exec_path}/process_base ${dataset_path} ${base_num} ${img_dim} ${text_dim} ${base_ratio}

        # gen_trace
        trace_path=${dataset_path}
        mode="delete"
        ${exec_path}/gen_trace ${base_num} ${update_ratio} ${rounds} ${trace_path} ${mode}

        # gen_gt
        base_img=${dataset_path}/base_img_emb.fvecs
        base_text=${dataset_path}/base_text_emb.fvecs
        query_img=${dataset_path}/query_img_emb.fvecs
        query_text=${dataset_path}/query_text_emb.fvecs
        threads=64
        ${exec_path}/gen_gt \
            ${base_img} \
            ${base_text} \
            ${query_img} \
            ${query_text} \
            ${alpha_file} \
            ${dataset_path} \
            ${base_ratio} \
            ${update_ratio} \
            ${rounds} \
            ${mode} \
            ${threads}
    done
done


bash /mnt/nvme3/linsy/EPDG/DEG/run/update_multi_1.sh