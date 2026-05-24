# 使用位置参数
root_path=$1
dataset_name=$2

# 验证参数是否传入
if [ -z "$root_path" ] || [ -z "$dataset_name" ]; then
    echo "用法: $0 <root_path> <dataset_name>"
    echo "示例: $0 /data/linsy/EPDG/DEG CC3M"
    exit 1
fi

# 使用参数
echo "root_path: $root_path"
echo "dataset_name: $dataset_name"

dataset_path=${root_path}/dataset/${dataset_name}
cpp_path=${root_path}/tools
exec_path=${cpp_path}/exe



echo "========================gen_base========================"
g++ ${cpp_path}/p_base.cpp -o ${exec_path}/p_base

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


base_ratio=0.5 # 1-update_rounds*update_ratio


# ${exec_path}/p_base\
#     ${dataset_path}\
#     ${base_num}\
#     ${img_dim}\
#     ${text_dim}\
#     ${base_ratio}



# echo "========================gen_alpha========================"
g++ ${cpp_path}/gen_alpha.cpp -o ${exec_path}/gen_alpha
start_range=0.0
end_range=1.0
# To get base_vector_num
file_img=${dataset_path}/query_img_emb.fvecs
alpha_file=${dataset_path}/alpha_${start_range}_${end_range}.fvecs
alpha_file_txt=${dataset_path}/alpha_${start_range}_${end_range}.txt

# ${exec_path}/gen_alpha\
#     ${file_img}\
#     ${start_range} ${end_range}\
#     ${alpha_file}\
#     ${alpha_file_txt}\


echo "========================gen_trace========================"
g++ ${cpp_path}/gen_trace.cpp -o ${exec_path}/gen_trace
update_ratio=0.01
rounds=50
trace_path=${dataset_path}/build_50
# mode=delete/insert/both
mode="both"

# ${exec_path}/gen_trace\
#     ${base_num}\
#     ${update_ratio}\
#     ${rounds}\
#     ${trace_path}\
#     ${mode}



echo "========================gen_groundtruth_sliding========================"

g++ ${cpp_path}/gen_gt_sliding_cache.cpp -o ${exec_path}/gen_gt_sliding_cache -fopenmp
base_img=${dataset_path}/base_img_emb.fvecs
base_text=${dataset_path}/base_text_emb.fvecs
query_img=${dataset_path}/query_img_emb.fvecs
query_text=${dataset_path}/query_text_emb.fvecs

update_ratio=0.01
rounds=50
${exec_path}/gen_gt_sliding_cache\
    ${dataset_path}\
    ${alpha_file}\
    ${update_ratio}\
    ${rounds}\


