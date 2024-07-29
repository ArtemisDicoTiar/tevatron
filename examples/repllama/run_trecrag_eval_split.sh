#4개로 병렬화
#parallel=4
#device=0
#for ((i = $device; i <= 59; i += $parallel)); do
#  formatted_number=$(printf "%02d" $i)
#  sh run_trecrag_doc_encode.sh $device 16 \
#  /workspace/trecrag/data/corpus/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_${formatted_number}.json \
#  /workspace/trecrag/data/corpus/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_${formatted_number}.pkl
#done

set -e;

input_dir=$1
qrels_file=$2
output_file=$3
echo "====================="
echo "input_dir: ${input_dir}"
echo "qrels_file: ${qrels_file}"
echo "output_file: ${output_file}"
echo "====================="

# eval
python3 evaluate_segment_sharded_rankings.py \
  --input_dir ${input_dir} \
  --qrels_file ${qrels_file} \
  --output_file ${output_file}
