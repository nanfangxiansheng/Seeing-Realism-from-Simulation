
export TMPDIR=
begin_id=0
end_id=0
gpu_a=0
gpu_b=1
master_port=12304
ctrl_id=0

out_dir=""
#note that the out_dir needs to be set
while [[ $# -gt 0 ]]; do
    case "$1" in
        --begin)
            begin_id="$2"
            shift 2
            ;;
        --end)
            end_id="$2"
            shift 2
            ;;
        --gpu_a)
            gpu_a="$2"
            shift 2
            ;;
        --master_port)
            master_port="$2"
            shift 2
            ;;
        --out_dir)
            out_dir="$2"
            shift 2
            ;;

        --*)
            echo "unkonwn: $1"
            usage
            ;;
    esac
done

check_port() {
    if lsof -i :"$1" > /dev/null 2>&1; then
        echo "wrong:  $1 in use"
        exit 1
    fi
}
PYTHON=""
#set uv python path

check_port "$master_port"
for ((ctrl_id=$begin_id; ctrl_id<=$end_id; ctrl_id++)); do
    echo "dealing with"$ctrl_id
    CUDA_VISIBLE_DEVICES=$gpu_a conda run -n videochat python video_chat/chat_video_depth.py \
    --video_contain_path "" \
    --output_dir ./$out_dir \
    --ctrl_id $ctrl_id
    CUDA_VISIBLE_DEVICES=$gpu_a $PYTHON -m torch.distributed.run --nproc_per_node=1 --master_port=$master_port -m examples.inference -i $out_dir/total_$ctrl_id/control.json -o $out_dir/total_$ctrl_id
    echo "====== finish ctrl_id=$ctrl_id ======"
done