#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -i <query.json> -o <output_dir> [options]"
    echo ""
    echo "Required:"
    echo "  -i  Input AF3-style EnsembleFold JSON (name/modelSeeds/sequences)."
    echo "  -o  Output/job root directory."
    echo ""
    echo "Options:"
    echo "  -d  CUDA device IDs. (default: 0)"
    echo "  -D  Run data pipeline. (default: true)"
    echo "  -P  Run inference. (default: true)"
    echo "  -r  One seed or comma-separated seeds."
    echo "  -s  Diffusion samples per seed."
    echo "  -y  Runner YAML."
    echo "  -k  Inference checkpoint path."
    echo "  -M  Use ColabFold MSA server. (default: true)"
    echo "  -T  Use templates. (default: true)"
    echo "  -m  Inclusive maximum template date, YYYY-MM-DD."
    echo "  -w  Write prepared JSON: auto/true/false. (default: auto)"
    echo "  -z  Compress prepared MSA/template files. (default: true)"
    echo "  -S  Skip complete query/seed outputs. (default: false)"
    echo "  -F  Enable TF32. (default: true)"
    echo "  -h  Show this help."
    echo ""
    echo "Examples:"
    echo "  $0 -i query.json -o result -D true -P false"
    echo "  $0 -i result/target/target_data.json -o result -D false -P true -r 40,41,42 -S true"
    exit 1
}

while getopts "i:o:d:D:P:r:s:y:k:M:T:m:w:z:S:F:h" opt; do
    case "${opt}" in
    i) input_path=$OPTARG ;;
    o) output_dir=$OPTARG ;;
    d) gpu_device=$OPTARG ;;
    D) run_data_pipeline=$OPTARG ;;
    P) run_inference=$OPTARG ;;
    r) model_seeds=$OPTARG ;;
    s) diffusion_samples=$OPTARG ;;
    y) runner_yaml=$OPTARG ;;
    k) checkpoint_path=$OPTARG ;;
    M) use_msa_server=$OPTARG ;;
    T) use_templates=$OPTARG ;;
    m) max_template_date=$OPTARG ;;
    w) write_input_json=$OPTARG ;;
    z) compress_fold_input=$OPTARG ;;
    S) skip=$OPTARG ;;
    F) use_tf32=$OPTARG ;;
    h) usage ;;
    *) usage ;;
    esac
done

if [[ "$input_path" == "" || "$output_dir" == "" ]]; then usage; fi
if [[ ! -f "$input_path" ]]; then
    echo "Error: input JSON does not exist: $input_path"
    exit 1
fi

if [[ "$gpu_device" == "" ]]; then gpu_device="0"; fi
if [[ "$run_data_pipeline" == "" ]]; then run_data_pipeline="true"; fi
if [[ "$run_inference" == "" ]]; then run_inference="true"; fi
if [[ "$use_msa_server" == "" ]]; then use_msa_server="true"; fi
if [[ "$use_templates" == "" ]]; then use_templates="true"; fi
if [[ "$write_input_json" == "" ]]; then write_input_json="auto"; fi
if [[ "$compress_fold_input" == "" ]]; then compress_fold_input="true"; fi
if [[ "$skip" == "" ]]; then skip="false"; fi
if [[ "$use_tf32" == "" ]]; then use_tf32="true"; fi

if [[ "$run_data_pipeline" == "false" && "$run_inference" == "false" ]]; then
    echo "Error: run_data_pipeline and run_inference cannot both be false."
    exit 1
fi
if [[ "$write_input_json" == "auto" ]]; then
    if [[ "$run_data_pipeline" == "false" ]]; then
        write_input_json="false"
    else
        write_input_json="true"
    fi
fi

openfold_bin="${OPENFOLD_BIN:-run_openfold}"
if ! command -v "$openfold_bin" >/dev/null 2>&1; then
    echo "Error: OpenFold executable is unavailable: $openfold_bin"
    exit 1
fi

# An absolute OPENFOLD_BIN does not activate its Pixi/Conda environment.  Triton
# 3.6 otherwise fails to discover the CUDA assembler on Blackwell GPUs even
# when the environment ships a sufficiently new ptxas binary.
openfold_executable="$(command -v "$openfold_bin")"
openfold_bin_directory="$(dirname "$openfold_executable")"
if [[ -x "$openfold_bin_directory/ptxas" ]]; then
    export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$openfold_bin_directory/ptxas}"
    export TRITON_PTXAS_BLACKWELL_PATH="${TRITON_PTXAS_BLACKWELL_PATH:-$openfold_bin_directory/ptxas}"
fi

if [[ "$run_inference" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_device"
fi

command_args=(
    predict
    --query_json "$input_path"
    --output_dir "$output_dir"
    --run_data_pipeline "$run_data_pipeline"
    --run_inference "$run_inference"
    --write_input_json "$write_input_json"
    --compress_fold_input "$compress_fold_input"
    --use_msa_server "$use_msa_server"
    --use_templates "$use_templates"
    --skip "$skip"
    --use_tf32 "$use_tf32"
)

if [[ "$model_seeds" != "" ]]; then command_args+=(--seeds "$model_seeds"); fi
if [[ "$diffusion_samples" != "" ]]; then
    command_args+=(--num_diffusion_samples "$diffusion_samples")
fi
if [[ "$runner_yaml" != "" ]]; then command_args+=(--runner_yaml "$runner_yaml"); fi
if [[ "$checkpoint_path" != "" ]]; then
    command_args+=(--inference_ckpt_path "$checkpoint_path")
fi
if [[ "$max_template_date" != "" ]]; then
    command_args+=(--max_template_date "$max_template_date")
fi

echo "$openfold_bin ${command_args[*]}"
"$openfold_bin" "${command_args[@]}"
