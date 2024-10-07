# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error
from typing import List, NamedTuple, Optional
<<<<<<< HEAD

from kfp.dsl import Artifact, Input, Model, Output, component, importer

from utils.consts import PYTHON_IMAGE

# TODO: replace with ilab image
<<<<<<< HEAD
EVAL_IMAGE = "quay.io/sallyom/instructlab-ocp:eval-7ee213"
=======
EVAL_IMAGE = "quay.io/sallyom/instructlab-ocp:eval-new"
>>>>>>> 1015837 (update mtbench-evaluator)
=======
from kfp.dsl import component, Input, Output, Artifact, Model, importer
from utils.consts import PYTHON_IMAGE, ILAB_IMAGE
>>>>>>> 8b21133 (use ilab image in mtbench)


@component(base_image=ILAB_IMAGE)
def run_mt_bench_op(
    models_path_prefix: str,
    mt_bench_output: Output[Artifact],
    merge_system_user_message: bool,
    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - `auto`
    # with `auto`, number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    max_workers: str,
    models_list: List[str] = None,
    models_folder: Optional[str] = None,
    device: str = None,
) -> NamedTuple("outputs", best_model=str, best_score=float):
<<<<<<< HEAD
    def launch_vllm(model_path: str, gpu_count: int, retries: int = 60, delay: int = 5):
=======
    def launch_vllm(
        model_path: str, gpu_count: int, retries: int = 60, delay: int = 5
    ):
>>>>>>> 8b21133 (use ilab image in mtbench)
        import subprocess
        import sys
        import time

        import requests

        # TODO: set centrally
        vllm_endpoint = "http://localhost:8000/v1"

        if gpu_count > 0:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
                "--tensor-parallel-size",
                str(gpu_count),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
            ]

        subprocess.Popen(args=command)

        print(f"Waiting for vLLM server to start at {vllm_endpoint}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{vllm_endpoint}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {vllm_endpoint}.")
                    return
            except requests.ConnectionError:
                pass

            print(
                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Failed to start vLLM server at {vllm_endpoint} after {retries} retries."
        )

    # This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn't work
    # Also, the base image does not include `pkill` cmd, so can't pkill -f vllm.entrypoints.openai.api_server either
    def stop_vllm():
        import psutil

        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            cmdline = process.info.get("cmdline")
            if cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                print(
                    f"Found vLLM server process with PID: {process.info['pid']}, terminating..."
                )
                try:
                    process.terminate()  # Try graceful termination
                    process.wait(timeout=5)  # Wait a bit for it to terminate
                    if process.is_running():
                        print(
                            f"Forcefully killing vLLM server process with PID: {process.info['pid']}"
                        )
                        process.kill()  # Force kill if it's still running
                    print(
                        f"Successfully stopped vLLM server with PID: {process.info['pid']}"
                    )
                except psutil.NoSuchProcess:
                    print(f"Process with PID {process.info['pid']} no longer exists.")
                except psutil.AccessDenied:
                    print(
                        f"Access denied when trying to terminate process with PID {process.info['pid']}."
                    )
                except Exception as e:
                    print(
                        f"Failed to terminate process with PID {process.info['pid']}. Error: {e}"
                    )

    import json
    import os

<<<<<<< HEAD
    import torch
    from instructlab.eval import mt_bench_answers, mt_bench_judgment
=======
    from instructlab.eval.mt_bench import MTBenchEvaluator
>>>>>>> 1015837 (update mtbench-evaluator)

    env_variables = {
        "HOME": "/ilab",
        "XDG_CACHE_HOME": "/ilab",
        "XDG_CONFIG_HOME": "/ilab",
        "XDG_DATA_HOME": "/ilab",
        "TRITON_CACHE_DIR": "/ilab",
        "HF_HOME": "/ilab",
        "TRANSFORMERS_CACHE": "/ilab",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }

    for key, value in env_variables.items():
        os.environ[key] = value

    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    print(f"GPU Available: {gpu_available}, {gpu_name}")

    if models_list is None and models_folder:
        models_list = os.listdir(models_folder)

    judge_api_key = os.getenv("JUDGE_API_KEY", "")
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    vllm_endpoint = "http://localhost:8000/v1"

    scores = {}
    all_mt_bench_data = []

    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - `auto`
    # with `auto`, number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    if max_workers == "auto":
        try:
            usable_cpu_count = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            usable_cpu_count = multiprocessing.cpu_count() // 2
        max_workers = usable_cpu_count

    for model_name in models_list:
        print(f"Serving candidate model: {model_name}")
        #model_path = f"{models_path_prefix}/{model_name}"
        # TODO: REMOVE THIS, UNCOMMENT ABOVE
        model_path = f"{models_path_prefix}/samples_588"

<<<<<<< HEAD
<<<<<<< HEAD
        launch_vllm(model_path, gpu_count)
=======
        launch_vllm_server_background(model_path, gpu_count)
>>>>>>> 1015837 (update mtbench-evaluator)
=======
        launch_vllm(model_path, gpu_count)
>>>>>>> 8b21133 (use ilab image in mtbench)

        # model ID is the model_path value in vLLM
        evaluator = MTBenchEvaluator(
            model_name=model_path,
            judge_model_name=judge_model_name,
            output_dir="/tmp/eval_output",
            merge_system_user_message=merge_system_user_message,
        )

        evaluator.gen_answers(
            server_url=vllm_endpoint,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        stop_vllm()

        overall_score, qa_pairs, turn_scores, error_rate = evaluator.judge_answers(
            server_url=judge_endpoint,
            api_key=judge_api_key,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        mt_bench_data = {
            "report_title": "SKILLS EVALUATION REPORT",
            "model": model_path,
            "judge_model": judge_model_name,
            "overall_score": overall_score,
            "turn_scores": turn_scores,
            "qa_scores": qa_pairs,
            "error_rate": error_rate,
        }

        all_mt_bench_data.append(mt_bench_data)
        scores[model_path] = overall_score

    with open(mt_bench_output.path, "w") as f:
        json.dump(all_mt_bench_data, f, indent=4)

    outputs = NamedTuple("outputs", best_model=str, best_score=float)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    return outputs(best_model=best_model, best_score=best_score)


@component(base_image=PYTHON_IMAGE)
def load_mt_bench_results_op(mt_bench_output: Input[Artifact]) -> list:
    import json

    mt_bench_score_list = []
    with open(mt_bench_output.path, "r") as f:
        mt_bench_score_list = json.load(f)

    print("MT_Bench Evaluation Data:")
    for mt_bench_score in mt_bench_score_list:
        print(json.dumps(mt_bench_score, indent=4))

    return mt_bench_score_list
