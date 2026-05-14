# 1) 시스템 패키지 설치
apt-get update
apt-get install -y git git-lfs python3 python3-pip

# 2) Git LFS 활성화
git lfs install

# 3) 저장소 clone
cd /workspace
git clone https://github.com/limjh0330/MedCon_KG.git
cd MedCon_KG

# 4) LFS 파일 내려받기
git lfs pull

# 5) 의존성 설치
pip install -U pip
pip install -r requirements.txt           
pip install -r experiments/requirements.txt 

# 6) CUDA 가용성 점검
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 7) 실험 실행
mkdir -p output/full_rag_experiment

# 사용 가능한 variant: only_llm, vector_rag, kg_no_cond, kg_with_cond, all
# --llm-model 미지정 시 기본값: meta-llama/Llama-3.1-8B-Instruct
# iLlama로 실행하려면 아래 명령에 --llm-model Codingchild/ILlama-8b-LoRA 추가
nohup python -u -m experiments.run \
    --dataset mediq \
    --dataset-path ./MediQ/all_craft_md.jsonl \
    --variants all \
    --output-dir ./output/full_rag_experiment \
    --llm-batch-size 8 \
    --llm-dtype bfloat16 \
    --vector-top-k 5 \
    --log-level INFO \
    > output/full_rag_experiment/stdout.log 2>&1 &

# arg별 설명
# --dataset: 데이터셋 이름(현재 mediq)
# --dataset-path: 입력 jsonl 경로
# --variants: all 또는 only_llm,vector_rag,kg_no_cond,kg_with_cond
# --output-dir: 결과/요약/trace 저장 폴더
# --llm-batch-size: 엔티티 추출 배치 크기(GPU 메모리 기준 조정)
# --llm-dtype: bfloat16|float16|float32
# --vector-top-k: vector_rag에서 참조할 상위 근거 수
# --log-level: DEBUG|INFO|WARNING|ERROR

# 선택 arg 예시(필요 시 위 명령에 추가)
# --llm-model Codingchild/ILlama-8b-LoRA   # 모델 교체(iLlama)
# --max-samples 100                         # 테스트
# --start-index 0                           # 이어서 실행/샤딩
# --no-trace                                # trace.log 저장 비활성화
# --no-deterministic                        # 샘플링 추론