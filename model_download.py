from huggingface_hub import snapshot_download

# 下载模型到指定目录
local_model_path = snapshot_download(repo_id="stabilityai/sdxl-turbo", local_dir="./sdxl-turbo")
print(f"Model downloaded to {local_model_path}")