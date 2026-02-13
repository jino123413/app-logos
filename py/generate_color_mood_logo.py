"""
ComfyUI + Flux.1 GGUF를 사용한 컬러 무드 앱 로고 생성
- 모델: flux1-schnell-Q4_K_S.gguf
- 프롬프트 템플릿: workflow.md 기본 브랜딩
- 출력: 600x600px → app-logos/color-mood.png
"""

import json
import urllib.request
import urllib.error
import time
import io
import os
from PIL import Image

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_PATH = r"C:\Users\USER-PC\Desktop\appintoss-project\app-logos\color-mood.png"

# Flux.1 schnell GGUF workflow
PROMPT = {
    "3": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
            "unet_name": "flux1-schnell-Q4_K_S.gguf"
        }
    },
    "4": {
        "class_type": "DualCLIPLoaderGGUF",
        "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
            "type": "flux"
        }
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": (
                "A minimal flat app icon for a daily color mood diary app. "
                "A minimal flat app icon for a color mood diary app. "
                "Soft violet purple solid background. "
                "Center features a rainbow color wheel circle with red orange yellow green blue purple segments, "
                "with a small white paint droplet below it. "
                "Clean, modern, minimal flat vector style. "
                "No text. App icon design. 512x512px."
            ),
            "clip": ["4", 0]
        }
    },
    "6": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        }
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["3", 0],
            "positive": ["5", 0],
            "negative": ["10", 0],
            "latent_image": ["6", 0],
            "seed": 777,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0
        }
    },
    "8": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "ae.safetensors"
        }
    },
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["8", 0]
        }
    },
    "10": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["4", 0]
        }
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["9", 0],
            "filename_prefix": "color-mood-logo"
        }
    }
}


def queue_prompt(prompt):
    """ComfyUI API에 프롬프트 전송"""
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/api/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def get_history(prompt_id):
    """프롬프트 실행 기록 조회"""
    req = urllib.request.Request(f"{COMFYUI_URL}/api/history/{prompt_id}")
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def get_image(filename, subfolder, folder_type):
    """ComfyUI 출력 이미지 다운로드"""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    req = urllib.request.Request(f"{COMFYUI_URL}/api/view?{params}")
    resp = urllib.request.urlopen(req)
    return resp.read()


def main():
    print("1. ComfyUI에 Flux.1 프롬프트 전송...")
    result = queue_prompt(PROMPT)
    prompt_id = result["prompt_id"]
    print(f"   prompt_id: {prompt_id}")

    print("2. 이미지 생성 대기...")
    while True:
        time.sleep(2)
        history = get_history(prompt_id)
        if prompt_id in history:
            if history[prompt_id].get("status", {}).get("completed", False) or \
               "outputs" in history[prompt_id]:
                break
        # Check queue status
        try:
            req = urllib.request.Request(f"{COMFYUI_URL}/api/prompt")
            resp = urllib.request.urlopen(req)
            queue_info = json.loads(resp.read())
            remaining = queue_info.get("exec_info", {}).get("queue_remaining", "?")
            print(f"   대기 중... (queue_remaining: {remaining})")
        except:
            pass

    print("3. 생성된 이미지 다운로드...")
    history = get_history(prompt_id)
    outputs = history[prompt_id]["outputs"]

    # SaveImage 노드 (11번)의 출력 가져오기
    image_info = outputs["11"]["images"][0]
    img_data = get_image(
        image_info["filename"],
        image_info.get("subfolder", ""),
        image_info.get("type", "output")
    )

    print("4. 600x600으로 리사이즈...")
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((600, 600), Image.LANCZOS)
    img.save(OUTPUT_PATH, "PNG")

    print(f"완료: {OUTPUT_PATH}")
    print(f"크기: {img.size}, 모드: {img.mode}")


if __name__ == "__main__":
    main()
