import asyncio
import httpx


async def llm_openai_api(
    messages,
    model_name: str,
    ip: str = "localhost",
    host: str = "8080",
    temperature: float = 0.1,
    max_tokens: int = 128,
    top_p: float = None,
    n: int = 1,
    **kwargs,
):
    openai_api_base = f"http://{ip}:{host}/v1"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
    }
    
    if kwargs:
        payload.update(kwargs)
    
    # print(f"--- Sending request to: {openai_api_base}/chat/completions ---")
    # print(f"Payload: {json.dumps(payload, indent=2)}")
    # print("----------------------------------------------------------")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        try:
            resp = await client.post(
                f"{openai_api_base}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()  # 如果请求失败（非2xx状态码），则抛出异常
            response_data = resp.json()
            return [choice["message"]["content"] for choice in response_data["choices"]]
        
        except httpx.ConnectError as e:
            print(f"❌ Connection Error: Failed to connect to {ip}:{host}. Is the server running and accessible?")
            raise e
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
            raise e

async def main_test():
    SERVER_IP = "localhost"
    SERVER_PORT = "8080"
    MODEL_NAME = "/root/bayes-gpfs-c9f955b46c074f048c960ec71693fa58/niechang/data1/huggingface_cache/Qwen3-30B-A3B-Instruct-2507"
    
    test_messages = [{"role": "user", "content": "写一个关于未来城市的简短故事开头。"}]

    print("\n===== TEST 1: Calling WITH custom parameter `enable_thinking=False` =====")
    try:
        results = await llm_openai_api(
            messages=test_messages,
            model_name=MODEL_NAME,
            ip=SERVER_IP,
            host=SERVER_PORT
        )
        print("\n✅ Success! Response:")
        print(results[0])
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main_test())


'''
MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    --model /root/bayes-gpfs-c9f955b46c074f048c960ec71693fa58/niechang/data1/huggingface_cache/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
'''

