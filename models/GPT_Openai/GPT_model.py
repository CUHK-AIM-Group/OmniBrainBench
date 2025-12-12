import base64
import os
import time
from io import BytesIO
from typing import List, Dict, Any, Union
from openai import OpenAI
import threading
import weakref

key = os.environ.get("openai_api_key")
baseurl = os.environ.get("openai_base_url")

class GPT_model:
    _instances = weakref.WeakSet()  # 跟踪所有实例
    _cleanup_lock = threading.Lock()
    
    def __init__(self, model_path: str, args):
        """
        Args:
            model_path (str): OpenAI model name, e.g., "gpt-4o"
            args: An object with attributes: temperature, top_p, repetition_penalty, max_new_tokens
        """
        self.model_name = model_path
        
        # Parameters
        self.temperature = getattr(args, 'temperature', 0.05)
        
        # 如果是gpt系列的模型就不设置top_p
        if "gpt" in model_path.lower():
            self.top_p = None  # GPT系列模型不使用top_p
        else:
            self.top_p = getattr(args, 'top_p', 0.9)  # 非GPT模型使用top_p
        
        # Timeout and retry parameters
        self.timeout = getattr(args, 'timeout', 300)  # 默认300秒超时
        self.max_retries = getattr(args, 'max_retries', 3)  # 最大重试次数
        self.retry_delay = getattr(args, 'retry_delay', 5)  # 重试间隔秒数
        
        # Initialize OpenAI client with timeout
        try:
            self.client = OpenAI(
                api_key=key,
                base_url=baseurl,
                timeout=self.timeout,  # 设置请求超时时间
            )
            GPT_model._instances.add(self)  # 添加到实例跟踪
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client: {e}")
            raise
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        with GPT_model._cleanup_lock:
            try:
                if hasattr(self, 'client') and self.client:
                    # 快速关闭client连接，设置短超时
                    if hasattr(self.client, '_client') and hasattr(self.client._client, 'close'):
                        # 对于httpx客户端，快速关闭
                        try:
                            self.client._client.close()
                        except Exception as e:
                            pass  # 忽略关闭时的错误
                    
                    # 置空客户端引用
                    self.client = None
            except Exception as e:
                print(f"[WARNING] Error during GPT_model cleanup: {e}")
    
    @classmethod
    def cleanup_all_instances(cls):
        """清理所有实例"""
        with cls._cleanup_lock:
            instances = list(cls._instances)
            for instance in instances:
                try:
                    instance.cleanup()
                except Exception as e:
                    print(f"[WARNING] Error cleaning up GPT_model instance: {e}")
            # 清空实例集合
            cls._instances.clear()
    
    def __del__(self):
        self.cleanup()

    def _encode_image(self, image: Union[str, bytes]) -> str:
        """
        Encodes an image from file path or bytes into base64 string.
        Supports PIL Image, file path, or bytes.
        """
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        else:
            # Assume it's a PIL.Image
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def process_messages(self, messages: Dict[str, Any]) -> List[Dict]:
        """
        Converts custom messages format to OpenAI chat format.
        Handles system, prompt, image(s).
        Returns: messages list for OpenAI API.
        """
        openai_messages = []

        # Add system message if present
        if "system" in messages:
            openai_messages.append({"role": "system", "content": messages["system"]})

        # Build user content
        content = []

        if "image" in messages:
            image = messages["image"]
            base64_image = self._encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
            content.append({"type": "text", "text": messages["prompt"]})

        elif "images" in messages:
            for i, image in enumerate(messages["images"]):
                base64_image = self._encode_image(image)
                content.append({"type": "text", "text": f"<image_{i+1}>:"})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            content.append({"type": "text", "text": messages["prompt"]})
        else:
            content.append({"type": "text", "text": messages["prompt"]})

        openai_messages.append({"role": "user", "content": content})
        return openai_messages

    def generate_output(self, messages: Dict[str, Any]) -> str:
        """
        Generate a single response using GPT multimodal API with retry mechanism.
        """
        openai_messages = self.process_messages(messages)

        # Prepare generation kwargs
        gen_kwargs = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            # 不设置max_tokens,让模型完整输出
        }
        
        # 只有非GPT系列模型才添加top_p参数
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        
        for attempt in range(self.max_retries + 1):
            try:
                # 检查是否需要提前退出
                try:
                    import threading
                    # 检查全局shutdown标志
                    if hasattr(__import__('eval'), 'shutdown_flag'):
                        shutdown_flag = __import__('eval').shutdown_flag
                        if shutdown_flag.is_set():
                            print("[INFO] Shutdown requested, stopping API request")
                            return "[INTERRUPTED] Request cancelled due to shutdown"
                except:
                    pass  # 如果无法访问shutdown_flag，继续执行
                
                print(f"[DEBUG] Sending request to model: {self.model_name} (attempt {attempt + 1}/{self.max_retries + 1})")
                print(f"[DEBUG] Timeout set to: {self.timeout} seconds")
                
                start_time = time.time()
                
                # 使用设置的超时时间(默认300秒)
                response = self.client.chat.completions.create(timeout=self.timeout, **gen_kwargs)
                
                end_time = time.time()
                
                result = response.choices[0].message.content
                print(f"[DEBUG] Response received in {end_time - start_time:.2f} seconds")
                print(f"[DEBUG] Response content (full): {result}")
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                print(f"[ERROR] Attempt {attempt + 1} failed with {error_type}: {str(e)}")
                
                # 检查是否是超时错误
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"[WARNING] Request timed out after {self.timeout} seconds")
                
                # 如果还有重试机会，等待后重试
                if attempt < self.max_retries:
                    print(f"[INFO] Waiting {self.retry_delay} seconds before retry...")
                    # 使用短暂的睡眠，方便中断
                    for _ in range(self.retry_delay):
                        time.sleep(1)
                        # 检查是否需要提前退出
                        try:
                            if hasattr(__import__('eval'), 'shutdown_flag'):
                                shutdown_flag = __import__('eval').shutdown_flag
                                if shutdown_flag.is_set():
                                    print("[INFO] Shutdown requested during retry wait")
                                    return "[INTERRUPTED] Request cancelled during retry"
                        except:
                            pass
                else:
                    # 最后一次尝试失败
                    print(f"[ERROR] All {self.max_retries + 1} attempts failed. Giving up.")
                    raise e

    def generate_outputs(self, messages_list: List[Dict[str, Any]]) -> List[str]:
        """
        Generate responses for a list of message inputs.
        """
        results = []
        for i, messages in enumerate(messages_list):
            try:
                print(f"[INFO] Processing message {i + 1}/{len(messages_list)}")
                result = self.generate_output(messages)
            except Exception as e:
                error_type = type(e).__name__
                result = f"[ERROR] {error_type}: {str(e)}"
                print(f"[ERROR] Failed to process message {i + 1}: {result}")
            finally:
                # 确保每次处理后都进行一些清理
                import gc
                gc.collect()
            results.append(result)
        return results