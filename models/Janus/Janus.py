import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

class Janus:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda").eval()
        self.processor : VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens


    def process_messages(self,messages):
        new_messages = []
        prompt = ""
        if "system" in messages:
            prompt = messages["system"]
        prompt = prompt + messages["prompt"]
        if "image" in messages:
            new_messages = [
              {
                  "role": "<|User|>",
                  "content": f"<image_placeholder>\n{prompt}",
                  "images": [messages["image"]],
              },
              {"role": "<|Assistant|>", "content": ""},
          ]
        elif "images" in messages:
            images = messages["images"]
            content = ""
            for i,image in enumerate(images):
                content = content + f"\n<image_{i+1}: <image_placeholder>" + "\n"
            content = content + prompt
            new_messages = [
              {
                  "role": "<|User|>",
                  "content": content,
                  "images": messages["images"],
              },
              {"role": "<|Assistant|>", "content": ""},
          ]
        else:
            new_messages = [
              {
                  "role": "<|User|>",
                  "content": f"{prompt}"
              },
              {"role": "<|Assistant|>", "content": ""},
          ]
        conversation = new_messages
        
        # Check if images are already PIL Image objects or file paths
        # If they are PIL Image objects, use them directly; otherwise load them
        has_images = any("images" in msg for msg in conversation)
        if has_images:
            # Check the type of the first image in the first message with images
            for msg in conversation:
                if "images" in msg and len(msg["images"]) > 0:
                    first_image = msg["images"][0]
                    if isinstance(first_image, Image.Image):
                        # Images are already PIL Image objects
                        pil_images = []
                        for msg in conversation:
                            if "images" in msg:
                                for img in msg["images"]:
                                    if isinstance(img, Image.Image):
                                        pil_images.append(img.convert("RGB"))
                                    else:
                                        # Fallback: treat as file path
                                        pil_images.append(Image.open(img).convert("RGB"))
                    else:
                        # Images are file paths or base64 strings, use load_pil_images
                        pil_images = load_pil_images(conversation)
                    break
        else:
            pil_images = []
        
        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.llm.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.llm.prepare_inputs_embeds(**prepare_inputs)

        return {"inputs_embeds": inputs_embeds, "prepare_inputs": prepare_inputs}


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        inputs_embeds = inputs["inputs_embeds"]
        prepare_inputs = inputs["prepare_inputs"]
        do_sample = False if self.temperature == 0 else True
        outputs = self.llm.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature = self.temperature,
            top_p = self.top_p,
            repetition_penalty = self.repetition_penalty,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages)
            res.append(result)
        return res
