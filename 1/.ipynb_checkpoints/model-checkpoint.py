import os

os.environ[
    "HF_HOME"
] = "/opt/tritonserver/model_repository/nlp_models/hf_cache"
import json

import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils

# 추가
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, TextStreamer, pipeline
import time
import sys
from datetime import datetime

torch.cuda.empty_cache()

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        default_hf_model = "/models/gemma-2b/gemma-2b"
        default_max_gen_length = "128"
        # Check for user-specified model name in model config parameters
        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )
        # Check for user-specified max length in model config parameters
        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get( 
                "string_value", default_max_gen_length
            )
        )
        
        self.base_model = hf_model

        self.logger.log_info(f"Max sequence length: {self.max_output_length}")
        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")
        
        self.config_model()
        self.model_eval()
        self.model_compile()
        
        self.logger.log_info(f"Initialized...")
        
    def config_model(self):
        self.bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        
        self.tokenzier = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenzier.pad_token = self.tokenzier.eos_token
        self.logger.log_info("tokenizer loaded")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            #quantization_config=self.bnb_config,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.enable_input_require_grads()

        self.logger.log_info(f"base.model.device : {self.model.device}")
        self.logger.log_info("base model loaded")
        
    def model_eval(self):
        self.logger.log_info(f"...model eval start")

        self.model.eval()
        self.model.config.use_cache = True

        self.logger.log_info(f"...model eval end")

    def model_compile(self):
        self.logger.log_info(f"...model compile start")

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
            self.logger.log_info(f"...model compiled!")

        self.logger.log_info(f"...model compile end")

    def execute(self, requests):
        self.logger.log_info("### inferenect start")
        
        temperature = 0.2
        top_p = 0.7
        instruction = ''
        max_new_tokens = 128
        stream_output = False

        responses = []
        answer = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input = input_tensor.as_numpy()[0].decode("utf-8")
            question = [input]

            self.logger.log_info("### Receive Time: {}\n".format(datetime.now()))
            self.logger.log_info("### Question : {}\n".format(input.encode('utf-8').strip()))
            self.logger.log_info(f"model device type is : {self.model.device}")

            start = time.time()
           
            generation_config = GenerationConfig(
                temperature=0.15,
                top_k=40,
                do_sample=True,
                eos_token_id=2,
                early_stopping=True,
                max_new_tokens=128
            )
            
            inputs = self.tokenzier(
                    f"### 질문: {input}\n\n### 답변:",
                    return_tensors='pt',
                    padding=True,
                    return_token_type_ids=False).to(device=self.model.device)
                    
            input_ids = inputs["input_ids"].to(device=self.model.device)
            attention_mask = inputs["attention_mask"].to(device=self.model.device)

            self.logger.log_info(f"inputs: {inputs}")

            gened = self.model.generate(
                generation_config=generation_config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=2
            )
            
            answer = self.tokenzier.decode(gened[0])
            end = time.time()

            self.logger.log_info(f"decoded_output answer : {answer}")
            self.logger.log_info(f"### generate elapsed time is {end - start}")

            output = self.post_process(answer)

            self.logger.log_info(f"output type is: {type(output)}")
            self.logger.log_info(f"### fianl output : {output}")

            output_tensor_0 = pb_utils.Tensor("text_output", np.array(output.encode('utf-8'), dtype=np.bytes_))

            self.logger.log_info(f"type : {type(output_tensor_0)}, value: {output_tensor_0}")

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor_0])
            responses.append(response)

        self.logger.log_info("### inferenect end")

        return responses

    def post_process(self, text):
        return str(text.split("### 답변:")[1].split("### 질문:")[0].strip())

    def finalize(self):
        print("Cleaning up...")