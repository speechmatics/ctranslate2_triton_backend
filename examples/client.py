import asyncio
import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from grpc import ChannelConnectivity
from transformers import AutoTokenizer
import logging
import numpy as np
import sys

async def main():
    MODEL_NAME = "opus-mt-en-de"
    URL = "127.0.0.1:8001"
    client = tritonclient.grpc.aio.InferenceServerClient(URL)
    
    en_text = sys.stdin.readline()
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/" + MODEL_NAME)
    
    input_ids = tokenizer(en_text, return_attention_mask=False, return_tensors="np").input_ids.astype(np.int32)
    logging.info(f"Tokenised input: {input_ids}")
    
    if client._channel.get_state() == ChannelConnectivity.SHUTDOWN:
        return
    
    inputs = [
        tritonclient.grpc.aio.InferInput("INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    outputs = [tritonclient.grpc.aio.InferRequestedOutput("OUTPUT_IDS")]

    res = await client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    out_tokens = res.as_numpy("OUTPUT_IDS")
    logging.info(f"Returned tokens: {out_tokens}")
    translated_text = tokenizer.batch_decode(out_tokens)
    print(translated_text)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
