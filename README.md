# CTranslate2 Backend for Triton Inference Server

This is a [backend](https://github.com/triton-inference-server/backend) based on [CTranslate2](https://github.com/OpenNMT/CTranslate2) for NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), which can be used to deploy translation and language models supported by CTranslate2 on Triton with both CPU and GPU capabilities.

It supports ragged and dynamic batching and setting of (a subset of) CTranslate decoding parameters in the model config.

## Building

Make sure to have [cmake](https://cmake.org) installed on your system.

1. Build and install CTranslate2: [https://opennmt.net/CTranslate2/installation.html#compile-the-c-library](https://opennmt.net/CTranslate2/installation.html#compile-the-c-library)
2. Build the backend 
```bash
mkdir build && cd build
export BACKEND_INSTALL_DIR=$(pwd)/install
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRITON_ENABLE_GPU=1 -DCMAKE_INSTALL_PREFIX=$BACKEND_INSTALL_DIR
make install
```

This builds the backend into `$BACKEND_INSTALL_DIR/backends/ctranslate2`.

## Setting up the backend

First install the pip package to convert models: `pip install ctranslate2`. Then create a model repository, which consists of a configuration (config.pbtxt) and the converted model.  

For example for the [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) HuggingFace transformer model, create a new directory e.g. `mkdir $MODEL_DIR/opus-mt-en-de`.
The model needs to be moved into a directory called `model` that is nested in a folder specifying a numerical version of the model:

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir 1/model
```

The minimum configuration for the model and backend is the following, you can see an example configs in [examples/model_repo](examples/model_repo):

```protobuf
backend: "ctranslate2" # must be ctranslate2
name: "opus-mt-en-de" # must be the same as the model name
max_batch_size: 128 # can be optimised based on available GPU memory
input [
  {
    name: "INPUT_IDS"
    data_type: TYPE_INT32
    dims: [ -1 ]
    allow_ragged_batch: true # needed for dynamic batching
  }
]
output [
  {
    name: "OUTPUT_IDS"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

instance_group [{ kind: KIND_GPU, count: 1 }] # use KIND_CPU for CPU inference
dynamic_batching {
  max_queue_delay_microseconds: 5000 # can be tuned based on latency requirements
}
```

Start the tritonserver with the ctranslate backend and model repository: 
```bash
tritonserver --backend-directory $BACKEND_INSTALL_DIR/backends --model-repository $MODEL_DIR
```

The backend is set up to use ragged, [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher).
This means you should send each input in it's own request and Triton will take care of batching, using the `dynamic_batching` and `max_batch_size` configuration to build appropriate batches, 
for best performance on GPUs. 

### Providing a target prefix

There are models that require a special prefix token for the decoder. For example the M2M models need a token that specifies the target language, or sometimes it might be useful to start the translation with a specific prefix. An example config can be found in the [facebook M2M config.pbtxt](examples/model_repo/facebook_m2m100_1.2B/config.pbtxt).


## Sending requests

The backend expects token IDs as input and output, which need to be INT32 or bigger data-types. This might include beginning-of-sentence and end-of-sentence token, depending on the model. 
No padding tokens need to be added though, the backend is taking care of padding and batching.

You can use the offical Triton clients to make requests, both HTTP and gRPC protocols are supported. We provide an example for a Python client [here](examples/client.py).

You can try the working translation by running
```bash
echo "How is your day going?" | python3 examples/client.py
```

In case you want to use a special prefix for decoding, the request also needs to have the input `TARGET_PREFIX` set, which could look like this in Python:

```python
inputs.append(
    tritonclient.grpc.InferInput("TARGET_PREFIX", prefix_ids.shape, np_to_triton_dtype(prefix_ids.dtype))
)
inputs[1].set_data_from_numpy(prefix_ids)
```

## Configuration

The backend exposes a few parameters for customisation, currently that is a subset of decoding options in the CTranslate2 C++ interface and some special ones that are useful for limiting inference compute requirements:

* `compute_type` overrides the numerical precision used for the majority of the Transformer computation, relates to [quantization types](https://opennmt.net/CTranslate2/quantization.html#quantization) in model conversion.
* `max_decoding_length_multiple` can be used to limit the number of output tokens as a multiple of input tokens. E.g if the longest input sequence is 10 tokens and `max_decoding_length_multiple: "2"` decoding is limitted to 20 tokens.

Decoding parameters passed through to CTranslate, more to be added:

* `beam_size`
* `repetition_penalty`

The parameters can be set like this in the config.pbtxt:

```protobuf
parameters [
  {
    key: "compute_type"
    value {
      string_value: "float16" # optional, can be used to force a compute type
    }
  },
  {
    key: "max_decoding_length_multiple"
    value {
      string_value: "2" 
    }
  },
    {
    key: "beam_size"
    value {
      string_value: "4" 
    }
  },
    {
    key: "repetition_penalty"
    value {
      string_value: "1.5"
    }
  }
]
```