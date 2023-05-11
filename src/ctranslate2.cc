// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonbackend.h"

#include "ctranslate2/models/model.h"
#include "ctranslate2/models/sequence_to_sequence.h"
#include "triton/core/tritonserver.h"

namespace triton {
namespace backend {
namespace ctranslate2 {

TRITONSERVER_Error *
ReadParameter(const triton::common::TritonJson::Value &params,
              const std::string &key, std::string *param) {
  triton::common::TritonJson::Value value;
  RETURN_ERROR_IF_FALSE(
      const_cast<triton::common::TritonJson::Value &>(params).Find(key.c_str(),
                                                                   &value),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration is missing the parameter ") + key);
  RETURN_IF_ERROR(value.MemberAsString("string_value", param));
  return nullptr; // success
}

TRITONSERVER_Error *
ReadParameter(const triton::common::TritonJson::Value &params,
              const std::string &key, int *param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = std::stoi(tmp);
  return nullptr; // success
}

TRITONSERVER_Error *
ReadParameter(const triton::common::TritonJson::Value &params,
              const std::string &key, size_t *param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = static_cast<size_t>(std::stoi(tmp));
  return nullptr; // success
}

TRITONSERVER_Error *
ReadParameter(const triton::common::TritonJson::Value &params,
              const std::string &key, float *param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = std::stof(tmp);
  return nullptr; // success
}

class ModelState : public BackendModel {
public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model,
                                    ModelState **state);
  virtual ~ModelState() = default;

  ModelState(TRITONBACKEND_Model *triton_model)
      : BackendModel(triton_model, false) {
    THROW_IF_BACKEND_MODEL_ERROR(ValidateModel());
    THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
  }

  TRITONSERVER_Error *ValidateModelConfig() {
    // If verbose logging is enabled, dump the model's configuration as
    // JSON into the console output.
    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
      common::TritonJson::WriteBuffer buffer;
      RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("model configuration:\n") + buffer.Contents()).c_str());
    }

    // ModelConfig is the model configuration as a TritonJson
    // object. Use the TritonJson utilities to parse the JSON and
    // determine if the configuration is supported by this backend.
    common::TritonJson::Value inputs, outputs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

    // The model must have exactly 1 input and 1 output.
    RETURN_ERROR_IF_FALSE(
        inputs.ArraySize() == 1 || inputs.ArraySize() == 2,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model configuration must have 1 or 2 inputs"));
    RETURN_ERROR_IF_FALSE(
        outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model configuration must have 1 output"));

    common::TritonJson::Value input, output;
    RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

    // Record the input and output name in the model state.
    const char *input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
    input_name_ = std::string(input_name);

    if (inputs.ArraySize() == 2) {
      RETURN_IF_ERROR(inputs.IndexAsObject(1, &input));
      const char *target_prefix_input_name;
      size_t target_prefix_input_name_len;
      RETURN_IF_ERROR(input.MemberAsString("name", &target_prefix_input_name,
                                           &target_prefix_input_name_len));
      target_prefix_input_name_ = std::string(target_prefix_input_name);
    }

    const char *output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    output_name_ = std::string(output_name);

    std::string io_dtype;
    RETURN_IF_ERROR(output.MemberAsString("data_type", &io_dtype));
    output_type_ =
        triton::backend::ModelConfigDataTypeToTritonServerDataType(io_dtype);

    triton::common::TritonJson::Value params;
    bool has_params = ModelConfig().Find("parameters", &params);
    if (has_params) {
      if (params.Find("compute_type")) {
        std::string compute_type_str;
        RETURN_IF_ERROR(
            ReadParameter(params, "compute_type", &compute_type_str));
        compute_type_ = ::ctranslate2::str_to_compute_type(compute_type_str);

        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("Running inference in compute type: ") +
                     compute_type_str)
                        .c_str());
      }
      if (params.Find("max_decoding_length_multiple")) {
        size_t max_decode_length_multiple;
        RETURN_IF_ERROR(ReadParameter(params, "max_decoding_length_multiple",
                                      &max_decode_length_multiple));
        max_decode_length_multiple_ = max_decode_length_multiple;
      }
      if (params.Find("beam_size")) {
        RETURN_IF_ERROR(ReadParameter(
            params, "beam_size", &(default_translation_options_.beam_size)));
      }
      if (params.Find("repetition_penalty")) {
        RETURN_IF_ERROR(ReadParameter(params, "repetition_penalty",
                                      &(default_translation_options_.repetition_penalty)));
      }
    }
    return nullptr;
  }

  const std::string &InputTensorName() const { return input_name_; }
  const std::optional<std::string> &TargetPrefixInputName() const {
    return target_prefix_input_name_;
  }
  const std::string &OutputTensorName() const { return output_name_; }
  TRITONSERVER_DataType OutputDataType() const { return output_type_; }
  ::ctranslate2::TranslationOptions DefaultTranslationOptions() const {
    return default_translation_options_;
  }
  const std::optional<size_t> &MaxDecodeLengthMultiple() const {
    return max_decode_length_multiple_;
  }

  TRITONSERVER_Error *
  LoadModel(const ::ctranslate2::Device device, std::int32_t device_index,
            std::shared_ptr<const ::ctranslate2::models::Model> *ct_model) {
    std::shared_ptr<const ::ctranslate2::models::Model> model;
    std::pair<::ctranslate2::Device, std::int32_t> device_pair =
        std::make_pair(device, device_index);
    auto mit = models_.find(device_pair);
    if (mit != models_.end()) {
      model = mit->second;
    } else {
      if (!models_.empty()) {
        model = models_.begin()->second->copy_to(device, device_index);
      } else {
        model = ::ctranslate2::models::Model::load(*model_reader_, device,
                                                   device_index, compute_type_);
      }
      models_.emplace(device_pair, model);
    }
    *ct_model = model;

    return nullptr;
  }

private:
  // TRITONBACKEND_Model *triton_model_;
  triton::common::TritonJson::Value model_config_;
  std::string input_name_;
  std::optional<std::string> target_prefix_input_name_;
  std::string output_name_;
  TRITONSERVER_DataType output_type_;
  ::ctranslate2::ComputeType compute_type_ =
      ::ctranslate2::ComputeType::DEFAULT;
  ::ctranslate2::TranslationOptions default_translation_options_;
  std::optional<size_t> max_decode_length_multiple_;
  std::string model_path_;
  std::shared_ptr<::ctranslate2::models::ModelReader> model_reader_;
  std::map<std::pair<::ctranslate2::Device, std::int32_t>,
           std::shared_ptr<const ::ctranslate2::models::Model>>
      models_;

  TRITONSERVER_Error *ValidateModel() {
    std::string artifact_filename;
    THROW_IF_BACKEND_MODEL_ERROR(ModelConfig().MemberAsString(
        "default_model_filename", &artifact_filename));
    // if default_model_filename not set default to "model"
    if (artifact_filename.empty()) {
      artifact_filename = "model";
    }

    model_path_ = JoinPath(
        {RepositoryPath(), std::to_string(Version()), artifact_filename});
    model_reader_ =
        std::make_shared<::ctranslate2::models::ModelFileReader>(model_path_);
    auto contains_model = ::ctranslate2::models::contains_model(model_path_);

    RETURN_ERROR_IF_FALSE(contains_model, TRITONSERVER_ERROR_UNAVAILABLE,
                          std::string("unable to find '") + model_path_ +
                              "' for model instance '" + Name() + "'");

    return nullptr;
  }
};

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model,
                                       ModelState **state) {

  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr; // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model) {
  const char *cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState *model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
      model, reinterpret_cast<void *>(model_state)));

  return nullptr; // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vstate);
  delete model_state;

  return nullptr; // success
}

} // extern "C"

template <typename T>
TRITONSERVER_Error *
ToIdVectorTyped(const char *buffer, const size_t element_count,
                std::vector<size_t> *ids, const size_t start_idx = 0) {
  const T *vals = reinterpret_cast<const T *>(buffer);
  *ids =
      std::vector<size_t>(vals + start_idx, vals + start_idx + element_count);
  return nullptr;
}

TRITONSERVER_Error *ToIdVector(const char *buffer,
                               TRITONSERVER_DataType datatype,
                               std::vector<size_t> *ids, const size_t start_idx,
                               const size_t element_cnt) {

  switch (datatype) {
  case TRITONSERVER_TYPE_UINT8:
    return ToIdVectorTyped<uint8_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_UINT16:
    return ToIdVectorTyped<uint16_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_UINT32:
    return ToIdVectorTyped<uint32_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_UINT64:
    return ToIdVectorTyped<uint64_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_INT8:
    return ToIdVectorTyped<int8_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_INT16:
    return ToIdVectorTyped<int16_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_INT32:
    return ToIdVectorTyped<int32_t>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_INT64:
    return ToIdVectorTyped<int64_t>(buffer, element_cnt, ids, start_idx);

  case TRITONSERVER_TYPE_FP32:
    return ToIdVectorTyped<float>(buffer, element_cnt, ids, start_idx);
  case TRITONSERVER_TYPE_FP64:
    return ToIdVectorTyped<double>(buffer, element_cnt, ids, start_idx);
  default:
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(std::string("class result not available for output due to "
                                "unsupported type '") +
                    std::string(TRITONSERVER_DataTypeString(datatype)) + "'")
            .c_str());
  }
}

template <typename T>
void ConvertToRawPointer(const std::vector<std::size_t> &out_tokens,
                         void *out_buffer) {
  T *buffer = static_cast<T *>(out_buffer);
  for (auto &token : out_tokens) {
    auto idx = &token - &out_tokens[0];
    buffer[idx] = static_cast<T>(token);
  }
}

size_t TritonTypeSize(TRITONSERVER_DataType datatype) {
  switch (datatype) {
  case TRITONSERVER_TYPE_UINT8:
    return sizeof(std::uint8_t);
  case TRITONSERVER_TYPE_UINT16:
    return sizeof(std::uint16_t);
  case TRITONSERVER_TYPE_UINT32:
    return sizeof(std::uint32_t);
  case TRITONSERVER_TYPE_UINT64:
    return sizeof(std::uint64_t);
  case TRITONSERVER_TYPE_INT8:
    return sizeof(std::int8_t);
  case TRITONSERVER_TYPE_INT16:
    return sizeof(std::int16_t);
  case TRITONSERVER_TYPE_INT32:
    return sizeof(std::int32_t);
  case TRITONSERVER_TYPE_INT64:
    return sizeof(std::int64_t);
    break;

  case TRITONSERVER_TYPE_FP32:
    return sizeof(std::float_t);
  case TRITONSERVER_TYPE_FP64:
    return sizeof(std::double_t);
  default:
    throw std::invalid_argument(std::string("Can't determine type size for ") +
                                std::to_string(TRITONSERVER_TYPE_FP64));
  }
}

TRITONSERVER_Error *ToOutBuffer(const std::vector<std::size_t> &out_tokens,
                                TRITONSERVER_DataType datatype,
                                void *out_buffer) {
  switch (datatype) {
  case TRITONSERVER_TYPE_UINT8:
    ConvertToRawPointer<std::uint8_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_UINT16:
    ConvertToRawPointer<std::uint16_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_UINT32:
    ConvertToRawPointer<std::uint32_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_UINT64:
    ConvertToRawPointer<std::uint64_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_INT8:
    ConvertToRawPointer<std::int8_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_INT16:
    ConvertToRawPointer<std::int16_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_INT32:
    ConvertToRawPointer<std::int32_t>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_INT64:
    ConvertToRawPointer<std::int64_t>(out_tokens, out_buffer);
    break;

  case TRITONSERVER_TYPE_FP32:
    ConvertToRawPointer<float>(out_tokens, out_buffer);
    break;
  case TRITONSERVER_TYPE_FP64:
    ConvertToRawPointer<double>(out_tokens, out_buffer);
    break;
  default:
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(std::string("class result not available for output due to "
                                "unsupported type '") +
                    std::string(TRITONSERVER_DataTypeString(datatype)) + "'")
            .c_str());
  }

  return nullptr;
}

std::string
TranslationOptionsToString(const ::ctranslate2::TranslationOptions &options) {
  std::stringstream ss;

  ss << "TranslationOptions("
     << "beam_size=" << options.beam_size << ", "
     << "patience=" << options.patience << ", "
     << "length_penalty=" << options.length_penalty << ", "
     << "coverage_penalty=" << options.coverage_penalty << ", "
     << "repetition_penalty=" << options.repetition_penalty << ", "
     << "no_repeat_ngram_size=" << options.no_repeat_ngram_size << ", "
     << "disable_unk=" << options.disable_unk << ", "
     << "size(suppress_sequences)=" << options.suppress_sequences.size() << ", "
     << "prefix_bias_beta=" << options.prefix_bias_beta << ", ";

  if (std::holds_alternative<std::string>(options.end_token)) {
    ss << "end_token=\"" << std::get<std::string>(options.end_token) << "\", ";
  } else if (std::holds_alternative<std::vector<std::string>>(
                 options.end_token)) {
    for (auto &end_token :
         std::get<std::vector<std::string>>(options.end_token)) {
      ss << "end_token[]=" << end_token << " ";
    }
    ss << ",";
  } else if (std::holds_alternative<std::vector<size_t>>(options.end_token)) {
    for (auto &end_token :
         std::get<std::vector<std::string>>(options.end_token)) {
      ss << "end_token[]=" << end_token << " ";
    }
    ss << ",";
  }

  ss << "max_input_length=" << options.max_input_length << ", "
     << "max_decoding_length=" << options.max_decoding_length << ", "
     << "min_decoding_length=" << options.min_decoding_length << ", "
     << "sampling_topk=" << options.sampling_topk << ", "
     << "sampling_temperature=" << options.sampling_temperature << ", "
     << "use_vmap=" << options.use_vmap << ", "
     << "num_hypotheses=" << options.num_hypotheses << ", "
     << "return_scores=" << options.return_scores << ", "
     << "return_attention=" << options.return_attention << ", "
     << "return_alternatives=" << options.return_alternatives << ", "
     << "min_alternative_expansion_prob="
     << options.min_alternative_expansion_prob << ", "
     << "replace_unknowns=" << options.replace_unknowns << ")";
  return ss.str();
}

TRITONSERVER_Error *InputBufferToRaggedTokens(
    size_t total_batch_size, TRITONBACKEND_Request **requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response *> *responses,
    BackendInputCollector *collector,
    std::vector<std::vector<size_t>> *ragged_tokens,
    size_t *max_sequence_length, const std::string &input_name,
    bool is_ragged_input = true, bool supports_batching = true) {
  std::vector<std::vector<size_t>> tokens;
  tokens.reserve(request_count);

  const char *input_buffer;
  size_t batchn_byte_size;
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  // TODO support data straight from GPU
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference = {
      {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

  RETURN_IF_ERROR(collector->ProcessTensor(
      input_name.c_str(), nullptr, 0, alloc_preference, &input_buffer,
      &batchn_byte_size, &memory_type, &memory_type_id));

  // bool is_ragged =
  //
  size_t max_seq_length = 0;
  if (is_ragged_input) {
    int64_t total_elements = 0;
    for (size_t request_idx = 0; request_idx < request_count; request_idx++) {
      TRITONBACKEND_Input *input;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &((*responses)[request_idx]),
          TRITONBACKEND_RequestInput(requests[request_idx], input_name.c_str(),
                                     &input));

      TRITONSERVER_DataType input_dt;
      const int64_t *input_shape;
      uint32_t input_dims_count;
      RETURN_IF_ERROR(
          TRITONBACKEND_InputProperties(input, nullptr, &input_dt, &input_shape,
                                        &input_dims_count, nullptr, nullptr));

      auto element_count = GetElementCount(input_shape, input_dims_count);
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                  (std::string("Element count for request ") +
                   std::to_string(request_idx) + std::string(": ") +
                   std::to_string(element_count))
                      .c_str());
      max_seq_length =
          std::max(max_seq_length, static_cast<size_t>(element_count));

      std::vector<size_t> ids;
      ToIdVector(input_buffer, input_dt, &ids, total_elements, element_count);
      total_elements += element_count;
      tokens.emplace_back(ids);
    }
  } else {
    // input type is the same for all
    TRITONBACKEND_Input *input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInput(requests[0], input_name.c_str(), &input));

    TRITONSERVER_DataType input_dt;
    const int64_t *input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(
        TRITONBACKEND_InputProperties(input, nullptr, &input_dt, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

    if (input_dims_count > 2) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("Inputs with more than two dimensions unsupported")
              .c_str());
    }

    std::vector<int64_t> batchn_shape =
        std::vector<int64_t>(input_shape, input_shape + input_dims_count);
    if (supports_batching) {
      batchn_shape[0] = total_batch_size;
    }

    for (size_t vector_idx = 0; vector_idx < total_batch_size; vector_idx++) {
      std::vector<size_t> ids;
      ToIdVector(input_buffer, input_dt, &ids, vector_idx * batchn_shape[1],
                 (vector_idx + 1) * batchn_shape[1]);
      tokens.emplace_back(ids);
    }
    max_seq_length = static_cast<size_t>(batchn_shape[1]);
  }

  *ragged_tokens = tokens;
  *max_sequence_length = max_seq_length;

  return nullptr;
}
/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
public:
  static TRITONSERVER_Error *
  Create(ModelState *model_state,
         TRITONBACKEND_ModelInstance *triton_model_instance,
         ModelInstanceState **state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState *StateForModel() const { return model_state_; }

  ModelInstanceState(ModelState *model_state,
                     TRITONBACKEND_ModelInstance *triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state) {
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
      device_ = ::ctranslate2::Device::CUDA;
#else
      throw triton::backend::BackendModelInstanceException(
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                                "Backend not built with GPU support"));
#endif
    } else {
      device_ = ::ctranslate2::Device::CPU;
    }
    model_state->LoadModel(device_, DeviceId(), &model_);
    supports_batching_ = model_state_->MaxBatchSize() > 0;
  }

  TRITONSERVER_Error *
  CreateInput(size_t total_batch_size, TRITONBACKEND_Request **requests,
              const uint32_t request_count,
              std::vector<TRITONBACKEND_Response *> *responses,
              BackendInputCollector *collector,
              const ::ctranslate2::Vocabulary &source_vocab,
              const ::ctranslate2::Vocabulary &target_vocab,
              std::vector<std::vector<std::string>> *input_tokens,
              std::vector<std::vector<std::string>> *input_target_prefix,
              size_t *max_sequence_length) {

    std::vector<std::vector<size_t>> input_token_ids;
    RETURN_IF_ERROR(InputBufferToRaggedTokens(
        total_batch_size, requests, request_count, responses, collector,
        &input_token_ids, max_sequence_length,
        StateForModel()->InputTensorName(),
        StateForModel()->IsInputRagged(StateForModel()->InputTensorName()),
        supports_batching_));
    *input_tokens = source_vocab.to_tokens(input_token_ids);
    if (StateForModel()->TargetPrefixInputName()) {
      std::vector<std::vector<size_t>> target_prefix_token_ids;
      size_t discard_seq_length;
      RETURN_IF_ERROR(InputBufferToRaggedTokens(
          total_batch_size, requests, request_count, responses, collector,
          &target_prefix_token_ids, &discard_seq_length,
          *(StateForModel()->TargetPrefixInputName()),
          StateForModel()->IsInputRagged(
              *(StateForModel()->TargetPrefixInputName())),
          supports_batching_));
      *input_target_prefix = target_vocab.to_tokens(target_prefix_token_ids);
    }
    return nullptr;
  }

  void ProcessRequests(TRITONBACKEND_Request **requests,
                       const uint32_t request_count) {

    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    std::vector<TRITONBACKEND_Response *> responses;
    responses.reserve(request_count);
    bool all_response_failed = false;

    for (size_t i = 0; i < request_count; i++) {
      TRITONBACKEND_Response *response;
      auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
      if (err == nullptr) {
        responses.emplace_back(response);
      } else {
        responses.emplace_back(nullptr);
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
        TRITONSERVER_ErrorDelete(err);
      }
    }

    const int max_batch_size = model_state_->MaxBatchSize();

    size_t total_batch_size = 0;
    for (size_t i = 0; i < request_count; i++) {
      if (max_batch_size > 0) {
        // Retrieve the batch size from one of the inputs, if the model
        // supports batching, the first dimension size is batch size.
        TRITONBACKEND_Input *input;
        TRITONSERVER_Error *err = TRITONBACKEND_RequestInput(
            requests[i], StateForModel()->InputTensorName().c_str(), &input);
        if (err == nullptr) {
          const int64_t *shape;
          err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape,
                                              nullptr, nullptr, nullptr);
          total_batch_size += shape[0];
        }
        if (err != nullptr) {
          RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count,
                                            all_response_failed, err);
        }
      } else {
        total_batch_size += 1;
      }
    }

    // If there are no valid payloads then no need to run the inference.
    if (total_batch_size == 0) {
      return;
    }

    // Make sure the maximum batch size is not exceeded. The
    // total_batch_size must be 1 for models that don't support batching
    // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
    // scheduler has done something badly wrong so fail and release all
    // requests.
    if (!all_response_failed) {
      if ((total_batch_size != 1) &&
          (total_batch_size > (size_t)max_batch_size)) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string("batch size " + std::to_string(total_batch_size) +
                            " for '" + Name() + "', max allowed is " +
                            std::to_string(max_batch_size))
                    .c_str()));
      }
    }

    const ::ctranslate2::models::SequenceToSequenceModel *seq2seq_model =
        dynamic_cast<const ::ctranslate2::models::SequenceToSequenceModel *>(
            model_.get());
    const auto source_vocab = seq2seq_model->get_source_vocabulary();
    const auto target_vocab = seq2seq_model->get_target_vocabulary();

    auto collector = std::make_unique<BackendInputCollector>(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(),
        model_state_->EnablePinnedInput() /* pinned_enabled */,
        nullptr /* stream*/);

    std::vector<std::vector<std::string>> inputs;
    std::vector<std::vector<std::string>> target_prefix;
    size_t max_input_seq_length;
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        CreateInput(total_batch_size, requests, request_count, &responses,
                    collector.get(), source_vocab, target_vocab, &inputs,
                    &target_prefix, &max_input_seq_length));

    std::unique_ptr<::ctranslate2::models::SequenceToSequenceReplica>
        seq2seq_replica = model_->as_sequence_to_sequence();

    // Finalize the collector. If 'true' is returned, 'input_buffer'
    // will not be valid until the backend synchronizes the CUDA
    // stream or event that was used when creating the collector. For
    // this backend, GPU is not supported and so no CUDA sync should
    // be needed; so if 'true' is returned simply log an error.
    const bool need_cuda_input_sync = collector->Finalize();
    if (need_cuda_input_sync) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  "backend: unexpected CUDA sync required by collector");
    }

    uint64_t compute_start_ns = 0;
    SET_TIMESTAMP(compute_start_ns);
    ::ctranslate2::TranslationOptions options =
        StateForModel()->DefaultTranslationOptions();
    auto max_decode_length_multiple =
        StateForModel()->MaxDecodeLengthMultiple();
    if (max_decode_length_multiple) {
      options.max_decoding_length =
          *max_decode_length_multiple * max_input_seq_length;
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                TranslationOptionsToString(options).c_str());
    std::vector<::ctranslate2::TranslationResult> translation_results =
        seq2seq_replica->translate(inputs, target_prefix, options);

    uint64_t compute_end_ns = 0;
    SET_TIMESTAMP(compute_end_ns);

    // This backend supports models that batch along the first dimension
    // and those that don't batch. For non-batch models the output shape
    // will be [ 4 ]. For batch models the output shape will be [ -1, 4
    // ] and the backend "responder" utility below will set the
    // appropriate batch dimension value for each response.
    std::vector<int64_t> output_batch_shape;
    bool supports_first_dim_batching;
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses, request_count,
                                      StateForModel()->SupportsFirstDimBatching(
                                          &supports_first_dim_batching));
    size_t idx = 0;
    for (auto &translation : translation_results) {

      std::vector<std::string> out_tokens = translation.output();
      // only output best hypotheses
      std::vector<size_t> out_ids = target_vocab.to_ids({out_tokens})[0];

      TRITONBACKEND_Output *response_output;
      std::vector<std::int64_t> out_shape = {(std::int64_t)out_ids.size()};
      if (supports_first_dim_batching) {
        out_shape.insert(out_shape.begin(), -1);
      }

      RESPOND_AND_SET_NULL_IF_ERROR(
          &responses[idx], TRITONBACKEND_ResponseOutput(
                               responses[idx], &response_output,
                               StateForModel()->OutputTensorName().c_str(),
                               StateForModel()->OutputDataType(),
                               out_shape.data(), out_shape.size()));
      if (responses[idx] != nullptr) {
        void *out_buffer;
        size_t out_buffer_size =
            TritonTypeSize(StateForModel()->OutputDataType()) * out_ids.size();
        TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t actual_memory_type_id = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &responses[idx], TRITONBACKEND_OutputBuffer(
                                 response_output, &out_buffer, out_buffer_size,
                                 &actual_memory_type, &actual_memory_type_id));
        ToOutBuffer(out_ids, StateForModel()->OutputDataType(), out_buffer);
      }
      idx += 1;
    }
    // Send all the responses that haven't already been sent because of
    // an earlier error.
    for (auto &response : responses) {
      if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send response");
      }
    }

    // Done with the request objects so release them.
    for (uint32_t r = 0; r < request_count; ++r) {
      auto &request = requests[r];
      LOG_IF_ERROR(TRITONBACKEND_RequestRelease(
                       request, TRITONSERVER_REQUEST_RELEASE_ALL),
                   "failed releasing request");
    }

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);

    if (!all_response_failed) {
#ifdef TRITON_ENABLE_STATS
      // Report batch statistics.
      LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                       TritonModelInstance(), total_batch_size, exec_start_ns,
                       compute_start_ns, compute_end_ns, exec_end_ns),
                   "failed reporting batch request statistics");
#endif // TRITON_ENABLE_STATS
    }
  }

private:
  ModelState *model_state_;
  ::ctranslate2::Device device_;
  std::shared_ptr<const ::ctranslate2::models::Model> model_;
  bool supports_batching_;
};

TRITONSERVER_Error *
ModelInstanceState::Create(ModelState *model_state,
                           TRITONBACKEND_ModelInstance *triton_model_instance,
                           ModelInstanceState **state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr; // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance) {
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model *model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void *vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void *>(instance_state)));

  return nullptr; // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState *instance_state =
      reinterpret_cast<ModelInstanceState *>(vstate);
  delete instance_state;

  return nullptr; // success
}

} // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                   TRITONBACKEND_Request **requests,
                                   const uint32_t request_count) {

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void **>(&instance_state)));
  ModelState *model_state = instance_state->StateForModel();

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("model ") + model_state->Name() + ", instance " +
               instance_state->Name() + ", executing " +
               std::to_string(request_count) + " requests")
                  .c_str());

  instance_state->ProcessRequests(requests, request_count);

  return nullptr; // success
}

} // extern "C"

} // namespace ctranslate2
} // namespace backend
} // namespace triton
