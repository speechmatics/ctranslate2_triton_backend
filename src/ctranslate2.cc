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

#include <memory>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

#include "ctranslate2/models/model.h"

namespace triton {
namespace backend {
namespace ctranslate2 {

//
// Minimal backend that demonstrates the TRITONBACKEND API. This
// backend works for any model that has 1 input called "IN0" with
// INT32 datatype and shape [ 4 ] and 1 output called "OUT0" with
// INT32 datatype and shape [ 4 ]. The backend supports both batching
// and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

const std::string INPUT_NAME("INPUT_IDS");
const std::string OUTPUT_NAME("OUTPUT_IDS");

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model,
                                    ModelState **state);
  virtual ~ModelState() = default;

  ModelState(TRITONBACKEND_Model *triton_model)
      : BackendModel(triton_model, false) {
    THROW_IF_BACKEND_MODEL_ERROR(ValidateModel());
  }

  TRITONSERVER_Error *
  LoadModel(const ::ctranslate2::Device device, std::int32_t device_index,
            std::shared_ptr<const ::ctranslate2::models::Model> *ct_model) {
    std::shared_ptr<const ::ctranslate2::models::Model> model;
    if (models_.empty()) {
      model = ::ctranslate2::models::Model::load(*model_reader_, device,
                                                 device_index);
      // TODO allow to specify COMPUTE_TYPE
    } else {
      model = models_.back()->copy_to(device, device_index);
    }
    models_.emplace_back(model);
    *ct_model = model;

    return nullptr;
  }

private:
  TRITONBACKEND_Model *triton_model_;
  triton::common::TritonJson::Value model_config_;
  std::string model_path_;
  std::shared_ptr<::ctranslate2::models::ModelReader> model_reader_;
  std::vector<std::shared_ptr<const ::ctranslate2::models::Model>> models_;

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
  TRITONSERVER_Message *config_message;

  const char *buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  triton::common::TritonJson::Value model_config;
  TRITONSERVER_Error *err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

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

      // Need to set the CUDA context so that the context that events are
      // created on match with contexts that events are recorded with.
      THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
          cudaSetDevice(DeviceId()), TRITONSERVER_ERROR_INTERNAL,
          "Failed to set the device"));
      THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
          cudaEventCreate(&compute_input_start_event_),
          TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
      THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
          cudaEventCreate(&compute_infer_start_event_),
          TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
      THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
          cudaEventCreate(&compute_output_start_event_),
          TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
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

  void CreateInput(size_t total_batch_size, TRITONBACKEND_Request **requests,
                   const uint32_t request_count,
                   std::vector<TRITONBACKEND_Response *> *responses,
                   BackendInputCollector *collector) {
  }

  void ProcessRequests(TRITONBACKEND_Request **requests,
                       const uint32_t request_count) {

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
        TRITONSERVER_Error *err = TRITONBACKEND_RequestInputByIndex(
            requests[i], 0 /* index */, &input);
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

    // At this point, the backend takes ownership of 'requests', which
    // means that it is responsible for sending a response for every
    // request. From here, even if something goes wrong in processing,
    // the backend must return 'nullptr' from this function to indicate
    // success. Any errors and failures must be communicated via the
    // response objects.
    //
    // To simplify error handling, the backend utilities manage
    // 'responses' in a specific way and it is recommended that backends
    // follow this same pattern. When an error is detected in the
    // processing of a request, an appropriate error response is sent
    // and the corresponding TRITONBACKEND_Response object within
    // 'responses' is set to nullptr to indicate that the
    // request/response has already been handled and no futher processing
    // should be performed for that request. Even if all responses fail,
    // the backend still allows execution to flow to the end of the
    // function. RESPOND_AND_SET_NULL_IF_ERROR, and
    // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
    // backend_common.h that assist in this management of response
    // objects.

    // The backend could iterate over the 'requests' and process each
    // one separately. But for performance reasons it is usually
    // preferred to create batched input tensors that are processed
    // simultaneously. This is especially true for devices like GPUs
    // that are capable of exploiting the large amount parallelism
    // exposed by larger data sets.
    //
    // The backend utilities provide a "collector" to facilitate this
    // batching process. The 'collector's ProcessTensor function will
    // combine a tensor's value from each request in the batch into a
    // single contiguous buffer. The buffer can be provided by the
    // backend or 'collector' can create and manage it. In this backend,
    // there is not a specific buffer into which the batch should be
    // created, so use ProcessTensor arguments that cause collector to
    // manage it.

    BackendInputCollector collector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(),
        model_state_->EnablePinnedInput() /* pinned_enabled */,
        nullptr /* stream*/);

    // To instruct ProcessTensor to "gather" the entire batch of IN0
    // input tensors into a single contiguous buffer in CPU memory, set
    // the "allowed input types" to be the CPU ones (see tritonserver.h
    // in the triton-inference-server/core repo for allowed memory
    // types).
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
        allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                               {TRITONSERVER_MEMORY_CPU, 0}};

    const char *input_buffer;
    size_t input_buffer_byte_size;
    TRITONSERVER_MemoryType input_buffer_memory_type;
    int64_t input_buffer_memory_type_id;

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            "IN0", nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types,
            &input_buffer, &input_buffer_byte_size, &input_buffer_memory_type,
            &input_buffer_memory_type_id));

    // Finalize the collector. If 'true' is returned, 'input_buffer'
    // will not be valid until the backend synchronizes the CUDA
    // stream or event that was used when creating the collector. For
    // this backend, GPU is not supported and so no CUDA sync should
    // be needed; so if 'true' is returned simply log an error.
    const bool need_cuda_input_sync = collector.Finalize();
    if (need_cuda_input_sync) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          "'minimal' backend: unexpected CUDA sync required by collector");
    }

    // 'input_buffer' contains the batched "IN0" tensor. The backend can
    // implement whatever logic is necesary to produce "OUT0". This
    // backend simply returns the IN0 value in OUT0 so no actual
    // computation is needed.

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("model ") + model_state->Name() +
                 ": requests in batch " + std::to_string(request_count))
                    .c_str());
    std::string tstr;
    IGNORE_ERROR(BufferAsTypedString(tstr, input_buffer, input_buffer_byte_size,
                                     TRITONSERVER_TYPE_INT32));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("batched IN0 value: ") + tstr).c_str());

    const char *output_buffer = input_buffer;
    TRITONSERVER_MemoryType output_buffer_memory_type =
        input_buffer_memory_type;
    int64_t output_buffer_memory_type_id = input_buffer_memory_type_id;

    // This backend supports models that batch along the first dimension
    // and those that don't batch. For non-batch models the output shape
    // will be [ 4 ]. For batch models the output shape will be [ -1, 4
    // ] and the backend "responder" utility below will set the
    // appropriate batch dimension value for each response.
    std::vector<int64_t> output_batch_shape;
    bool supports_first_dim_batching;
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        model_state->SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      output_batch_shape.push_back(-1);
    }
    output_batch_shape.push_back(4);

    // Because the OUT0 values are concatenated into a single contiguous
    // 'output_buffer', the backend must "scatter" them out to the
    // individual response OUT0 tensors.  The backend utilities provide
    // a "responder" to facilitate this scattering process.

    // The 'responders's ProcessTensor function will copy the portion of
    // 'output_buffer' corresonding to each request's output into the
    // response for that request.

    BackendOutputResponder responder(
        requests, request_count, &responses, model_state->TritonMemoryManager(),
        supports_first_dim_batching, false /* pinned_enabled */,
        nullptr /* stream*/);

    responder.ProcessTensor("OUT0", TRITONSERVER_TYPE_INT32, output_batch_shape,
                            output_buffer, output_buffer_memory_type,
                            output_buffer_memory_type_id);

    // Finalize the responder. If 'true' is returned, the OUT0
    // tensors' data will not be valid until the backend synchronizes
    // the CUDA stream or event that was used when creating the
    // responder. For this backend, GPU is not supported and so no
    // CUDA sync should be needed; so if 'true' is returned simply log
    // an error.
    const bool need_cuda_output_sync = responder.Finalize();
    if (need_cuda_output_sync) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          "'minimal' backend: unexpected CUDA sync required by responder");
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

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

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

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
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
