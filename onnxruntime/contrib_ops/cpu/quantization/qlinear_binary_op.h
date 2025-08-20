// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <core/common/safeint.h>
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class QLinearAdd final : public OpKernel {
 public:
  QLinearAdd(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class QLinearMul final : public OpKernel {
 public:
  QLinearMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class FAQLinearMul final : public OpKernel {
 public:
  FAQLinearMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

struct FAUpsamplinglinearParams {
  std::vector<float> x_original;
  std::vector<float> y_original;

  BufferUniquePtr idx_scale_data_buffer_holder;

  int32_t* input_width_mul_y1{nullptr};
  int32_t* input_width_mul_y2{nullptr};

  int32_t* in_x1{nullptr};
  int32_t* in_x2{nullptr};
};

template <typename T>
class FAUpSamplingLinear : public OpKernel {
 public:
  FAUpSamplingLinear(const OpKernelInfo& info) : OpKernel(info) {
      const Tensor* scale;
      info.TryGetConstantInput(1, &scale);
      const auto* scale_data = scale->Data<float>();
      int64_t scales_size = scale->Shape().Size();
      if (scales_.empty()) {
          scales_.resize(onnxruntime::narrow<size_t>(scales_size));
      }
      memcpy(scales_.data(), scale_data, SafeInt<size_t>(scales_size) * sizeof(float));
  }

  Status Compute(OpKernelContext* context) const override;
    
private:
    std::vector<float> scales_;
};

template <typename T>
class FAQLinearShuffleNet : public OpKernel {
 public:
  FAQLinearShuffleNet(const OpKernelInfo& info) : OpKernel(info) {
      const Tensor* param;
      info.TryGetConstantInput(2, &param);
      const auto* param_data = param->Data<float>();
      int64_t param_size = param->Shape().Size();
      if (shuffle_params_.empty()) {
          shuffle_params_.resize(onnxruntime::narrow<size_t>(param_size));
      }
      memcpy(shuffle_params_.data(), param_data, SafeInt<size_t>(param_size) * sizeof(float));
  }

  Status Compute(OpKernelContext* context) const override;
    
private:
    std::vector<float> shuffle_params_;
};

}  // namespace contrib
}  // namespace onnxruntime
