// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_binary_op.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

namespace {
struct QLinearBroadcastHelper : public BroadcastHelper {
  QLinearBroadcastHelper(InputBroadcaster& input_broadcaster,
                         OutputBroadcaster& output_broadcaster,
                         ThreadPool* threadpool,
                         double unit_cost,
                         float A_scale_in, float B_scale_in, float C_scale_in,
                         uint8_t A_zero_point_in, uint8_t B_zero_point_in, uint8_t C_zero_point_in)
      : BroadcastHelper{input_broadcaster, output_broadcaster, nullptr, threadpool, unit_cost},
        A_scale{A_scale_in},
        B_scale{B_scale_in},
        C_scale{C_scale_in},
        A_zero_point{A_zero_point_in},
        B_zero_point{B_zero_point_in},
        C_zero_point{C_zero_point_in} {
  }

  QLinearBroadcastHelper(const QLinearBroadcastHelper& rhs, size_t offset, size_t num_elements)
      : BroadcastHelper(rhs, offset, num_elements),
        A_scale{rhs.A_scale},
        B_scale{rhs.B_scale},
        C_scale{rhs.C_scale},
        A_zero_point{rhs.A_zero_point},
        B_zero_point{rhs.B_zero_point},
        C_zero_point{rhs.C_zero_point} {
  }

  float A_scale;
  float B_scale;
  float C_scale;
  // storage for these is uint8_t but original value may be uint8_t or int8_t.
  // typed code that uses values needs to cast to correct representation
  uint8_t A_zero_point;
  uint8_t B_zero_point;
  uint8_t C_zero_point;
};

template <typename T>
void QLinearImpl(OpKernelContext& context, double unit_cost, const ProcessBroadcastSpanFuncs& functors) {
  auto tensor_a_scale = context.Input<Tensor>(1);
  auto tensor_a_zero_point = context.Input<Tensor>(2);
  auto tensor_b_scale = context.Input<Tensor>(4);
  auto tensor_b_zero_point = context.Input<Tensor>(5);
  auto tensor_c_scale = context.Input<Tensor>(6);
  auto tensor_c_zero_point = context.Input<Tensor>(7);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_a_scale),
              "MatmulInteger : input1 A_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_a_zero_point == nullptr || IsScalarOr1ElementVector(tensor_a_zero_point),
              "MatmulInteger : input1 A_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_b_scale),
              "MatmulInteger : input1 B_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_b_zero_point == nullptr || IsScalarOr1ElementVector(tensor_b_zero_point),
              "MatmulInteger : input1 B_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_c_scale),
              "MatmulInteger : input1 C_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_c_zero_point == nullptr || IsScalarOr1ElementVector(tensor_c_zero_point),
              "MatmulInteger : input1 C_zero_point must be a scalar or 1D tensor of size 1 if given");

  const float A_scale = *(tensor_a_scale->Data<float>());
  const T A_zero_point = (nullptr == tensor_a_zero_point) ? T{} : *(tensor_a_zero_point->Data<T>());
  const float B_scale = *(tensor_b_scale->Data<float>());
  const T B_zero_point = (nullptr == tensor_b_zero_point) ? T{} : *(tensor_b_zero_point->Data<T>());
  const float C_scale = *(tensor_c_scale->Data<float>());
  const T C_zero_point = (nullptr == tensor_c_zero_point) ? T{} : *(tensor_c_zero_point->Data<T>());

  InputBroadcaster input_broadcaster{*context.Input<Tensor>(0), *context.Input<Tensor>(3)};
  OutputBroadcaster output_broadcaster{input_broadcaster.GetSpanSize(),
                                       *context.Output(0, input_broadcaster.GetOutputShape())};

  QLinearBroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster,
                                          context.GetOperatorThreadPool(), unit_cost,
                                          A_scale, B_scale, C_scale,
                                          static_cast<uint8_t>(A_zero_point),
                                          static_cast<uint8_t>(B_zero_point),
                                          static_cast<uint8_t>(C_zero_point));

  BroadcastLooper(broadcast_helper, functors);
}
}  // namespace

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  const ProcessBroadcastSpanFuncs functors = {
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        const T input0 = per_iter_bh.ScalarInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearAdd(input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       &input0,
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        const T input1 = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        MlasQLinearAdd(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       &input1,
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearAdd(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), false);
      }};

  QLinearImpl<T>(*context, 1.0, functors);

  return Status::OK();
}

template <typename T>
Status QLinearMul<T>::Compute(OpKernelContext* context) const {
  const ProcessBroadcastSpanFuncs functors = {
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        const T input0 = per_iter_bh.ScalarInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearMul(input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       &input0,
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        const T input1 = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        MlasQLinearMul(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       &input1,
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), true);
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        MlasQLinearMul(input0.data(),
                       qlbh.A_scale, static_cast<T>(qlbh.A_zero_point),
                       input1.data(),
                       qlbh.B_scale, static_cast<T>(qlbh.B_zero_point),
                       qlbh.C_scale, static_cast<T>(qlbh.C_zero_point),
                       output.data(), output.size(), false);
      }};

  QLinearImpl<T>(*context, 1.0, functors);

  return Status::OK();
}

template <typename T>
Status FAQLinearMul<T>::Compute(OpKernelContext* context) const {
    ProcessBroadcastSpanFuncs funcs{
        nullptr,
        [](BroadcastHelper& per_iter_bh) {
            auto input = per_iter_bh.SpanInput0<T>();
            auto scale = per_iter_bh.ScalarInput1<float>();
            auto output = per_iter_bh.OutputEigen<T>();
            MlasFAQLinearMul(input.data(), scale, output.data(), output.size());
            
        },
        nullptr};
    UntypedBroadcastTwo(*context, funcs, 1.0);
    return Status::OK();
}

FAUpsamplinglinearParams SetupFAUpsamplingLinear(const int32_t input_height,
                                                 const int32_t input_width,
                                                 const int32_t output_height,
                                                 const int32_t output_width,
                                                 const float height_scale,
                                                 const float width_scale,
                                                 AllocatorPtr& alloc) {
    FAUpsamplinglinearParams p;
    
    p.x_original.reserve(output_width);
    p.y_original.reserve(output_height);
    
    const SafeInt<size_t> idx_buffer_size = SafeInt<size_t>(2) * sizeof(int32_t) * (output_height + output_width);
    const SafeInt<size_t> scale_buffer_size = SafeInt<size_t>(2) * sizeof(int32_t) * (output_height + output_width);
    
    const auto inx_scale_data_buffer = alloc->Alloc(idx_buffer_size + scale_buffer_size);
    p.idx_scale_data_buffer_holder = BufferUniquePtr(inx_scale_data_buffer, BufferDeleter(alloc));
    
    auto* const idx_data = static_cast<int32_t*>(p.idx_scale_data_buffer_holder.get());
    
    p.input_width_mul_y1 = idx_data;
    p.input_width_mul_y2 = p.input_width_mul_y1 + output_height;
    
    const auto output_height_x2 = output_height * 2;  // this is to make prefast happy
    p.in_x1 = p.input_width_mul_y1 + output_height_x2;
    p.in_x2 = p.in_x1 + output_width;
    
    for (int32_t y = 0; y < output_height; ++y) {
        float in_y = height_scale == 1 ? static_cast<float>(y)
        : ((static_cast<float>(y) + 0.5f) / height_scale - 0.5f);
        p.y_original.emplace_back(in_y);
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(input_height - 1)));
        const int32_t in_y1 = std::min(static_cast<int32_t>(in_y), input_height - 1);
        const int32_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        p.input_width_mul_y1[narrow<size_t>(y)] = input_width * in_y1;
        p.input_width_mul_y2[narrow<size_t>(y)] = input_width * in_y2;
    }
    
    for (int32_t x = 0; x < output_width; ++x) {
        float in_x = width_scale == 1 ? static_cast<float>(x)
        : ((static_cast<float>(x) + 0.5f) / height_scale - 0.5f);
        p.x_original.emplace_back(in_x);
        in_x = std::max(0.0f, std::min(in_x, static_cast<float>(input_width - 1)));
        p.in_x1[narrow<size_t>(x)] = std::min(static_cast<int32_t>(in_x), input_width - 1);
        p.in_x2[narrow<size_t>(x)] = std::min(p.in_x1[narrow<size_t>(x)] + 1, input_width - 1);
    }
    return p;
}


template <typename T>
void FAUpsamplelinear(const int32_t batch_size,
                      const int32_t num_channels,
                      const int32_t input_height,
                      const int32_t input_width,
                      const int32_t output_height,
                      const int32_t output_width,
                      const float height_scale,
                      const float width_scale,
                      const T* const XdataBase,
                      T* const YdataBase,
                      AllocatorPtr& alloc,
                      concurrency::ThreadPool* tp) {
    FAUpsamplinglinearParams p = SetupFAUpsamplingLinear(input_height, input_width, output_height, output_width, height_scale, width_scale, alloc);
    for (int32_t n = 0; n < batch_size; ++n) {
        concurrency::ThreadPool::TrySimpleParallelFor
        (
         tp, num_channels,
         [&](std::ptrdiff_t c) {
             const T* const Xdata =
             XdataBase + (n * num_channels + static_cast<int32_t>(c)) * (input_height * input_width);
             T* const Ydata = YdataBase + (n * num_channels + static_cast<int32_t>(c)) * (output_height * output_width);
             for (int32_t y = 0; y < output_height; ++y) {
                 for (int32_t x = 0; x < output_width; ++x) {
                     const int32_t output_offset = output_width * y + x;
                     
                     int64_t f11 = 0x24000;
                     int64_t f21 = 0xC000;
                     int64_t f12 = 0xC000;
                     int64_t f22 = 0x4000;
                     int64_t data = 0;
                     
                     if (y == 0 || y == output_height - 1) {
                         int32_t index1 = p.input_width_mul_y1[y] + p.in_x1[x];
                         int32_t index2 = p.input_width_mul_y1[y] + p.in_x2[x];
                         T X11 = Xdata[index1];
                         T X21 = Xdata[index2];
                         if (x == 0 || x == output_width - 1) {
                             data = (int64_t)(f11 * X11) >> 18;
                         } else {
                             if (x % 2 == 0) {
                                 f11 = 0xC000;
                                 f21 = 0x24000;
                             }
                             data = (int64_t)(f11 * X11 + f21 * X21) >> 18;
                         }
                         Ydata[output_offset] = static_cast<T>(data);
                     } else {
                         int32_t index1 = p.input_width_mul_y1[y] + p.in_x1[x];
                         int32_t index2 = p.input_width_mul_y1[y] + p.in_x2[x];
                         int32_t index3 = p.input_width_mul_y2[y] + p.in_x1[x];
                         int32_t index4 = p.input_width_mul_y2[y] + p.in_x2[x];
                         T X11 = Xdata[index1];
                         T X21 = Xdata[index2];
                         T X12 = Xdata[index3];
                         T X22 = Xdata[index4];
                         if (x == 0 || x == output_width - 1) {
                             if (y % 2 == 0) {
                                 f11 = 0xC000;
                                 f12 = 0x24000;
                             }
                             data = (int64_t)(f11 * X11 + f12 * X12) >> 18;
                             Ydata[output_offset] = static_cast<T>(data);
                         } else {
                             if (y % 2 == 1) {
                                 if (x % 2 == 0) {
                                     f11 = 0xC000;
                                     f21 = 0x24000;
                                     f12 = 0x4000;
                                     f22 = 0xC000;
                                 }
                             } else {
                                 if (x % 2 == 0) {
                                     f11 = 0x4000;
                                     f21 = 0xC000;
                                     f12 = 0xC000;
                                     f22 = 0x24000;
                                 } else {
                                     f11 = 0xC000;
                                     f21 = 0x4000;
                                     f12 = 0x24000;
                                     f22 = 0xC000;
                                 }
                             }
                             data = (int64_t)(f11 * X11 + f21 * X21 + f12 * X12 + f22 * X22) >> 18;
                             Ydata[output_offset] = static_cast<T>(data);
                         }
                     }
                 }
             }
         });
    }
}

template <typename T>
Status FAUpSamplingLinear<T>::Compute(OpKernelContext* context) const {
    const auto* X = context->Input<Tensor>(0);
    auto dims = X->Shape().GetDims();
    TensorShapeVector output_dims(dims.size());
    
    for (std::size_t i = 0; i < dims.size(); i++) {
        output_dims[i] = static_cast<int64_t>(scales_[i] * dims[i]);
    }
    
    ORT_RETURN_IF_NOT(output_dims.size() == dims.size(), "Rank of input and output tensor should be same.");
    Tensor* Y = context->Output(0, output_dims);
    if (Y->Shape().Size() == 0) {
        return Status::OK();
    }
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    
    if (dims.size() == 2 || dims.size() == 4) {
        bool is_2D = dims.size() == 2;
        
        int32_t batch_size;
        int32_t num_channels;
        int32_t input_height;
        int32_t input_width;
        
        int32_t output_height;
        int32_t output_width;
        
        float height_scale;
        float width_scale;
        
        if (is_2D) {
            batch_size = 1;
            num_channels = 1;
            input_height = static_cast<int32_t>(dims[0]);
            input_width = static_cast<int32_t>(dims[1]);
            
            output_height = static_cast<int32_t>(output_dims[0]);
            output_width = static_cast<int32_t>(output_dims[1]);
            
            height_scale = scales_[0];
            width_scale = scales_[1];
        } else {
            batch_size = static_cast<int32_t>(dims[0]);
            num_channels = static_cast<int32_t>(dims[1]);
            input_height = static_cast<int32_t>(dims[2]);
            input_width = static_cast<int32_t>(dims[3]);
            
            output_height = static_cast<int32_t>(output_dims[2]);
            output_width = static_cast<int32_t>(output_dims[3]);
            
            height_scale = scales_[2];
            width_scale = scales_[3];
        }
        
        FAUpsamplelinear(batch_size, num_channels, input_height, input_width, output_height, output_width,
                         height_scale, width_scale, X->Data<T>(), Y->MutableData<T>(), alloc,
                         output_height * output_width > 64 ? context->GetOperatorThreadPool() : nullptr);
        
    }
    return Status::OK();
}

Status FAQlinearShuffleNetCopyData(const int8_t* src_left, int8_t* dst_left, const int8_t* src_right, int8_t* dst_right, std::vector<float> scale, const int64_t block_size, const int64_t M, const int64_t N, const int64_t data_batch_bytes, const int64_t gathered_batch_bytes, concurrency::ThreadPool* tp)
{
    auto lambda = [&](int64_t index) {
        int64_t batch = index / N;
        int64_t i = index % N;
        
        const int64_t src_offset_batch = batch * data_batch_bytes;
        const int64_t dst_offset_batch = batch * gathered_batch_bytes;
        int64_t offset = narrow<size_t>(block_size);
        if (i == 0) {
            float left_scale1 = scale[0];
            if (std::abs(left_scale1 - 1.0) < 0.000001) {
                memcpy(dst_left + dst_offset_batch, src_left + src_offset_batch, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_left + src_offset_batch, left_scale1, dst_left + dst_offset_batch, narrow<size_t>(block_size));
            }
            float left_scale2 = scale[3];
            if (std::abs(left_scale2 - 1.0) < 0.000001) {
                memcpy(dst_left + offset + dst_offset_batch, src_right + src_offset_batch, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_right + src_offset_batch, left_scale2, dst_left + offset + dst_offset_batch, narrow<size_t>(block_size));
            }
        } else if (i == 1) {
            float left_scale1 = scale[0];
            if (std::abs(left_scale1 - 1.0) < 0.000001) {
                memcpy(dst_left + dst_offset_batch + 2 * offset, src_left + src_offset_batch + offset, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_left + src_offset_batch + offset, left_scale1, dst_left + dst_offset_batch + 2 * offset, narrow<size_t>(block_size));
            }
            float left_scale2 = scale[3];
            if (std::abs(left_scale2 - 1.0) < 0.000001) {
                memcpy(dst_left + dst_offset_batch + 3 * offset, src_right + src_offset_batch + offset, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_right + src_offset_batch + offset, left_scale2, dst_left + dst_offset_batch + 3 * offset, narrow<size_t>(block_size));
            }
        } else if (i == 2) {
            float right_scale1 = scale[4];
            if (std::abs(right_scale1 - 1.0) < 0.000001) {
                memcpy(dst_right + dst_offset_batch, src_left + src_offset_batch + 2 * offset, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_left + src_offset_batch + 2 * offset, right_scale1, dst_right + dst_offset_batch, narrow<size_t>(block_size));
            }
            float right_scale2 = scale[1];
            if (std::abs(right_scale2 - 1.0) < 0.000001) {
                memcpy(dst_right + dst_offset_batch + offset, src_right + src_offset_batch + 2 * offset, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_right + src_offset_batch + 2 * offset, right_scale2, dst_right + dst_offset_batch + offset, narrow<size_t>(block_size));
            }
        } else if (i == 3) {
            float right_scale1 = scale[4];
            if (std::abs(right_scale1 - 1.0) < 0.000001) {
                memcpy(dst_right + dst_offset_batch + 2 * offset, src_left + src_offset_batch + 3 * offset, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_left + src_offset_batch + 3 * offset, right_scale1, dst_right + dst_offset_batch + 2 * offset, narrow<size_t>(block_size));
            }
            float right_scale2 = scale[1];
            if (std::abs(right_scale2 - 1.0) < 0.000001) {
                memcpy(dst_right + dst_offset_batch + 3 * offset, src_right + src_offset_batch + 3 * offset, narrow<size_t>(block_size));
            } else {
                MlasFAQLinearMul(src_right + src_offset_batch + 3 * offset, right_scale2, dst_right + dst_offset_batch + 3 * offset, narrow<size_t>(block_size));
            }
        }
    };
    ORT_RETURN_IF_NOT(scale.size() == 5, "ShuffleNet param size must be more than 5.");
    concurrency::ThreadPool::TryParallelFor(tp, SafeInt<ptrdiff_t>(M) * N, static_cast<double>(block_size), [&lambda](ptrdiff_t first, ptrdiff_t last) {
        for (int index = static_cast<int>(first), end = static_cast<int>(last); index < end; ++index) {
            lambda(index);
        }
    });
    
    return Status::OK();
}

template <typename T>
Status FAQLinearShuffleNet<T>::Compute(OpKernelContext* context) const {
    const auto* input_left = context->Input<Tensor>(0);
    auto dims_left = input_left->Shape().GetDims();
    
    const auto* input_right = context->Input<Tensor>(1);
    auto dims_right = input_right->Shape().GetDims();
    
    TensorShapeVector output_dims_left(dims_left.size());
    TensorShapeVector output_dims_right(dims_right.size());
    ORT_RETURN_IF_NOT(dims_left.size() == dims_right.size() && dims_left.size() == 4, "Rank of input and output tensor should be same.");
    ORT_RETURN_IF_NOT(shuffle_params_.size() == 5, "ShuffleNet param size must be more than 5.");
    for (std::size_t i = 0; i < dims_left.size(); i++) {
        int64_t left_size = static_cast<int64_t>(dims_left[i]);
        int64_t right_size = static_cast<int64_t>(dims_right[i]);
        ORT_RETURN_IF_NOT(left_size == right_size, "input of size should be same.");
        output_dims_left[i] = left_size;
        output_dims_right[i] = right_size;
    }
    
    Tensor* output_left = context->Output(0, output_dims_left);
    Tensor* output_right = context->Output(1, output_dims_right);

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    
    const TensorShape& input_data_shape = input_left->Shape();
    
    int32_t shuffle_count =  static_cast<int32_t>(shuffle_params_[2]);
    const size_t element_bytes = input_left->DataType()->Size();
    const int64_t block = input_data_shape.SizeFromDimension(2);
    const int64_t block_size = SafeInt<int64_t>(element_bytes) * block * shuffle_count;
    const int64_t M = input_data_shape.SizeToDimension(1);
    const int64_t N = dims_left[1] / shuffle_count;
    const int64_t data_batch_bytes = input_data_shape.SizeFromDimension(1) * element_bytes;
    const int64_t gathered_batch_bytes = N * shuffle_count * block * SafeInt<int64_t>(element_bytes);
    
    const auto* src_left = static_cast<const int8_t*>(input_left->DataRaw());
    auto* dst_left = static_cast<int8_t*>(output_left->MutableDataRaw());
    
    const auto* src_right = static_cast<const int8_t*>(input_right->DataRaw());
    auto* dst_right = static_cast<int8_t*>(output_right->MutableDataRaw());
    
    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    
    return FAQlinearShuffleNetCopyData(src_left, dst_left, src_right, dst_right, shuffle_params_, block_size, M, N, data_batch_bytes, gathered_batch_bytes, tp);
}

#define REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                    \
      op_name, version, data_type,                                                      \
      KernelDefBuilder()                                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),               \
      KERNEL_CLASS<data_type>);

REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, int8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, uint8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, int8_t, QLinearMul);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, uint8_t, QLinearMul);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(FAQLinearMul, 1, int8_t, FAQLinearMul);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(FAQLinearMul, 1, uint8_t, FAQLinearMul);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(FAUpSamplingLinear, 1, int8_t, FAUpSamplingLinear);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(FAUpSamplingLinear, 1, uint8_t, FAUpSamplingLinear);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(FAQLinearShuffleNet, 1, int8_t, FAQLinearShuffleNet);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(FAQLinearShuffleNet, 1, uint8_t, FAQLinearShuffleNet);
}  // namespace contrib
}  // namespace onnxruntime
