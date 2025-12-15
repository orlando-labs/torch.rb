#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <rice/rice.hpp>
#include <ruby/ruby.h>

#include "tensor_functions.h"
#include "ruby_arg_parser.h"
#include "templates.h"
#include "utils.h"

using Rice::Array;
using Rice::Object;
using torch::indexing::TensorIndex;

static inline VALUE to_ruby_value(uint8_t value) { return UINT2NUM(value); }
static inline VALUE to_ruby_value(int8_t value) { return INT2NUM(static_cast<long>(value)); }
static inline VALUE to_ruby_value(int16_t value) { return INT2NUM(static_cast<long>(value)); }
static inline VALUE to_ruby_value(int32_t value) { return INT2NUM(static_cast<long>(value)); }
static inline VALUE to_ruby_value(int64_t value) { return LL2NUM(value); }
static inline VALUE to_ruby_value(float value) { return DBL2NUM(static_cast<double>(value)); }
static inline VALUE to_ruby_value(double value) { return DBL2NUM(value); }
static inline VALUE to_ruby_value(bool value) { return value ? Qtrue : Qfalse; }
static inline VALUE to_ruby_value(const c10::complex<float>& value) {
  return rb_dbl_complex_new(value.real(), value.imag());
}
static inline VALUE to_ruby_value(const c10::complex<double>& value) {
  return rb_dbl_complex_new(value.real(), value.imag());
}

// normalize tensor for direct memory access
// moves to CPU, resolves view metadata, and ensures contiguous storage
static Tensor prepare_tensor_for_read(const Tensor& source) {
  auto tensor = source;

  if (tensor.device().type() != torch::kCPU) {
    torch::Device device("cpu");
    tensor = tensor.to(device);
  }

  if (tensor.is_conj()) {
    tensor = tensor.resolve_conj();
  }
  if (tensor.is_neg()) {
    tensor = tensor.resolve_neg();
  }

  if (!tensor.is_contiguous()) {
    tensor = tensor.contiguous();
  }

  return tensor;
}

template<typename T>
Array flat_data(Tensor& tensor) {
  // tensor must already be on CPU and contiguous so data_ptr covers flattened storage
  const auto numel = tensor.numel();
  const T* data = tensor.data_ptr<T>();

  VALUE ary = rb_ary_new_capa(static_cast<long>(numel));
  for (int64_t i = 0; i < numel; i++) {
    rb_ary_push(ary, to_ruby_value(data[i]));
  }

  return Array(ary);
}

template<typename T>
static VALUE nested_array_from_data(const T*& data, const std::vector<int64_t>& sizes, size_t dim) {
  const int64_t dim_size = sizes[dim];
  VALUE ary = rb_ary_new_capa(static_cast<long>(dim_size));

  if (dim + 1 == sizes.size()) {
    for (int64_t i = 0; i < dim_size; ++i) {
      rb_ary_push(ary, to_ruby_value(*data));
      data++;
    }
    return ary;
  }

  for (int64_t i = 0; i < dim_size; ++i) {
    rb_ary_push(ary, nested_array_from_data(data, sizes, dim + 1));
  }

  return ary;
}

template<typename T>
static Object tensor_to_ruby_array_for_type(Tensor& tensor, const std::vector<int64_t>& sizes) {
  const auto numel = tensor.numel();
  const T* data = tensor.data_ptr<T>();

  if (sizes.empty()) {
    VALUE ary = rb_ary_new_capa(static_cast<long>(numel));
    for (int64_t i = 0; i < numel; ++i) {
      rb_ary_push(ary, to_ruby_value(data[i]));
    }
    return Object(ary);
  }

  return Object(nested_array_from_data(data, sizes, 0));
}

Rice::Class rb_cTensor;

std::vector<TensorIndex> index_vector(Array a) {
  Object obj;

  std::vector<TensorIndex> indices;
  indices.reserve(a.size());

  for (long i = 0; i < a.size(); i++) {
    obj = a[i];

    if (obj.is_instance_of(rb_cInteger)) {
      indices.push_back(Rice::detail::From_Ruby<int64_t>().convert(obj.value()));
    } else if (obj.is_instance_of(rb_cRange)) {
      torch::optional<c10::SymInt> start_index = torch::nullopt;
      torch::optional<c10::SymInt> stop_index = torch::nullopt;

      Object begin = obj.call("begin");
      if (!begin.is_nil()) {
        start_index = c10::SymInt(Rice::detail::From_Ruby<int64_t>().convert(begin.value()));
      }

      Object end = obj.call("end");
      if (!end.is_nil()) {
        stop_index = c10::SymInt(Rice::detail::From_Ruby<int64_t>().convert(end.value()));
      }

      Object exclude_end = obj.call("exclude_end?");
      if (stop_index.has_value() && !exclude_end) {
        if (stop_index.value() == -1) {
          stop_index = torch::nullopt;
        } else {
          stop_index = stop_index.value() + 1;
        }
      }

      indices.push_back(torch::indexing::Slice(start_index, stop_index));
    } else if (obj.is_instance_of(rb_cTensor)) {
      indices.push_back(Rice::detail::From_Ruby<Tensor>().convert(obj.value()));
    } else if (obj.is_nil()) {
      indices.push_back(torch::indexing::None);
    } else if (obj == Rice::True || obj == Rice::False) {
      indices.push_back(Rice::detail::From_Ruby<bool>().convert(obj.value()));
    } else {
      throw Rice::Exception(rb_eArgError, "Unsupported index type: %s", rb_obj_classname(obj));
    }
  }
  return indices;
}

// hack (removes inputs argument)
// https://github.com/pytorch/pytorch/commit/2e5bfa9824f549be69a28e4705a72b4cf8a4c519
// TODO add support for inputs argument
// _backward
static VALUE tensor__backward(int argc, VALUE* argv, VALUE self_) {
  HANDLE_TH_ERRORS
  Tensor& self = Rice::detail::From_Ruby<Tensor&>().convert(self_);
  static RubyArgParser parser({
    "_backward(Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False)"
  });
  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, argc, argv, parsed_args);
  // _backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
  auto dispatch__backward = [](const Tensor & self, TensorList inputs, const c10::optional<at::Tensor> & gradient, c10::optional<bool> retain_graph, bool create_graph) -> void {
    // in future, release GVL
    self._backward(inputs, gradient, retain_graph, create_graph);
  };
  dispatch__backward(self, {}, _r.optionalTensor(0), _r.toBoolOptional(1), _r.toBool(2));
  RETURN_NIL
  END_HANDLE_TH_ERRORS
}

void init_tensor(Rice::Module& m, Rice::Class& c, Rice::Class& rb_cTensorOptions) {
  rb_cTensor = c;
  add_tensor_functions(rb_cTensor);
  THPVariableClass = rb_cTensor.value();

  rb_define_method(rb_cTensor, "backward", (VALUE (*)(...)) tensor__backward, -1);

  rb_cTensor
    .define_method("cuda?", [](Tensor& self) { return self.is_cuda(); })
    .define_method("mps?", [](Tensor& self) { return self.is_mps(); })
    .define_method("sparse?", [](Tensor& self) { return self.is_sparse(); })
    .define_method("quantized?", [](Tensor& self) { return self.is_quantized(); })
    .define_method("dim", [](Tensor& self) { return self.dim(); })
    .define_method("numel", [](Tensor& self) { return self.numel(); })
    .define_method("element_size", [](Tensor& self) { return self.element_size(); })
    .define_method("requires_grad", [](Tensor& self) { return self.requires_grad(); })
    .define_method(
      "_size",
      [](Tensor& self, int64_t dim) {
        return self.size(dim);
      })
    .define_method(
      "_stride",
      [](Tensor& self, int64_t dim) {
        return self.stride(dim);
      })
    // in C++ for performance
    .define_method(
      "shape",
      [](Tensor& self) {
        Array a;
        for (auto &size : self.sizes()) {
          a.push(size, false);
        }
        return a;
      })
    .define_method(
      "_strides",
      [](Tensor& self) {
        Array a;
        for (auto &stride : self.strides()) {
          a.push(stride, false);
        }
        return a;
      })
    .define_method(
      "_index",
      [](Tensor& self, Array indices) {
        auto vec = index_vector(indices);
        return self.index(vec);
      })
    .define_method(
      "_index_put_custom",
      [](Tensor& self, Array indices, torch::Tensor& value) {
        auto vec = index_vector(indices);
        return self.index_put_(vec, value);
      })
    .define_method(
      "contiguous?",
      [](Tensor& self) {
        return self.is_contiguous();
      })
    .define_method(
      "_requires_grad!",
      [](Tensor& self, bool requires_grad) {
        return self.set_requires_grad(requires_grad);
      })
    .define_method(
      "grad",
      [](Tensor& self) {
        auto grad = self.grad();
        return grad.defined() ? Object(Rice::detail::To_Ruby<torch::Tensor>().convert(grad)) : Rice::Nil;
      })
    // can't use grad=
    // assignment methods fail with Ruby 3.0
    .define_method(
      "_set_grad",
      [](Tensor& self, Rice::Object value) {
        if (value.is_nil()) {
          self.mutable_grad().reset();
          return;
        }

        const auto& grad = Rice::detail::From_Ruby<torch::Tensor>().convert(value.value());

        // TODO support sparse grad
        if (!grad.options().type_equal(self.options())) {
          rb_raise(rb_eArgError, "assigned grad has data of a different type");
        }

        if (self.is_cuda() && grad.get_device() != self.get_device()) {
          rb_raise(rb_eArgError, "assigned grad has data located on a different device");
        }

        if (!self.sizes().equals(grad.sizes())) {
          rb_raise(rb_eArgError, "assigned grad has data of a different size");
        }

        self.mutable_grad() = grad;
      })
    .define_method(
      "_dtype",
      [](Tensor& self) {
        return static_cast<int>(at::typeMetaToScalarType(self.dtype()));
      })
    .define_method(
      "_type",
      [](Tensor& self, int dtype) {
        return self.toType((torch::ScalarType) dtype);
      })
    .define_method(
      "_layout",
      [](Tensor& self) {
        std::stringstream s;
        s << self.layout();
        return s.str();
      })
    .define_method(
      "device",
      [](Tensor& self) {
        return self.device();
      })
    .define_method(
      "_data_str",
      [](Tensor& self) {
        auto tensor = prepare_tensor_for_read(self);
        auto data_ptr = (const char *) tensor.data_ptr();
        return std::string(data_ptr, tensor.numel() * tensor.element_size());
      })
    // for TorchVision
    .define_method(
      "_data_ptr",
      [](Tensor& self) {
        return reinterpret_cast<uintptr_t>(self.data_ptr());
      })
    // TODO figure out a better way to do this
    .define_method(
      "_flat_data",
      [](Tensor& self) {
        auto tensor = prepare_tensor_for_read(self);
        auto dtype = tensor.dtype();
        if (dtype == torch::kByte) {
          return flat_data<uint8_t>(tensor);
        } else if (dtype == torch::kChar) {
          return flat_data<int8_t>(tensor);
        } else if (dtype == torch::kShort) {
          return flat_data<int16_t>(tensor);
        } else if (dtype == torch::kInt) {
          return flat_data<int32_t>(tensor);
        } else if (dtype == torch::kLong) {
          return flat_data<int64_t>(tensor);
        } else if (dtype == torch::kFloat) {
          return flat_data<float>(tensor);
        } else if (dtype == torch::kDouble) {
          return flat_data<double>(tensor);
        } else if (dtype == torch::kBool) {
          return flat_data<bool>(tensor);
        } else if (dtype == torch::kComplexFloat) {
          return flat_data<c10::complex<float>>(tensor);
        } else if (dtype == torch::kComplexDouble) {
          return flat_data<c10::complex<double>>(tensor);
        } else {
          throw std::runtime_error("Unsupported type");
        }
      })
    .define_method(
      "_to_a",
      [](Tensor& self) {
        auto tensor = prepare_tensor_for_read(self);

        std::vector<int64_t> sizes;
        sizes.reserve(tensor.dim());
        for (auto size : tensor.sizes()) {
          sizes.push_back(size);
        }

        auto dtype = tensor.dtype();
        if (dtype == torch::kByte) {
          return tensor_to_ruby_array_for_type<uint8_t>(tensor, sizes);
        } else if (dtype == torch::kChar) {
          return tensor_to_ruby_array_for_type<int8_t>(tensor, sizes);
        } else if (dtype == torch::kShort) {
          return tensor_to_ruby_array_for_type<int16_t>(tensor, sizes);
        } else if (dtype == torch::kInt) {
          return tensor_to_ruby_array_for_type<int32_t>(tensor, sizes);
        } else if (dtype == torch::kLong) {
          return tensor_to_ruby_array_for_type<int64_t>(tensor, sizes);
        } else if (dtype == torch::kFloat) {
          return tensor_to_ruby_array_for_type<float>(tensor, sizes);
        } else if (dtype == torch::kDouble) {
          return tensor_to_ruby_array_for_type<double>(tensor, sizes);
        } else if (dtype == torch::kBool) {
          return tensor_to_ruby_array_for_type<bool>(tensor, sizes);
        } else if (dtype == torch::kComplexFloat) {
          return tensor_to_ruby_array_for_type<c10::complex<float>>(tensor, sizes);
        } else if (dtype == torch::kComplexDouble) {
          return tensor_to_ruby_array_for_type<c10::complex<double>>(tensor, sizes);
        } else {
          throw std::runtime_error("Unsupported type");
        }
      })
    .define_method(
      "_to",
      [](Tensor& self, torch::Device& device, int dtype, bool non_blocking, bool copy) {
        return self.to(device, (torch::ScalarType) dtype, non_blocking, copy);
      });

  rb_cTensorOptions
    .define_method(
      "dtype",
      [](torch::TensorOptions& self, int dtype) {
        return self.dtype((torch::ScalarType) dtype);
      })
    .define_method(
      "layout",
      [](torch::TensorOptions& self, const std::string& layout) {
        torch::Layout l;
        if (layout == "strided") {
          l = torch::kStrided;
        } else if (layout == "sparse") {
          l = torch::kSparse;
          throw std::runtime_error("Sparse layout not supported yet");
        } else {
          throw std::runtime_error("Unsupported layout: " + layout);
        }
        return self.layout(l);
      })
    .define_method(
      "device",
      [](torch::TensorOptions& self, const std::string& device) {
        torch::Device d(device);
        return self.device(d);
      })
    .define_method(
      "requires_grad",
      [](torch::TensorOptions& self, bool requires_grad) {
        return self.requires_grad(requires_grad);
      });
}
