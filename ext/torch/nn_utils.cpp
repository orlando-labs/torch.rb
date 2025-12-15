#include <torch/torch.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/rnn.h>

#include <rice/rice.hpp>

#include <optional>
#include <vector>

namespace {

VALUE packed_sequence_class() {
  VALUE rb_mTorch = rb_const_get(rb_cObject, rb_intern("Torch"));
  VALUE rb_mNN = rb_const_get(rb_mTorch, rb_intern("NN"));
  VALUE rb_mUtils = rb_const_get(rb_mNN, rb_intern("Utils"));
  return rb_const_get(rb_mUtils, rb_intern("PackedSequence"));
}

VALUE make_packed_sequence(const torch::nn::utils::rnn::PackedSequence& ps) {
  VALUE rb_cPackedSequence = packed_sequence_class();
  VALUE args[4];
  args[0] = Rice::detail::To_Ruby<torch::Tensor>().convert(ps.data());
  args[1] = Rice::detail::To_Ruby<torch::Tensor>().convert(ps.batch_sizes());
  args[2] = ps.sorted_indices().defined()
      ? Rice::detail::To_Ruby<torch::Tensor>().convert(ps.sorted_indices())
      : Qnil;
  args[3] = ps.unsorted_indices().defined()
      ? Rice::detail::To_Ruby<torch::Tensor>().convert(ps.unsorted_indices())
      : Qnil;
  return rb_class_new_instance(4, args, rb_cPackedSequence);
}

std::vector<torch::Tensor> tensor_list_from(VALUE obj) {
  std::vector<torch::Tensor> tensors;

  if (RB_TYPE_P(obj, T_ARRAY)) {
    auto size = RARRAY_LEN(obj);
    tensors.reserve(size);
    for (long i = 0; i < size; i++) {
      VALUE entry = rb_ary_entry(obj, i);
      tensors.push_back(Rice::detail::From_Ruby<torch::Tensor>().convert(entry));
    }
    return tensors;
  }

  tensors.push_back(Rice::detail::From_Ruby<torch::Tensor>().convert(obj));
  return tensors;
}

VALUE utils_clip_grad_norm(int argc, VALUE* argv, VALUE self) {
  if (argc < 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2+)", argc);
  }

  auto parameters = tensor_list_from(argv[0]);
  double max_norm = NUM2DBL(argv[1]);
  double norm_type = argc > 2 ? NUM2DBL(argv[2]) : 2.0;
  bool error_if_nonfinite = argc > 3 ? RTEST(argv[3]) : false;

  double total_norm = torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite);
  auto tensor_norm = torch::tensor(total_norm);
  return Rice::detail::To_Ruby<torch::Tensor>().convert(tensor_norm);
}

VALUE utils_clip_grad_value(int argc, VALUE* argv, VALUE self) {
  if (argc < 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2)", argc);
  }

  auto parameters = tensor_list_from(argv[0]);
  double clip_value = NUM2DBL(argv[1]);

  torch::nn::utils::clip_grad_value_(parameters, clip_value);
  return Qnil;
}

VALUE utils_parameters_to_vector(VALUE self, VALUE parameters) {
  auto params = tensor_list_from(parameters);
  auto result = torch::nn::utils::parameters_to_vector(params);
  return Rice::detail::To_Ruby<torch::Tensor>().convert(result);
}

VALUE utils_vector_to_parameters(VALUE self, VALUE vec, VALUE parameters) {
  auto vec_tensor = Rice::detail::From_Ruby<torch::Tensor>().convert(vec);
  auto params = tensor_list_from(parameters);
  torch::nn::utils::vector_to_parameters(vec_tensor, params);
  return Qnil;
}

VALUE utils_invert_permutation(VALUE self, VALUE permutation) {
  auto perm = Rice::detail::From_Ruby<torch::Tensor>().convert(permutation);
  auto result = torch::nn::utils::rnn::invert_permutation(perm);
  return Rice::detail::To_Ruby<torch::Tensor>().convert(result);
}

VALUE utils_pack_padded_sequence(int argc, VALUE* argv, VALUE self) {
  if (argc < 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2+)", argc);
  }
  auto input = Rice::detail::From_Ruby<torch::Tensor>().convert(argv[0]);
  auto lengths = Rice::detail::From_Ruby<torch::Tensor>().convert(argv[1]);
  bool batch_first = argc > 2 ? RTEST(argv[2]) : false;
  bool enforce_sorted = argc > 3 ? RTEST(argv[3]) : true;

  auto ps = torch::nn::utils::rnn::pack_padded_sequence(input, lengths, batch_first, enforce_sorted);
  return make_packed_sequence(ps);
}

VALUE utils_pad_packed_sequence(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1+)", argc);
  }
  VALUE seq = argv[0];
  VALUE rb_data = rb_funcall(seq, rb_intern("data"), 0);
  VALUE rb_batch_sizes = rb_funcall(seq, rb_intern("batch_sizes"), 0);
  VALUE rb_sorted_indices = rb_funcall(seq, rb_intern("sorted_indices"), 0);
  VALUE rb_unsorted_indices = rb_funcall(seq, rb_intern("unsorted_indices"), 0);

  auto data = Rice::detail::From_Ruby<torch::Tensor>().convert(rb_data);
  auto batch_sizes = Rice::detail::From_Ruby<torch::Tensor>().convert(rb_batch_sizes);
  torch::Tensor sorted_indices;
  torch::Tensor unsorted_indices;
  if (!NIL_P(rb_sorted_indices)) {
    sorted_indices = Rice::detail::From_Ruby<torch::Tensor>().convert(rb_sorted_indices);
  }
  if (!NIL_P(rb_unsorted_indices)) {
    unsorted_indices = Rice::detail::From_Ruby<torch::Tensor>().convert(rb_unsorted_indices);
  }
  torch::nn::utils::rnn::PackedSequence ps(data, batch_sizes, sorted_indices, unsorted_indices);

  bool batch_first = argc > 1 ? RTEST(argv[1]) : false;
  double padding_value = argc > 2 ? NUM2DBL(argv[2]) : 0.0;
  std::optional<int64_t> total_length = std::nullopt;
  if (argc > 3 && !NIL_P(argv[3])) {
    total_length = NUM2LONG(argv[3]);
  }

  auto result = torch::nn::utils::rnn::pad_packed_sequence(ps, batch_first, padding_value, total_length);
  VALUE ary = rb_ary_new2(2);
  rb_ary_store(ary, 0, Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(result)));
  rb_ary_store(ary, 1, Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(result)));
  return ary;
}

VALUE utils_pad_sequence(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1+)", argc);
  }
  auto sequences = tensor_list_from(argv[0]);
  bool batch_first = argc > 1 ? RTEST(argv[1]) : false;
  double padding_value = argc > 2 ? NUM2DBL(argv[2]) : 0.0;
  std::string padding_side = "right";
  if (argc > 3 && !NIL_P(argv[3])) {
    padding_side = StringValueCStr(argv[3]);
  }

  auto result = torch::nn::utils::rnn::pad_sequence(sequences, batch_first, padding_value, padding_side);
  return Rice::detail::To_Ruby<torch::Tensor>().convert(result);
}

VALUE utils_pack_sequence(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1+)", argc);
  }
  auto sequences = tensor_list_from(argv[0]);
  bool enforce_sorted = argc > 1 ? RTEST(argv[1]) : true;
  auto ps = torch::nn::utils::rnn::pack_sequence(sequences, enforce_sorted);
  return make_packed_sequence(ps);
}

} // namespace

void init_nn_utils(Rice::Module& m) {
  auto m_utils = Rice::define_module_under(m, "Utils");
  VALUE rb_mUtils = m_utils.value();

  rb_define_singleton_method(rb_mUtils, "_clip_grad_norm", RUBY_METHOD_FUNC(utils_clip_grad_norm), -1);
  rb_define_singleton_method(rb_mUtils, "_clip_grad_value", RUBY_METHOD_FUNC(utils_clip_grad_value), -1);
  rb_define_singleton_method(rb_mUtils, "_parameters_to_vector", RUBY_METHOD_FUNC(utils_parameters_to_vector), 1);
  rb_define_singleton_method(rb_mUtils, "_vector_to_parameters", RUBY_METHOD_FUNC(utils_vector_to_parameters), 2);
  rb_define_singleton_method(rb_mUtils, "_invert_permutation", RUBY_METHOD_FUNC(utils_invert_permutation), 1);
  rb_define_singleton_method(rb_mUtils, "_pack_padded_sequence", RUBY_METHOD_FUNC(utils_pack_padded_sequence), -1);
  rb_define_singleton_method(rb_mUtils, "_pad_packed_sequence", RUBY_METHOD_FUNC(utils_pad_packed_sequence), -1);
  rb_define_singleton_method(rb_mUtils, "_pad_sequence", RUBY_METHOD_FUNC(utils_pad_sequence), -1);
  rb_define_singleton_method(rb_mUtils, "_pack_sequence", RUBY_METHOD_FUNC(utils_pack_sequence), -1);
}
