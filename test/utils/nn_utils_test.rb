require_relative "../test_helper"

class NNUtilsTest < Minitest::Test
  def test_clip_grad_norm_and_value
    p1 = Torch::NN::Parameter.new(Torch.tensor([3.0]))
    p2 = Torch::NN::Parameter.new(Torch.tensor([4.0]))
    p1.grad = Torch.tensor([3.0])
    p2.grad = Torch.tensor([4.0])

    total = Torch::NN::Utils.clip_grad_norm!([p1, p2], 1.0)
    assert_in_delta 5.0, total.item
    assert_tensor [0.6], p1.grad
    assert_tensor [0.8], p2.grad

    Torch::NN::Utils.clip_grad_value!([p1, p2], 0.5)
    assert_tensor [0.5], p1.grad
    assert_tensor [0.5], p2.grad
  end

  def test_parameters_vector_conversion
    p1 = Torch.tensor([[1.0, 2.0]], requires_grad: true)
    p2 = Torch.tensor([3.0], requires_grad: true)
    vec = Torch::NN::Utils.parameters_to_vector([p1, p2])
    assert_tensor [1.0, 2.0, 3.0], vec

    target = Torch.tensor([-1.0, -2.0, -3.0])
    Torch::NN::Utils.vector_to_parameters(target, [p1, p2])
    assert_tensor [[-1.0, -2.0]], p1
    assert_tensor [-3.0], p2
  end

  def test_pad_pack_sequence_roundtrip
    seq1 = Torch.tensor([[1, 2], [3, 4]])
    seq2 = Torch.tensor([[5, 6]])
    padded = Torch::NN::Utils::RNN.pad_sequence([seq1, seq2], batch_first: true)
    assert_tensor [[[1, 2], [3, 4]], [[5, 6], [0, 0]]], padded

    lengths = Torch.tensor([2, 1], dtype: :long)
    packed = Torch::NN::Utils::RNN.pack_padded_sequence(padded, lengths, batch_first: true, enforce_sorted: false)
    unpacked, unpacked_lengths = Torch::NN::Utils::RNN.pad_packed_sequence(packed, batch_first: true)
    assert_tensor [[[1, 2], [3, 4]], [[5, 6], [0, 0]]], unpacked
    assert_tensor [2, 1], unpacked_lengths
  end

  def test_pack_and_unpack_sequence
    seqs = [
      Torch.tensor([1, 2, 3]),
      Torch.tensor([4, 5])
    ]
    packed = Torch::NN::Utils::RNN.pack_sequence(seqs, enforce_sorted: true)
    unpacked = Torch::NN::Utils::RNN.unpack_sequence(packed)
    assert_equal seqs.map(&:to_a), unpacked.map(&:to_a)
  end
end
