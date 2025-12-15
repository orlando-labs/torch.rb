require_relative "test_helper"

class FlatDataTest < Minitest::Test
  def test_non_contiguous_to_a
    tensor = Torch.arange(6, dtype: :int64).reshape([2, 3]).transpose(0, 1)
    assert_equal [[0, 3], [1, 4], [2, 5]], tensor.to_a
  end

  def test_bool_to_a
    tensor = Torch.tensor([[true, false], [false, true]], dtype: :bool)
    assert_equal [[true, false], [false, true]], tensor.to_a
  end

  def test_complex_to_a
    tensor = Torch.tensor([1 + 2i, 3 - 4i], dtype: :complex128)
    assert_equal [1 + 2i, 3 - 4i], tensor.to_a
  end

  def test_conjugate_view_to_a
    tensor = Torch.tensor([1 + 2i, 3 - 4i], dtype: :complex128).conj
    assert_equal [1 - 2i, 3 + 4i], tensor.to_a
  end

  def test_uint8_to_a
    tensor = Torch.tensor([0, 127, 255], dtype: :uint8)
    assert_equal [0, 127, 255], tensor.to_a
  end
end
