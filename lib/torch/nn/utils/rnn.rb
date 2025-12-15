module Torch
  module NN
    module Utils
      class PackedSequence
        attr_reader :data, :batch_sizes, :sorted_indices, :unsorted_indices

        def initialize(data, batch_sizes, sorted_indices = nil, unsorted_indices = nil)
          unsorted_indices ||= RNN.invert_permutation(sorted_indices)
          @data = data
          @batch_sizes = batch_sizes
          @sorted_indices = sorted_indices
          @unsorted_indices = unsorted_indices
        end

        def pin_memory
          PackedSequence.new(
            data.pin_memory,
            batch_sizes,
            sorted_indices&.pin_memory,
            unsorted_indices&.pin_memory
          )
        end

        def to(*args, **kwargs)
          new_data = data.to(*args, **kwargs)
          return self if new_data.equal?(data)

          kwargs = kwargs.reject { |k, _| k == :device || k == "device" || k == :dtype || k == "dtype" }
          device = new_data.device
          new_sorted = sorted_indices&.to(device, **kwargs)
          new_unsorted = unsorted_indices&.to(device, **kwargs)
          PackedSequence.new(new_data, batch_sizes, new_sorted, new_unsorted)
        end

        def cuda(*args, **kwargs)
          to(device: "cuda", **kwargs)
        end

        def cpu(*args, **kwargs)
          to(device: "cpu", **kwargs)
        end

        def float
          to(dtype: :float32)
        end

        def double
          to(dtype: :float64)
        end

        def long
          to(dtype: :int64)
        end

        def is_cuda
          data.device.type == "cuda"
        end

        def is_pinned
          data.respond_to?(:is_pinned) ? data.is_pinned : false
        end
      end

      module RNN
        extend self

        PackedSequence = Utils::PackedSequence

        def invert_permutation(permutation)
          return nil if permutation.nil?
          Utils._invert_permutation(permutation)
        end

        def pack_padded_sequence(input, lengths, batch_first: false, enforce_sorted: true)
          lengths =
            if lengths.is_a?(Torch::Tensor)
              lengths
            else
              Torch.tensor(lengths, dtype: :long, device: "cpu")
            end
          Utils._pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
        end

        def pad_packed_sequence(sequence, batch_first: false, padding_value: 0.0, total_length: nil)
          Utils._pad_packed_sequence(sequence, batch_first, padding_value, total_length)
        end

        def pad_sequence(sequences, batch_first: false, padding_value: 0.0, padding_side: "right")
          seqs =
            if sequences.is_a?(Torch::Tensor)
              sequences.unbind(0)
            else
              sequences.to_a
            end
          Utils._pad_sequence(seqs, batch_first, padding_value, padding_side)
        end

        def unpad_sequence(padded_sequences, lengths, batch_first: false)
          padded =
            if batch_first
              padded_sequences
            else
              padded_sequences.transpose(0, 1)
            end

          sequences = []
          lengths.to_a.each_with_index do |length, idx|
            len = length.to_i
            sequences << padded[idx, 0...len]
          end

          sequences
        end

        def pack_sequence(sequences, enforce_sorted: true)
          Utils._pack_sequence(sequences, enforce_sorted)
        end

        def unpack_sequence(packed_sequences)
          padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first: true)
          unpad_sequence(padded_sequences, lengths, batch_first: true)
        end
      end
    end
  end
end
