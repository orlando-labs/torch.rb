module Torch
  module NN
    module Utils
      extend self

      def _single(value)
        _ntuple(1, value)
      end

      def _pair(value)
        _ntuple(2, value)
      end

      def _triple(value)
        _ntuple(3, value)
      end

      def _quadrupal(value)
        _ntuple(4, value)
      end

      def _ntuple(n, value)
        value.is_a?(Array) ? value : [value] * n
      end

      def _clones(mod, n)
        ModuleList.new(n.times.map { mod.deep_dup })
      end

      def _activation_fn(activation)
        case activation.to_sym
        when :relu then F.method(:relu)
        when :gelu then F.method(:gelu)
        else raise ArgumentError, "Activation should be relu/gelu, not `#{activation}`"
        end
      end
    end
  end
end

require_relative "utils/hooks"
require_relative "utils/clip_grad"
require_relative "utils/convert_parameters"
require_relative "utils/fusion"
require_relative "utils/init"
require_relative "utils/memory_format"
require_relative "utils/parametrize"
require_relative "utils/rnn"
require_relative "utils/stateless"
require_relative "utils/weight_norm"
require_relative "utils/spectral_norm"
