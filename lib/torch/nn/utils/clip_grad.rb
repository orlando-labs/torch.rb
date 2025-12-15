module Torch
  module NN
    module Utils
      def clip_grad_norm!(parameters, max_norm, norm_type: 2.0, error_if_nonfinite: false)
        _clip_grad_norm(parameters, max_norm, norm_type, error_if_nonfinite)
      end

      def clip_grad_norm(parameters, max_norm, norm_type: 2.0, error_if_nonfinite: false)
        clip_grad_norm!(parameters, max_norm, norm_type: norm_type, error_if_nonfinite: error_if_nonfinite)
      end

      def clip_grads_with_norm!(parameters, max_norm, total_norm, eps: 1e-6)
        Torch.no_grad do
          clip_coef = max_norm.to_f / (total_norm + eps)
          clip_coef = clip_coef.clamp(max: 1.0)
          _tensor_list(parameters).map(&:grad).compact.each do |grad|
            grad.mul!(clip_coef.to(grad.device))
          end
        end
      end

      def clip_grad_value!(parameters, clip_value)
        _clip_grad_value(parameters, clip_value)
      end

      def get_total_norm(tensors, norm_type: 2.0, error_if_nonfinite: false)
        tensors = _tensor_list(tensors)
        return Torch.tensor(0.0) if tensors.empty?

        norm_type = norm_type.to_f
        total_norm =
          if norm_type == Float::INFINITY
            norms = tensors.map { |g| g.abs.max }
            Torch.stack(norms).max
          else
            norms = tensors.map { |g| g.norm(norm_type) }
            Torch.stack(norms).norm(norm_type)
          end

        if error_if_nonfinite
          val = total_norm.item
          unless val.finite?
            raise RuntimeError, "The total norm of order #{norm_type} for gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite: false`"
          end
        end

        total_norm
      end

      private

      def _tensor_list(obj)
        if obj.is_a?(Torch::Tensor)
          [obj]
        elsif obj.respond_to?(:to_a)
          obj.to_a
        else
          Array(obj)
        end
      end
    end
  end
end
