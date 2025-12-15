module Torch
  module NN
    module Utils
      def fuse_conv_bn_eval(conv, bn, transpose: false)
        raise ArgumentError, "Fusion only for eval!" if conv.training || bn.training

        fused_conv = conv.deep_dup
        bn_eps = bn.instance_variable_get(:@eps)
        fused_w, fused_b = fuse_conv_bn_weights(
          fused_conv.weight,
          fused_conv.bias,
          bn.running_mean,
          bn.running_var,
          bn_eps,
          bn.weight,
          bn.bias,
          transpose: transpose
        )
        fused_conv.instance_variable_set(:@weight, fused_w)
        fused_conv.instance_variable_set(:@bias, fused_b)
        fused_conv
      end

      def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose: false)
        conv_weight_dtype = conv_w.dtype
        conv_bias_dtype = conv_b ? conv_b.dtype : conv_weight_dtype

        conv_b = Torch.zeros_like(bn_rm) if conv_b.nil?
        bn_w = Torch.ones_like(bn_rm) if bn_w.nil?
        bn_b = Torch.zeros_like(bn_rm) if bn_b.nil?
        bn_var_rsqrt = Torch.rsqrt(bn_rv + bn_eps)

        shape =
          if transpose
            [1, -1] + [1] * (conv_w.dim - 2)
          else
            [-1, 1] + [1] * (conv_w.dim - 2)
          end

        fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(*shape)).to(dtype: conv_weight_dtype)
        fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(dtype: conv_bias_dtype)

        [
          Torch::NN::Parameter.new(fused_conv_w, requires_grad: conv_w.requires_grad),
          Torch::NN::Parameter.new(fused_conv_b, requires_grad: conv_b.requires_grad)
        ]
      end

      def fuse_linear_bn_eval(linear, bn)
        raise ArgumentError, "Fusion only for eval!" if linear.training || bn.training

        fused_linear = linear.deep_dup
        bn_eps = bn.instance_variable_get(:@eps)
        num_features = bn.instance_variable_get(:@num_features)
        unless fused_linear.out_features == num_features || num_features == 1
          raise ArgumentError, "To fuse, linear.out_features == bn.num_features or bn.num_features == 1"
        end

        fused_w, fused_b = fuse_linear_bn_weights(
          fused_linear.weight,
          fused_linear.bias,
          bn.running_mean,
          bn.running_var,
          bn_eps,
          bn.weight,
          bn.bias
        )
        fused_linear.instance_variable_set(:@weight, fused_w)
        fused_linear.instance_variable_set(:@bias, fused_b)
        fused_linear
      end

      def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b)
        linear_weight_dtype = linear_w.dtype
        linear_bias_dtype = linear_b ? linear_b.dtype : linear_weight_dtype
        linear_b = Torch.zeros_like(bn_rm) if linear_b.nil?

        bn_scale = bn_w * Torch.rsqrt(bn_rv + bn_eps)
        fused_w = linear_w * bn_scale.unsqueeze(-1).to(dtype: linear_weight_dtype)
        fused_b = ((linear_b - bn_rm) * bn_scale + bn_b).to(dtype: linear_bias_dtype)

        [
          Torch::NN::Parameter.new(fused_w, requires_grad: linear_w.requires_grad),
          Torch::NN::Parameter.new(fused_b, requires_grad: linear_b.requires_grad)
        ]
      end
    end
  end
end
