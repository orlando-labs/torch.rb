module Torch
  module NN
    module Utils
      class WeightNorm
        attr_reader :name, :dim

        def initialize(name, dim)
          @name = name
          @dim = dim.nil? ? -1 : dim
        end
      end

      def weight_norm(module_obj, name: "weight", dim: 0)
        dim = -1 if dim.nil?
        if module_obj.instance_variable_defined?(:"@#{name}_g")
          raise RuntimeError, "Cannot register two weight_norm hooks on the same parameter #{name}"
        end

        weight = module_obj.public_send(name)
        if weight.nil?
          raise ArgumentError, "The module passed to `weight_norm` needs a `#{name}` parameter"
        end

        module_obj.instance_variable_get(:@parameters)&.delete(name.to_s)

        g = Torch::NN::Parameter.new(Torch.norm_except_dim(weight, 2, dim).detach, requires_grad: weight.requires_grad)
        v = Torch::NN::Parameter.new(weight.detach, requires_grad: weight.requires_grad)

        module_obj.instance_variable_set(:"@#{name}_g", g)
        module_obj.instance_variable_set(:"@#{name}_v", v)
        dims = module_obj.instance_variable_get(:@_weight_norm_dims) || {}
        dims = dims.dup
        dims[name] = dim
        module_obj.instance_variable_set(:@_weight_norm_dims, dims)

        module_obj.instance_variable_set(:"@#{name}", _weight_norm_compute_weight(v, g, dim))

        _add_forward_pre_hook(module_obj, :"weight_norm_#{name}") do |mod|
          g_param = mod.instance_variable_get(:"@#{name}_g")
          v_param = mod.instance_variable_get(:"@#{name}_v")
          mod.instance_variable_set(:"@#{name}", _weight_norm_compute_weight(v_param, g_param, dim))
        end

        module_obj
      end

      def remove_weight_norm(module_obj, name: "weight")
        dims = module_obj.instance_variable_get(:@_weight_norm_dims) || {}
        unless module_obj.instance_variable_defined?(:"@#{name}_g")
          raise ArgumentError, "weight_norm of '#{name}' not found in #{module_obj}"
        end
        dim = dims[name] || 0

        g = module_obj.instance_variable_get(:"@#{name}_g")
        v = module_obj.instance_variable_get(:"@#{name}_v")
        weight = _weight_norm_compute_weight(v, g, dim)

        module_obj.instance_variable_set(:"@#{name}", Torch::NN::Parameter.new(weight.detach, requires_grad: g.requires_grad))
        module_obj.instance_variable_set(:"@#{name}_g", nil)
        module_obj.instance_variable_set(:"@#{name}_v", nil)
        dims.delete(name)
        module_obj.instance_variable_set(:@_weight_norm_dims, dims)

        _remove_forward_pre_hook(module_obj, :"weight_norm_#{name}")
        module_obj
      end

      private

      def _weight_norm_compute_weight(v, g, dim)
        dim = v.dim + dim if dim < 0
        norm = Torch.norm_except_dim(v, 2, dim)
        norm = norm.clamp_min(1e-12) if norm.respond_to?(:clamp_min)
        view_shape = Array.new(v.dim, 1)
        view_shape[dim] = norm.shape[0]
        scale = g.view(*view_shape) / norm.view(*view_shape)
        v * scale
      end
    end
  end
end
