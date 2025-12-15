module Torch
  module NN
    module Utils
      class SpectralNorm
        attr_reader :name, :dim, :n_power_iterations, :eps

        def initialize(name, n_power_iterations, dim, eps)
          @name = name
          @dim = dim
          @n_power_iterations = n_power_iterations
          @eps = eps
        end
      end

      def spectral_norm(module_obj, name: "weight", n_power_iterations: 1, eps: 1e-12, dim: nil)
        dim ||= spectral_norm_dim(module_obj)
        if module_obj.instance_variable_defined?(:"@#{name}_orig")
          raise RuntimeError, "Cannot register two spectral_norm hooks on the same parameter #{name}"
        end

        weight = module_obj.public_send(name)
        raise ArgumentError, "`spectral_norm` expects a `#{name}` parameter" if weight.nil?

        weight_orig = Torch::NN::Parameter.new(weight.detach, requires_grad: weight.requires_grad)
        module_obj.instance_variable_get(:@parameters)&.delete(name.to_s)
        module_obj.instance_variable_set(:"@#{name}_orig", weight_orig)
        module_obj.instance_variable_set(:"@#{name}", weight_orig.data)

        weight_mat = _spectral_norm_reshape_weight(weight_orig, dim)
        h = weight_mat.size(0)
        w = weight_mat.size(1)
        u = Torch::NN::F.normalize(Torch.randn(h, device: weight.device, dtype: weight.dtype), dim: 0, eps: eps)
        v = Torch::NN::F.normalize(Torch.randn(w, device: weight.device, dtype: weight.dtype), dim: 0, eps: eps)

        module_obj.register_buffer("#{name}_u", u)
        module_obj.register_buffer("#{name}_v", v)

        opts = module_obj.instance_variable_get(:@_spectral_norm_opts) || {}
        opts = opts.dup
        opts[name] = SpectralNorm.new(name, n_power_iterations, dim, eps)
        module_obj.instance_variable_set(:@_spectral_norm_opts, opts)

        module_obj.instance_variable_set(:"@#{name}", _spectral_norm_compute_weight(module_obj, name, do_power_iteration: true))

        _add_forward_pre_hook(module_obj, :"spectral_norm_#{name}") do |mod|
          weight_sn = _spectral_norm_compute_weight(mod, name, do_power_iteration: mod.training)
          mod.instance_variable_set(:"@#{name}", weight_sn)
        end

        module_obj
      end

      def remove_spectral_norm(module_obj, name: "weight")
        opts = module_obj.instance_variable_get(:@_spectral_norm_opts) || {}
        unless module_obj.instance_variable_defined?(:"@#{name}_orig")
          raise ArgumentError, "spectral_norm of '#{name}' not found in #{module_obj}"
        end
        weight = _spectral_norm_compute_weight(module_obj, name, do_power_iteration: false)
        orig = module_obj.instance_variable_get(:"@#{name}_orig")

        module_obj.instance_variable_set(:"@#{name}", Torch::NN::Parameter.new(weight.detach, requires_grad: orig.requires_grad))
        module_obj.instance_variable_get(:@parameters)&.delete("#{name}_orig")
        module_obj.instance_variable_set(:"@#{name}_orig", nil)

        buffers = module_obj.instance_variable_get(:@buffers)
        buffers.delete("#{name}_u") if buffers
        buffers.delete("#{name}_v") if buffers
        module_obj.instance_variable_set(:"@#{name}_u", nil)
        module_obj.instance_variable_set(:"@#{name}_v", nil)

        opts.delete(name)
        module_obj.instance_variable_set(:@_spectral_norm_opts, opts)
        _remove_forward_pre_hook(module_obj, :"spectral_norm_#{name}")
        module_obj
      end

      private

      def spectral_norm_dim(module_obj)
        if module_obj.is_a?(Torch::NN::ConvTranspose1d) ||
            module_obj.is_a?(Torch::NN::ConvTranspose2d) ||
            module_obj.is_a?(Torch::NN::ConvTranspose3d)
          1
        else
          0
        end
      rescue NameError
        0
      end

      def _spectral_norm_reshape_weight(weight, dim)
        weight_mat = weight
        dim = weight.dim + dim if dim < 0
        if dim != 0
          perm = [dim]
          perm += (0...weight.dim).to_a.reject { |d| d == dim }
          weight_mat = weight_mat.permute(*perm)
        end
        height = weight_mat.size(0)
        weight_mat.view(height, -1)
      end

      def _spectral_norm_compute_weight(module_obj, name, do_power_iteration:)
        opts = module_obj.instance_variable_get(:@_spectral_norm_opts) || {}
        config = opts[name]
        raise ArgumentError, "No spectral_norm config found for #{name}" unless config

        weight = module_obj.instance_variable_get(:"@#{name}_orig")
        u = module_obj.instance_variable_get(:"@#{name}_u")
        v = module_obj.instance_variable_get(:"@#{name}_v")
        weight_mat = _spectral_norm_reshape_weight(weight, config.dim)

        if do_power_iteration
          Torch.no_grad do
            config.n_power_iterations.times do
              v.copy!(Torch::NN::F.normalize(Torch.matmul(weight_mat.transpose(0, 1), u), dim: 0, eps: config.eps))
              u.copy!(Torch::NN::F.normalize(Torch.matmul(weight_mat, v), dim: 0, eps: config.eps))
            end
          end
        end

        sigma = Torch.dot(u, Torch.matmul(weight_mat, v))
        weight_mat = weight_mat / sigma
        weight_mat.view_as(weight)
      end
    end
  end
end
