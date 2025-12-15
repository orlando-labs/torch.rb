module Torch
  module NN
    module Utils
      private

      def _add_forward_pre_hook(mod, key, &block)
        hooks = mod.instance_variable_get(:@_nn_utils_forward_pre_hooks) || {}
        return if hooks.key?(key)

        hooks = hooks.dup
        hooks[key] = block
        mod.instance_variable_set(:@_nn_utils_forward_pre_hooks, hooks)

        unless mod.instance_variable_defined?(:@_nn_utils_original_forward)
          mod.instance_variable_set(:@_nn_utils_original_forward, mod.method(:forward))
        end

        original = mod.instance_variable_get(:@_nn_utils_original_forward)
        mod.define_singleton_method(:forward) do |*args, **kwargs|
          (instance_variable_get(:@_nn_utils_forward_pre_hooks) || {}).each_value { |h| h.call(self) }
          original.bind(self).call(*args, **kwargs)
        end
      end

      def _remove_forward_pre_hook(mod, key)
        hooks = mod.instance_variable_get(:@_nn_utils_forward_pre_hooks) || {}
        hooks.delete(key)
        mod.instance_variable_set(:@_nn_utils_forward_pre_hooks, hooks)

        if hooks.empty?
          original = mod.instance_variable_get(:@_nn_utils_original_forward)
          if original
            mod.define_singleton_method(:forward) do |*args, **kwargs|
              original.bind(self).call(*args, **kwargs)
            end
          end
        end
      end
    end
  end
end
