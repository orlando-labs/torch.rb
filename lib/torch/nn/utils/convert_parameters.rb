module Torch
  module NN
    module Utils
      def parameters_to_vector(parameters)
        _parameters_to_vector(parameters)
      end

      def vector_to_parameters(vec, parameters)
        _vector_to_parameters(vec, parameters)
      end

      private

      def _check_param_device(param, old_device)
        device = param.device
        if old_device && device != old_device
          raise TypeError, "Found two parameters on different devices, this is currently not supported."
        end
        device
      end
    end
  end
end
