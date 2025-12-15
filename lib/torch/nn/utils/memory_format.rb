module Torch
  module NN
    module Utils
      def convert_conv2d_weight_memory_format(module_obj, memory_format)
        if module_obj.is_a?(Torch::NN::Conv2d) || (defined?(Torch::NN::ConvTranspose2d) && module_obj.is_a?(Torch::NN::ConvTranspose2d))
          weight = module_obj.weight
          weight_data = weight.detach.clone(memory_format: memory_format)
          weight_data = weight_data.resize!(weight_data.size, memory_format: memory_format)
          module_obj.instance_variable_set(:@weight, Torch::NN::Parameter.new(weight_data, requires_grad: weight.requires_grad))
        end

        module_obj.children.each do |child|
          convert_conv2d_weight_memory_format(child, memory_format)
        end

        module_obj
      end

      def convert_conv3d_weight_memory_format(module_obj, memory_format)
        if module_obj.is_a?(Torch::NN::Conv3d) || (defined?(Torch::NN::ConvTranspose3d) && module_obj.is_a?(Torch::NN::ConvTranspose3d))
          weight = module_obj.weight
          weight_data = weight.detach.clone(memory_format: memory_format)
          weight_data = weight_data.resize!(weight_data.size, memory_format: memory_format)
          module_obj.instance_variable_set(:@weight, Torch::NN::Parameter.new(weight_data, requires_grad: weight.requires_grad))
        end

        module_obj.children.each do |child|
          convert_conv3d_weight_memory_format(child, memory_format)
        end

        module_obj
      end
    end
  end
end
