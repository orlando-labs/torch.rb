module Torch
  module NN
    module Utils
      def skip_init(module_cls, *args, **kwargs)
        unless module_cls <= Torch::NN::Module
          raise RuntimeError, "Expected a Module; got #{module_cls}"
        end

        device = kwargs.delete(:device) { "cpu" }
        module_obj = module_cls.new(*args, **kwargs)

        Torch.no_grad do
          module_obj.parameters.each do |param|
            param.copy!(Torch.empty_like(param, device: device))
          end

          module_obj.buffers.each do |buf|
            next if buf.nil?
            buf.copy!(Torch.empty_like(buf, device: device))
          end
        end

        module_obj.to(device)
      end
    end
  end
end
