import torch
from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes

# class Buffer(torch.nn.Module):
#     def __init__(self, model, params):
#         super().__init__()
#         self.params = params
#         self.model = model
#         self.cuda = self.params.cuda
#         self.current_index = 0
#         self.n_seen_so_far = 0
#         self.device = "cuda" if self.params.cuda else "cpu"

#         # define buffer
#         buffer_size = params.mem_size
#         print('buffer has %d slots' % buffer_size)
#         input_size = input_size_match[params.data]
#         buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
#         buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

#         # registering as buffer allows us to save the object using `torch.save`
#         self.register_buffer('buffer_img', buffer_img)
#         self.register_buffer('buffer_label', buffer_label)

#         # define update and retrieve method
#         self.update_method = name_match.update_methods[params.update](params)
#         self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

#         if self.params.buffer_tracker:
#             self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

#     def update(self, x, y,**kwargs):
#         return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


#     def retrieve(self, **kwargs):
#         return self.retrieve_method.retrieve(buffer=self, **kwargs)
    
class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        # 定义 buffer
        buffer_size = params.mem_size
        input_size = input_size_match[params.data]
        output_size = self.detect_output_size(input_size)  # 假设你已经实现了这个方法
        print('buffer has %d slots' % buffer_size)
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        buffer_logits = maybe_cuda(torch.FloatTensor(buffer_size, output_size).fill_(0))  # 新添加的logits buffer

        # 注册 buffer，以便可以使用 `torch.save` 保存对象
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('buffer_logits', buffer_logits)  # 注册新的 buffer

        # 定义 update 和 retrieve 方法
        self.update_method = name_match.update_methods[params.update](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    def update(self, x, y, logits=None, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, logits=logits, **kwargs)

    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def detect_output_size(self, input_size):
        input_sample = torch.randn((1, *input_size))
        if self.cuda:
            input_sample = input_sample.to(self.device)
        with torch.no_grad():
            if self.params.agent == 'SUPER':
                output = self.model(input_sample, time=0)
            else:
                output = self.model(input_sample)
        return output.shape[1]


# class Second_Buffer(torch.nn.Module):
#     def __init__(self, model, params):
#         super().__init__()
#         self.params = params
#         self.model = model
#         self.cuda = self.params.cuda
#         self.current_index = 0
#         self.n_seen_so_far = 0
#         self.device = "cuda" if self.params.cuda else "cpu"

#         # define buffer
#         buffer_size = params.mem_size
#         print('buffer has %d slots' % buffer_size)
#         input_size = input_size_match[params.data]
#         buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
#         buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

#         # registering as buffer allows us to save the object using `torch.save`
#         self.register_buffer('buffer_img', buffer_img)
#         self.register_buffer('buffer_label', buffer_label)

#         # define update and retrieve method
#         self.update_method = name_match.update_methods[params.update2](params)
#         self.retrieve_method = name_match.retrieve_methods[params.retrieve2](params)

#         if self.params.buffer_tracker:
#             self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

#     def update(self, x, y,**kwargs):
#         return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


#     def retrieve(self, **kwargs):
#         return self.retrieve_method.retrieve(buffer=self, **kwargs)


class Second_Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        # 定义 buffer
        buffer_size = params.mem_size
        output_size = self.detect_output_size()  
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        buffer_logits = maybe_cuda(torch.FloatTensor(buffer_size, output_size).fill_(0))  

        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('buffer_logits', buffer_logits)  

        # 定义 update 和 retrieve 方法
        self.update_method = name_match.update_methods[params.update2](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve2](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    def update(self, x, y, logits=None, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, logits=logits, **kwargs)

    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def detect_output_size(self):
        input_sample = torch.randn((1, *input_size))
        if self.cuda:
            input_sample = input_sample.to(self.device)
        with torch.no_grad():
            output = self.model(input_sample)
        return output.shape[1]