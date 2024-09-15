import torch
from models.resnet import Reduced_ResNet18, SupConResNet
from models.resnet_vanilla import ResNet18
from models.resnet_hash import HashResNet18
from models.resnet_hash_dev import Reduced_HashResNet18
from models.resnet_m import Reduced_ResNet18_m, Reduced_ResNet50_m, HierarchicalCrossAttention, cosLinear
from models.resnet_sub import Reduced_ResNet18_sub, cosSubCentroids
from models.resnet_prerelu import Reduced_ResNet18_prerelu
from models.resnet_psp import Reduced_ResNet18_psp
from torchvision import transforms
import torch.nn as nn
import torch.nn.init as init


default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
    'food100': [3, 224, 224],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50],
    'places100': [3, 224, 224],
    'tinyimagenet': [3, 64, 64],
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'food100': 100,
    'mini_imagenet': 100,
    'openloris': 69,
    'places100': 100,
    'tinyimagenet': 200,
}


transforms_match = {
    'food100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'core50': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
            transforms.ToTensor()]),
    'places100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'tinyimagenet': transforms.Compose([
        transforms.ToTensor(),
        ]),
}


transforms_aug = {
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
    'cifar10': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ]),
    'core50': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=128, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ]),
    'food100': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ]),
    'places100': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ]),
    'tinyimagenet': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ]),

}


# def setup_architecture(params):
#     nclass = n_classes[params.data]
#     if params.agent in ['SCR', 'SCP']:
#         if params.data == 'mini_imagenet':
#             return SupConResNet(640, head=params.head)
#         return SupConResNet(head=params.head)
#     if params.agent == 'CNDPM':
#         from models.ndpm.ndpm import Ndpm
#         return Ndpm(params)
#     if params.data == 'cifar100':
#         return Reduced_ResNet18(nclass)
#     elif params.data == 'cifar10':
#         return Reduced_ResNet18(nclass)
#     elif params.data == 'core50':
#         model = Reduced_ResNet18(nclass)
#         model.linear = nn.Linear(2560, nclass, bias=True)
#         return model
#     elif params.data == 'mini_imagenet':
#         model = Reduced_ResNet18(nclass)
#         model.linear = nn.Linear(640, nclass, bias=True)
#         return model
#     elif params.data == 'openloris':
#         return Reduced_ResNet18(nclass)
    

def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.agent in ['SCR', 'SCP']:
        if params.data == 'mini_imagenet':
            return SupConResNet(640, head=params.head)
        return SupConResNet(head=params.head)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)
    elif params.data == 'cifar100' and params.agent == 'PCR_m':
        return Reduced_ResNet18_m(nclass)
    # elif params.data == 'cifar100' and params.agent == 'PCR_sub':
    #     return Reduced_ResNet18_sub(nclass, num_subcentroids=params.num_subcentroids)
    # elif params.data == 'mini_imagenet'and params.agent == 'PCR_sub':
    #     model = Reduced_ResNet18_sub(nclass, num_subcentroids=params.num_subcentroids)
    #     model.pcrLinear_sub = cosSubCentroids(640, nclass, params.num_subcentroids)
    #     return model
    # elif params.data == 'tinyimagenet'and params.agent == 'PCR_sub':
    #     model = Reduced_ResNet18_sub(nclass, num_subcentroids=params.num_subcentroids)
    #     model.pcrLinear_sub = cosSubCentroids(640, nclass, params.num_subcentroids)
    #     return model
    # elif params.data == 'core50'and params.agent == 'PCR_sub':
    #     model = Reduced_ResNet18_sub(nclass, num_subcentroids=params.num_subcentroids)
    #     model.pcrLinear_sub = cosSubCentroids(2560, nclass, params.num_subcentroids)
    #     return model
    # elif params.data == 'food100'and params.agent == 'PCR_sub':
    #     model = Reduced_ResNet18_sub(nclass, num_subcentroids=params.num_subcentroids)
    #     model.pcrLinear_sub = cosSubCentroids(7840, nclass, params.num_subcentroids)
    #     return model
    # elif params.data == 'places100'and params.agent == 'PCR_sub':
    #     model = Reduced_ResNet18_sub(nclass, num_subcentroids=params.num_subcentroids)
    #     model.pcrLinear_sub = cosSubCentroids(7840, nclass, params.num_subcentroids)
    #     return model
    
    elif params.data == 'cifar10' and params.agent == 'PCR_m':
        return Reduced_ResNet18_m(nclass)
    elif params.data == 'core50' and params.agent == 'PCR_m':
        model = Reduced_ResNet18_m(nclass)
        model.linear = nn.Linear(2560, nclass, bias=True)
        return model
    elif params.data == 'mini_imagenet' and params.agent == 'PCR_m':
        model = Reduced_ResNet18_m(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        return model
    elif params.data == 'openloris' and params.agent == 'PCR_m':
        return Reduced_ResNet18_m(nclass)
    
    elif params.data == 'cifar100' and params.agent == 'PCR_swav':
        return Reduced_ResNet50_m(nclass)  
    elif params.data == 'cifar10' and params.agent == 'PCR':
        return Reduced_ResNet18(nclass)
    elif params.data == 'cifar100' and params.agent == 'PCR':
        # return HashResNet18(nclass)
        return Reduced_ResNet18(nclass)
    elif params.data == 'mini_imagenet'and params.agent == 'PCR':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        model.pcrLinear = cosLinear(640, nclass)
        return model
    elif params.data == 'food100'and params.agent == 'PCR':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(7840, nclass, bias=True)
        model.pcrLinear = cosLinear(7840, nclass)
        return model
    elif params.data == 'core50' and params.agent == 'PCR':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(2560, nclass, bias=True)
        model.pcrLinear = cosLinear(2560, nclass)
        return model
    elif params.data == 'places100'and params.agent == 'PCR':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(7840, nclass, bias=True)
        model.pcrLinear = cosLinear(7840, nclass)
        return model   
    elif params.data == 'bird100'and params.agent == 'PCR':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        model.pcrLinear = cosLinear(640, nclass)
        return model
    elif params.data == 'tinyimagenet'and params.agent == 'PCR':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        model.pcrLinear = cosLinear(640, nclass)
        return model
    
    # elif params.data == 'cifar100' and params.agent == 'SuperPCR':
    #     return Reduced_ResNet18_psp(nclass)
    # elif params.data == 'mini_imagenet'and params.agent == 'SuperPCR':
    #     model = Reduced_ResNet18_psp(nclass)
    #     model.linear = nn.Linear(640, nclass, bias=True)
    #     model.pcrLinear = cosLinear(640, nclass)
    #     return model
    

    elif params.data == 'cifar100' and params.agent == 'SuperPCR':
        return Reduced_HashResNet18(nclass, params.num_tasks)
    elif params.data == 'mini_imagenet'and params.agent == 'SuperPCR':
        model = Reduced_HashResNet18(nclass, params.num_tasks)
        model.linear = nn.Linear(640, nclass, period=num_tasks)
        model.pcrLinear = cosLinear(640, nclass, period=num_tasks)
        return model

    # elif params.data == 'core50' and params.agent == 'PCR_ca':
    #     model = Reduced_ResNet18_m(nclass, memory_bank_size = params.mem_bank_size, num_subcentroids=params.num_subcentroids, gamma=params.gamma)
    #     model.embedding_dim = 2560
    #     model.prototypes = nn.Parameter(torch.torch.randn(100, params.num_subcentroids, 2560),
    #                                    requires_grad=True)
    #     init.kaiming_normal_(model.prototypes, mode='fan_out', nonlinearity='relu')
    #     model.feat_norm = nn.LayerNorm(2560)
    #     return model
    
    # elif params.data == 'food100'and params.agent == 'PCR_ca':
    #     model = Reduced_ResNet18_m(nclass, memory_bank_size = params.mem_bank_size, num_subcentroids=params.num_subcentroids, gamma=params.gamma)
    #     model.embedding_dim = 7840
    #     model.prototypes = nn.Parameter(torch.torch.randn(100, params.num_subcentroids, 7840),
    #                                    requires_grad=True)
    #     init.kaiming_normal_(model.prototypes, mode='fan_out', nonlinearity='relu')
    #     model.feat_norm = nn.LayerNorm(7840)
    #     return model
    
    # elif params.data == 'places100'and params.agent == 'PCR_ca':
    #     model = Reduced_ResNet18_m(nclass, memory_bank_size = params.mem_bank_size, num_subcentroids=params.num_subcentroids, gamma=params.gamma)
    #     model.embedding_dim = 7840
    #     model.prototypes = nn.Parameter(torch.torch.randn(100, params.num_subcentroids, 7840),
    #                                    requires_grad=True)
    #     init.kaiming_normal_(model.prototypes, mode='fan_out', nonlinearity='relu')
    #     model.feat_norm = nn.LayerNorm(7840)
    #     return model
    # elif params.data == 'mini_imagenet'and params.agent == 'PCR_ca':
    #     model = Reduced_ResNet18_m(nclass, memory_bank_size = params.mem_bank_size, num_subcentroids=params.num_subcentroids, gamma=params.gamma)
    #     # model.pcrLinear = cosLinear(640, nclass)
    #     model.embedding_dim = 640
    #     model.prototypes = nn.Parameter(torch.torch.randn(100, params.num_subcentroids, 640),
    #                                    requires_grad=True)
    #     init.kaiming_normal_(model.prototypes, mode='fan_out', nonlinearity='relu')
    #     model.feat_norm = nn.LayerNorm(640)
    #     return model
    # elif params.data == 'tinyimagenet'and params.agent == 'PCR_ca':
    #     model = Reduced_ResNet18_m(nclass, memory_bank_size = params.mem_bank_size, num_subcentroids=params.num_subcentroids, gamma=params.gamma)
    #     # model.pcrLinear = cosLinear(640, nclass)
    #     model.embedding_dim = 640
    #     model.prototypes = nn.Parameter(torch.torch.randn(100, params.num_subcentroids, 640),
    #                                    requires_grad=True)
    #     init.kaiming_normal_(model.prototypes, mode='fan_out', nonlinearity='relu')
    #     model.feat_norm = nn.LayerNorm(640)
    #     return model
    # elif params.data == 'cifar100' and params.agent == 'PCR_ca':
    #     model = Reduced_ResNet18_m(nclass, memory_bank_size = params.mem_bank_size, num_subcentroids=params.num_subcentroids, gamma=params.gamma)
    #     model.embedding_dim = 160
    #     model.prototypes = nn.Parameter(torch.zeros(100, params.num_subcentroids, 160),
    #                                    requires_grad=True)
    #     init.kaiming_normal_(model.prototypes, mode='fan_out', nonlinearity='relu')
    #     model.feat_norm = nn.LayerNorm(160)
    #     return model
    elif params.data == 'mini_imagenet'and params.agent == 'PCR_m':
        model = Reduced_ResNet18_m(nclass)
        model.pcrLinear = cosLinear(640, nclass)
        # model.attention2 = SimplifiedCrossAttention(640, nclass)
        return model
    elif params.agent == 'SUPER':
        if params.data == 'cifar100':
            return HashResNet18(nclass)
        elif params.data == 'mini_imagenet':
            return HashResNet18(nclass)
    elif params.data == 'openloris' and params.agent == 'PCR':
        return Reduced_ResNet18(nclass)
    elif params.agent == 'ER_ACE_L':
        if params.data == 'cifar100':
            return Reduced_ResNet18_prerelu(nclass)
        elif params.data == 'cifar10':
            return Reduced_ResNet18_prerelu(nclass)
        elif params.data == 'core50':
            model = Reduced_ResNet18_prerelu(nclass)
            model.linear = nn.Linear(2560, nclass, bias=True)
            return model
        elif params.data == 'mini_imagenet':
            model = Reduced_ResNet18_prerelu(nclass)
            model.linear = nn.Linear(640, nclass, bias=True)
            return model
        elif params.data == 'tinyimagenet':
            model = Reduced_ResNet18_prerelu(nclass)
            model.linear = nn.Linear(640, nclass, bias=True)
            return model
        elif params.data == 'food100':
            model = Reduced_ResNet18_prerelu(nclass)
            model.linear = nn.Linear(7840, nclass, bias=True)
            return model
        elif params.data == 'places100':
            model = Reduced_ResNet18_prerelu(nclass)
            model.linear = nn.Linear(7840, nclass, bias=True)
            return model
        elif params.data == 'openloris':
            return Reduced_ResNet18_prerelu(nclass)
    else:
        if params.data == 'cifar100':
            return ResNet18(nclass)
        elif params.data == 'cifar10':
            return ResNet18(nclass)
        elif params.data == 'core50':
            model = ResNet18(nclass)
            model.linear = nn.Linear(2560, nclass, bias=True)
            return model
        elif params.data == 'mini_imagenet':
            model = ResNet18(nclass)
            model.linear = nn.Linear(640, nclass, bias=True)
            return model
        elif params.data == 'tinyimagenet':
            model = ResNet18(nclass)
            model.linear = nn.Linear(640, nclass, bias=True)
            return model
        elif params.data == 'food100':
            model = ResNet18(nclass)
            model.linear = nn.Linear(7840, nclass, bias=True)
            return model
        elif params.data == 'places100':
            model = ResNet18(nclass)
            model.linear = nn.Linear(7840, nclass, bias=True)
            return model
        elif params.data == 'openloris':
            return ResNet18(nclass)



def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
