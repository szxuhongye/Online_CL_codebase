import os
from PIL import Image
import numpy as np
from torchvision import datasets
from torchvision import transforms
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns


class TINYIMAGENET(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'tinyimagenet'
        self.params = params
        # self.label_path = params.dataset
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(TINYIMAGENET, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)


    def download_load(self):
        train_root = '/home/hx5239/PCR/datasets/tinyimagenet200/train'
        test_root = '/home/hx5239/PCR/datasets/tinyimagenet200/val'

        self.train_data, self.train_label = self.load_tiny_imagenet(train_root)
        self.test_data, self.test_label = self.load_tiny_imagenet(test_root)
        # self.val_data, self.val_label = self.load_data('val.lst', val_dataset_path)
        # print(self.train_data.shape, self.test_data.shape)
        # exit(0)

        # dataset_train = datasets.CIFAR100(root=self.root, train=True, download=True)
        # self.train_data = dataset_train.data
        # self.train_label = np.array(dataset_train.targets)
        # dataset_test = datasets.CIFAR100(root=self.root, train=False, download=True)
        # self.test_data = dataset_test.data
        # self.test_label = np.array(dataset_test.targets)

    
    def load_tiny_imagenet(self, path, num_classes=200):
        """
        Load Tiny ImageNet dataset.
        
        Parameters:
        - path: str, path to the Tiny ImageNet dataset directory.
        - num_classes: int, number of classes in the dataset (default is 200 for Tiny ImageNet).
        
        Returns:
        - images: np.array, array of images.
        - labels: np.array, array of labels corresponding to the images.
        """
        images = []
        labels = []
        
        # Loop through all classes
        for i, class_folder in enumerate(os.listdir(path)):
            if i >= num_classes:  # Limit to the specified number of classes
                break
            class_path = os.path.join(path, class_folder)
            # Loop through all images in the class folder
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = Image.open(image_path).convert('RGB')  
                image = image.resize((64, 64))
                images.append(np.array(image))
                labels.append(i)
        
        return np.array(images), np.array(labels)

    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 32,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)
        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(class_nums=100, num_tasks=self.task_nums, fixed_order=self.params.fix_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                                                         self.params.ns_factor)
