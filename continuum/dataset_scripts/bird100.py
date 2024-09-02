import os
from PIL import Image
import numpy as np
from torchvision import datasets
from torchvision import transforms
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns


class BIRD100(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'bird100'
        self.params = params
        # self.label_path = params.dataset
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(BIRD100, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)


    def download_load(self):
        train_dataset_path = self.params.imagenet_path + '/train'
        # val_dataset_path = self.params.imagenet_path + '/val'


        self.train_data, self.train_label = self.load_data('train.lst', train_dataset_path)
        self.test_data, self.test_label = self.load_data('test.lst', train_dataset_path)
        # self.val_data, self.val_label = self.load_data('val.lst', val_dataset_path)
        print(self.train_data.shape, self.test_data.shape)
        exit(0)

        # dataset_train = datasets.CIFAR100(root=self.root, train=True, download=True)
        # self.train_data = dataset_train.data
        # self.train_label = np.array(dataset_train.targets)
        # dataset_test = datasets.CIFAR100(root=self.root, train=False, download=True)
        # self.test_data = dataset_test.data
        # self.test_label = np.array(dataset_test.targets)

    def load_data(self, filne_name, dataset_path):

        images = []
        labels = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        with open(os.path.join(self.root, filne_name), 'r') as file:
            for line in file:
                image_path, label = line.strip().split(' ')
                full_path = image_path.replace('/DATASET_PATH', dataset_path)

                # try:
                image = Image.open(full_path)
                image_transformed = preprocess(image)
                images.append(np.array(image_transformed))
                labels.append(int(label))
                # except IOError:
                #     print(f"Cannot load image {full_path}")

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
