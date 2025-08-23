from multiprocessing import shared_memory
import atexit
import weakref
import os

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import SVHN, ImageNet, ImageFolder

# Global registry to track shared memory objects for cleanup
_shared_memory_registry = weakref.WeakSet()


class FasterDatasetBase:
    """Base class for faster dataset implementations with shared memory"""

    def __init__(self, *args, limit_count=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_main_process = True

        if not hasattr(self, '_shared_data_name'):
            self._create_shared_data(limit_count)
            _shared_memory_registry.add(self)
        else:
            self._is_main_process = False
            self._connect_shared_data()

    def _get_raw_data(self):
        """Override this method in subclasses to provide raw data"""
        raise NotImplementedError

    def _create_shared_data(self, limit_count):
        raw_data, raw_labels = self._get_raw_data()

        if limit_count:
            if limit_count > len(raw_data):
                print('limiting samples to max count of', len(raw_data))
            else:
                rand_idx = np.random.choice(len(raw_data), limit_count, replace=False)
                raw_data = [raw_data[i] for i in rand_idx]
                raw_labels = [raw_labels[i] for i in rand_idx]

        processed_data = []
        processed_labels = []

        for img, target in zip(raw_data, raw_labels):
            processed_img = self._process_image(img)
            processed_target = self._process_target(target)
            processed_data.append(processed_img)
            processed_labels.append(processed_target)

        # Convert to numpy arrays
        self.processed_data = np.array(processed_data)
        self.processed_labels = np.array(processed_labels, dtype=np.int64)

        # Create shared memory
        data_size = self.processed_data.nbytes
        label_size = self.processed_labels.nbytes

        try:
            self.shared_data_mem = shared_memory.SharedMemory(
                create=True, size=data_size, name=f'{self._get_memory_prefix()}_data_{id(self)}')
            self.shared_labels_mem = shared_memory.SharedMemory(
                create=True, size=label_size, name=f'{self._get_memory_prefix()}_labels_{id(self)}')

            # Copy data to shared memory
            shared_data_array = np.ndarray(
                self.processed_data.shape, dtype=self.processed_data.dtype,
                buffer=self.shared_data_mem.buf)
            shared_labels_array = np.ndarray(
                self.processed_labels.shape, dtype=self.processed_labels.dtype,
                buffer=self.shared_labels_mem.buf)

            shared_data_array[:] = self.processed_data[:]
            shared_labels_array[:] = self.processed_labels[:]

            # Store metadata
            self._shared_data_name = self.shared_data_mem.name
            self._shared_labels_name = self.shared_labels_mem.name
            self._data_shape = self.processed_data.shape
            self._data_dtype = self.processed_data.dtype
            self._labels_shape = self.processed_labels.shape
            self._labels_dtype = self.processed_labels.dtype

            self.data = shared_data_array
            self.labels = shared_labels_array

        except Exception as e:
            raise f"Failed to create shared memory: {e}"

    def _process_image(self, img):
        """Override in subclasses for dataset-specific image processing"""
        if self.transform is not None:
            img = self.transform(img)
        return img.numpy() if torch.is_tensor(img) else np.array(img)

    def _process_target(self, target):
        """Override in subclasses for dataset-specific target processing"""
        target = int(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

    def _get_memory_prefix(self):
        """Override in subclasses to provide memory name prefix"""
        raise NotImplementedError

    # Reuse all other methods from FasterSVHN
    def _connect_shared_data(self):
        try:
            self.shared_data_mem = shared_memory.SharedMemory(name=self._shared_data_name)
            self.shared_labels_mem = shared_memory.SharedMemory(name=self._shared_labels_name)

            self.data = np.ndarray(
                self._data_shape, dtype=self._data_dtype,
                buffer=self.shared_data_mem.buf)
            self.labels = np.ndarray(
                self._labels_shape, dtype=self._labels_dtype,
                buffer=self.shared_labels_mem.buf)

        except Exception as e:
            print(f"Failed to connect to shared memory: {e}")
            raise

    def __getitem__(self, index: int):
        return torch.from_numpy(self.data[index].copy()), torch.tensor(self.labels[index], dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, '_shared_data_name'):
            state['_shared_data_name'] = self._shared_data_name
            state['_shared_labels_name'] = self._shared_labels_name
            state['_data_shape'] = self._data_shape
            state['_data_dtype'] = self._data_dtype
            state['_labels_shape'] = self._labels_shape
            state['_labels_dtype'] = self._labels_dtype

        state.pop('data', None)
        state.pop('labels', None)
        state.pop('processed_data', None)
        state.pop('processed_labels', None)
        state.pop('shared_data_mem', None)
        state.pop('shared_labels_mem', None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._is_main_process = False
        if hasattr(self, '_shared_data_name'):
            self._connect_shared_data()

    def __del__(self):
        self.cleanup_shared_memory()

    def cleanup_shared_memory(self):
        try:
            if hasattr(self, 'shared_data_mem'):
                self.shared_data_mem.close()
                if self._is_main_process:
                    try:
                        self.shared_data_mem.unlink()
                    except FileNotFoundError:
                        pass

            if hasattr(self, 'shared_labels_mem'):
                self.shared_labels_mem.close()
                if self._is_main_process:
                    try:
                        self.shared_labels_mem.unlink()
                    except FileNotFoundError:
                        pass
        except Exception:
            pass


class FasterSVHN(FasterDatasetBase, SVHN):
    def _get_raw_data(self):
        return self.data, self.labels

    def _process_image(self, img):
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        return super()._process_image(img)

    def _get_memory_prefix(self):
        return 'svhn'


class FasterImageNet(FasterDatasetBase, ImageNet):
    def _get_raw_data(self):
        samples = self.samples
        images = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        return images, labels

    def _process_image(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return super()._process_image(img)

    def _get_memory_prefix(self):
        return 'imagenet'


class FasterImageNette(FasterDatasetBase, ImageFolder):
    """ImageNette dataset implementation that doesn't require ILSVRC2012_devkit_t12.tar.gz"""

    def __init__(self, root, split='train', **kwargs):
        # ImageNette has 'train' and 'val' splits
        data_path = os.path.join(root, 'imagenette2', split)
        super().__init__(data_path, **kwargs)

    def _get_raw_data(self):
        samples = self.samples
        images = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        return images, labels

    def _process_image(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return super()._process_image(img)

    def _get_memory_prefix(self):
        return 'imagenette'


def _cleanup_all_shared_memory():
    """Cleanup function to be called at exit"""
    for dataset in list(_shared_memory_registry):
        if dataset is not None:
            dataset.cleanup_shared_memory()


# Register cleanup function to run at exit
atexit.register(_cleanup_all_shared_memory)



if __name__ == "__main__":
    from components.FL_sim import CustomSampler
    import time
    from torchvision import transforms

    dataset = [
        FasterSVHN(
            # limit_count = 10,
            root='../../data/SVHN', split=s,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4377, 0.4438, 0.4728],
                    std=[0.1980, 0.2010, 0.1970]),
            ])
        ) for s in ['train', 'test']]

    sampler = CustomSampler(len(dataset[0]), 5,
        True, True, False)

    shared_train_loader = torch.utils.data.DataLoader(
        dataset[0], batch_size=15000,
        num_workers=10, persistent_workers=True, sampler=sampler)

    shared_test_loader = torch.utils.data.DataLoader(
        dataset[1], batch_size=15000 * 3, shuffle=False,
        num_workers=5, persistent_workers=True)

    print(f"Dataset length: {len(dataset[0])}, {len(dataset[1])}")

    # Measure time for dataset iteration
    start_time = time.time()
    for _ in range(10):
        for i in range(5):
            for _ in dataset[0]:
                pass

            sampler.set_agent_partition(i)
            idxes = list(iter(sampler))
            for _ in dataset[0][idxes]:
                pass
            for _ in dataset[1]:
                pass
    end_time = time.time()
    print(f"dataset: {(end_time - start_time)*1000:.2f} ms")

    # Measure time for DataLoader iteration
    start_time = time.time()
    for _ in range(10):
        for i in range(5):
            sampler.set_agent_partition('ALL')
            for _ in shared_train_loader:
                pass
            sampler.set_agent_partition(i)
            for _ in shared_train_loader:
                pass
            for _ in shared_test_loader:
                pass
    end_time = time.time()
    print(f"dataloader: {(end_time - start_time)*1000:.2f} ms")
