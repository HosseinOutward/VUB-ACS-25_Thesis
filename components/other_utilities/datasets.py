from multiprocessing import shared_memory
import atexit
import weakref

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import SVHN

# Global registry to track shared memory objects for cleanup
_shared_memory_registry = weakref.WeakSet()


class FasterSVHN(SVHN):
    def __init__(self, *args, limit_count=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_main_process = True  # Track if this is the main process that created shared memory

        if not hasattr(self, '_shared_data_name'):
            self._create_shared_data(limit_count)
            # Register for cleanup
            _shared_memory_registry.add(self)
        else:
            self._is_main_process = False  # This is a worker process
            self._connect_shared_data()

    def _create_shared_data(self, limit_count):
        data = self.data
        labels = self.labels
        if limit_count:
            if limit_count>len(data):
                print('limiting samples to max count of', len(data))
            else:
                rand_idx = np.random.choice(data.shape[0], limit_count, replace=False)
                data = data[rand_idx]
                labels = labels[rand_idx]

        processed_data = []
        processed_labels = []

        for i, (img, target) in enumerate(zip(data, labels)):
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)
            processed_data.append(img.numpy() if torch.is_tensor(img) else np.array(img))

            target = int(target)
            if self.target_transform is not None:
                target = self.target_transform(target)
            processed_labels.append(target)

        # Convert to numpy arrays
        self.processed_data = np.array(processed_data)
        self.processed_labels = np.array(processed_labels, dtype=int)

        # Create shared memory for data
        data_size = self.processed_data.nbytes
        label_size = self.processed_labels.nbytes

        try:
            # Create shared memory blocks
            self.shared_data_mem = shared_memory.SharedMemory(
                create=True, size=data_size, name=f'svhn_data_{id(self)}')
            self.shared_labels_mem = shared_memory.SharedMemory(
                create=True, size=label_size, name=f'svhn_labels_{id(self)}')

            # Copy data to shared memory
            shared_data_array = np.ndarray(
                self.processed_data.shape, dtype=self.processed_data.dtype,
                buffer=self.shared_data_mem.buf)
            shared_labels_array = np.ndarray(
                self.processed_labels.shape, dtype=self.processed_labels.dtype,
                buffer=self.shared_labels_mem.buf)

            shared_data_array[:] = self.processed_data[:]
            shared_labels_array[:] = self.processed_labels[:]

            # Store metadata for workers
            self._shared_data_name = self.shared_data_mem.name
            self._shared_labels_name = self.shared_labels_mem.name
            self._data_shape = self.processed_data.shape
            self._data_dtype = self.processed_data.dtype
            self._labels_shape = self.processed_labels.shape
            self._labels_dtype = self.processed_labels.dtype

            # Use shared arrays as our data
            self.data = shared_data_array
            self.labels = shared_labels_array

        except Exception as e:
            raise f"Failed to create shared memory: {e}"

    def _connect_shared_data(self):
        """Connect to existing shared memory (for worker processes)"""
        try:
            # Connect to existing shared memory
            self.shared_data_mem = shared_memory.SharedMemory(name=self._shared_data_name)
            self.shared_labels_mem = shared_memory.SharedMemory(name=self._shared_labels_name)

            # Create numpy arrays backed by shared memory
            self.data = np.ndarray(
                self._data_shape, dtype=self._data_dtype,
                buffer=self.shared_data_mem.buf
            )
            self.labels = np.ndarray(
                self._labels_shape, dtype=self._labels_dtype,
                buffer=self.shared_labels_mem.buf
            )

        except Exception as e:
            print(f"Failed to connect to shared memory: {e}")
            # This shouldn't happen in normal operation
            raise

    def __getitem__(self, index: int):
        return torch.from_numpy(self.data[index].copy()), self.labels[index]

    def __getstate__(self):
        """Custom pickling to transfer shared memory info to workers"""
        state = self.__dict__.copy()
        # Include shared memory metadata for workers
        if hasattr(self, '_shared_data_name'):
            state['_shared_data_name'] = self._shared_data_name
            state['_shared_labels_name'] = self._shared_labels_name
            state['_data_shape'] = self._data_shape
            state['_data_dtype'] = self._data_dtype
            state['_labels_shape'] = self._labels_shape
            state['_labels_dtype'] = self._labels_dtype

        # Remove large arrays and shared memory objects from pickle
        state.pop('data', None)
        state.pop('labels', None)
        state.pop('processed_data', None)
        state.pop('processed_labels', None)
        state.pop('shared_data_mem', None)
        state.pop('shared_labels_mem', None)

        return state

    def __setstate__(self, state):
        """Custom unpickling to reconnect to shared memory"""
        self.__dict__.update(state)
        self._is_main_process = False  # Worker processes are not main
        if hasattr(self, '_shared_data_name'):
            # Worker process: connect to shared memory
            self._connect_shared_data()

    def __del__(self):
        """Destructor to clean up shared memory"""
        self.cleanup_shared_memory()

    def cleanup_shared_memory(self):
        """Clean up shared memory (call this when done)"""
        try:
            if hasattr(self, 'shared_data_mem'):
                self.shared_data_mem.close()
                # Only unlink (delete) if this is the main process that created it
                if self._is_main_process:
                    try:
                        self.shared_data_mem.unlink()
                    except FileNotFoundError:
                        pass  # Already unlinked

            if hasattr(self, 'shared_labels_mem'):
                self.shared_labels_mem.close()
                # Only unlink (delete) if this is the main process that created it
                if self._is_main_process:
                    try:
                        self.shared_labels_mem.unlink()
                    except FileNotFoundError:
                        pass  # Already unlinked

        except Exception:
            pass  # Ignore cleanup errors


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
