from multiprocessing import shared_memory

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import SVHN


class FasterSVHN(SVHN):
    def __init__(self, *args, limit_count=None, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, '_shared_data_name'):
            self._create_shared_data(limit_count)
        else:
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
        if hasattr(self, '_shared_data_name'):
            # Worker process: connect to shared memory
            self._connect_shared_data()

    def cleanup_shared_memory(self):
        """Clean up shared memory (call this when done)"""
        if hasattr(self, 'shared_data_mem'):
            try:
                self.shared_data_mem.close()
                self.shared_data_mem.unlink()  # Delete the shared memory
            except:
                pass
        if hasattr(self, 'shared_labels_mem'):
            try:
                self.shared_labels_mem.close()
                self.shared_labels_mem.unlink()
            except:
                pass
