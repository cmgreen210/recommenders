import tensorflow as tf
from typing import List


def _generate_synthetic_data(num_dense: int,
                            vocab_sizes: List[int],
                            dataset_size: int,
                            is_training: bool,
                            batch_size: int) -> tf.data.Dataset:
  dense_tensor = tf.random.uniform(
      shape=(dataset_size, num_dense), maxval=1.0, dtype=tf.float32)

  sparse_tensors = []
  for size in vocab_sizes:
    sparse_tensors.append(
        tf.random.uniform(
            shape=(dataset_size,), maxval=int(size), dtype=tf.int32))

  sparse_tensor_elements = {
      str(i): sparse_tensors[i] for i in range(len(sparse_tensors))
  }

  # The mean is in [0, 1] interval.
  dense_tensor_mean = tf.math.reduce_mean(dense_tensor, axis=1)

  sparse_tensors = tf.stack(sparse_tensors, axis=-1)
  sparse_tensors_mean = tf.math.reduce_sum(sparse_tensors, axis=1)
  # The mean is in [0, 1] interval.
  sparse_tensors_mean = tf.cast(sparse_tensors_mean, dtype=tf.float32)
  sparse_tensors_mean /= sum(vocab_sizes)
  # The label is in [0, 1] interval.
  label_tensor = (dense_tensor_mean + sparse_tensors_mean) / 2.0
  # Using the threshold 0.5 to convert to 0/1 labels.
  label_tensor = tf.cast(label_tensor + 0.5, tf.int32)

  input_elem = {
      "dense_features": dense_tensor,
      "sparse_features": sparse_tensor_elements
  }, label_tensor

  dataset = tf.data.Dataset.from_tensor_slices(input_elem)
  if is_training:
    dataset = dataset.repeat()

  return dataset.batch(batch_size, drop_remainder=True)


def datasets(num_training, num_eval, dense_size, vocab_sizes, batch_size):
  training_ds = _generate_synthetic_data(num_dense=dense_size, vocab_sizes=vocab_sizes,
                                         dataset_size=num_training, is_training=True,
                                         batch_size=batch_size)
  eval_ds = _generate_synthetic_data(num_dense=dense_size, vocab_sizes=vocab_sizes,
                                     dataset_size=num_eval, is_training=False,
                                     batch_size=batch_size)
  return training_ds, eval_ds
