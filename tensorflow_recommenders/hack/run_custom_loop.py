import time

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow_recommenders.hack import config, fake_data

from tensorflow_recommenders.experimental.models import RankingModel


def _model():
  return RankingModel(
      vocab_sizes=config.vocab_sizes,
      embedding_dim=config.embedding_dim,
      bottom_stack=tfrs.experimental.models.MlpBlock(units=config.bottom_mlp_list,
                                                     out_activation="relu"),
      feature_interaction=tfrs.experimental.models.DotInteraction(),
      top_stack=tfrs.experimental.models.MlpBlock(units=config.top_mlp_list,
                                                  out_activation="sigmoid"),
  )


def run():
  @tf.function
  def train_step(model, loss, x, y, optimizer_and_trainables):
    with tf.GradientTape() as tape:
      output = model(x, training=True)
      loss_value = loss(y, output)
    trainable_list = [t() for _, t in optimizer_and_trainables]
    optimizer_list = [o for o, _ in optimizer_and_trainables]
    grads = tape.gradient(loss_value, trainable_list)
    for i, (optimizer, trainables) in enumerate(zip(optimizer_list, trainable_list)):
      optimizer.apply_gradients(zip(grads[i], trainables))
    return loss_value

  train_ds, eval_ds = fake_data.datasets(config.NUM_TRAIN_EXAMPLES,
                                         config.NUM_EVAL_EXAMPLES,
                                         config.num_of_dense_features,
                                         config.vocab_sizes,
                                         config.DEFAULT_BATCH_SIZE)

  embedding_optimizer = tf.keras.optimizers.Adagrad()
  dense_optimizer = tf.keras.optimizers.Adam()
  model = _model()
  optimizers_and_trainables = [(embedding_optimizer, lambda: model.embedding_trainable_variables),
                               (dense_optimizer, lambda: model.dense_trainable_variables)]
  loss = tf.keras.losses.BinaryCrossentropy()

  epochs = 1
  steps_per_epoch = config.NUM_TRAIN_EXAMPLES // config.DEFAULT_BATCH_SIZE
  for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in zip(range(steps_per_epoch), train_ds):
      if step >= steps_per_epoch:
        break

      loss_value = train_step(model, loss, x_batch_train, y_batch_train, optimizers_and_trainables)

      # Log every 200 batches.
      if step % 20 == 0:
        print(
          "Training loss (for one batch) at step %d: %.4f"
          % (step, float(loss_value))
        )
        print("Seen so far: %d samples" % ((step + 1) * 64))

    # # Display metrics at the end of each epoch.
    # train_acc = train_acc_metric.result()
    # print("Training acc over epoch: %.4f" % (float(train_acc),))
    #
    # # Reset training metrics at the end of each epoch
    # train_acc_metric.reset_states()
    #
    # # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #   test_step(x_batch_val, y_batch_val)
    #
    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))


if __name__ == "__main__":
  run()
