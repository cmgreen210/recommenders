import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow_recommenders.hack import config, fake_data
from tensorflow_recommenders.hack.model import ranking_model


def run(evaluate=True):
  model = ranking_model()
  embedding_optimizer = tf.keras.optimizers.Adagrad()
  dense_optimizer = tf.keras.optimizers.Adam()

  optimizer = tfrs.experimental.optimizers.CompositeOptimizer([
          (embedding_optimizer, lambda: model.embedding_trainable_variables),
          (dense_optimizer, lambda: model.dense_trainable_variables),
      ])
  model.compile(optimizer)

  train_ds, eval_ds = fake_data.datasets(config.NUM_TRAIN_EXAMPLES,
                                         config.NUM_EVAL_EXAMPLES,
                                         config.num_of_dense_features,
                                         config.vocab_sizes,
                                         config.DEFAULT_BATCH_SIZE)
  model.fit(train_ds, epochs=1, steps_per_epoch=config.NUM_TRAIN_EXAMPLES // config.DEFAULT_BATCH_SIZE)

  if evaluate:
    metrics = model.evaluate(eval_ds, return_dict=True)

    print(metrics)


if __name__ == '__main__':
  run(evaluate=False)
