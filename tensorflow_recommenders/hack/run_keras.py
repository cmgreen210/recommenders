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
  model = _model()
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

  metrics = model.evaluate(eval_ds, return_dict=True)

  print(metrics)


if __name__ == '__main__':
  run()