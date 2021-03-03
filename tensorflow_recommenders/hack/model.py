import tensorflow_recommenders as tfrs

from tensorflow_recommenders.experimental.models import RankingModel
from tensorflow_recommenders.hack import config


def ranking_model():
  return RankingModel(
      vocab_sizes=config.vocab_sizes,
      embedding_dim=config.embedding_dim,
      bottom_stack=tfrs.experimental.models.MlpBlock(units=config.bottom_mlp_list,
                                                     out_activation="relu"),
      feature_interaction=tfrs.experimental.models.DotInteraction(),
      top_stack=tfrs.experimental.models.MlpBlock(units=config.top_mlp_list,
                                                  out_activation="sigmoid"),
  )