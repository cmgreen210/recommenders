num_of_dense_features = 13
vocab_sizes = [10, 21, 119, 155, 4, 97, 14, 108, 36]

embedding_dim = 8
bottom_mlp_list = [256, 64, embedding_dim]
top_mlp_list = [512, 256, 1]

DEFAULT_BATCH_SIZE = 4096
NUM_TRAIN_EXAMPLES = 400_000
NUM_EVAL_EXAMPLES = 10_000
