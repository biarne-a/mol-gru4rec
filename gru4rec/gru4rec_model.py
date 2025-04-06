import tensorflow as tf
from tensorflow import keras


step_signature = [{
    "input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "input_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    "label": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
}]


class Gru4RecModel(keras.models.Model):
    def __init__(self, vocab_size: int, hidden_size: int, dropout_p_embed=0.0, dropout_p_hidden=0.0, **kwargs):
        super().__init__()
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dropout_p_embed = dropout_p_embed
        self._dropout_p_hidden = dropout_p_hidden
        self._movie_id_embedding = tf.keras.layers.Embedding(vocab_size + 1, hidden_size)
        self._dropout_emb = tf.keras.layers.Dropout(dropout_p_embed)
        self._gru_layer = tf.keras.layers.GRU(hidden_size, return_sequences=True, recurrent_dropout=dropout_p_hidden)

    def call(self, inputs, training=False):
        ctx_movie_emb = self._movie_id_embedding(inputs["input_ids"])
        ctx_movie_emb = self._dropout_emb(ctx_movie_emb, training=training)
        gru_output = self._gru_layer(ctx_movie_emb, training=training)
        sequence_lengths = tf.cast(
          tf.reduce_sum(inputs["input_mask"], axis=1), tf.int32
        )
        # slice the gru sequence of output hidden state to get the last hidden
        # state according to sequence lengths
        batch_size = tf.shape(inputs["input_mask"])[0]
        last_hidden_state_idx = tf.concat(
            [
                tf.expand_dims(tf.range(0, batch_size), 1),
                tf.expand_dims(sequence_lengths - 1, 1),
            ],
            axis=1,
        )
        last_hidden_state = tf.gather_nd(gru_output, last_hidden_state_idx)
        logits = tf.matmul(last_hidden_state, tf.transpose(self._movie_id_embedding.embeddings))
        return logits

    # @tf.function(input_signature=step_signature)
    def train_step(self, inputs):
        y_true = inputs["label"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compute_loss(inputs, y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self._update_metrics(y_true, y_pred, loss)

    @tf.function(input_signature=step_signature)
    def test_step(self, inputs):
        y_true = inputs["label"]
        y_pred = self(inputs, training=False)

        loss = self.compute_loss(inputs, y_true, y_pred)
        return self._update_metrics(y_true, y_pred, loss)

    def _update_metrics(self, y_true, y_pred, loss):
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self._vocab_size,
            "hidden_size": self._hidden_size,
            "dropout_p_embed": self._dropout_p_embed,
            "dropout_p_hidden": self._dropout_p_hidden
        })
        return config

    @classmethod
    def from_config(cls, config, custom_object=None):
        return cls(**config)
