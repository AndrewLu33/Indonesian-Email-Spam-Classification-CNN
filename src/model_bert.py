import tensorflow as tf
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel

MAX_LEN = 128

# ============================================================
# TOKENIZER (AMAN disimpan global)
# ============================================================
tokenizer = BertTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p1"
)

def encode_bert(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="tf"
    )

# ============================================================
# BERT + CNN MODEL (FIXED)
# ============================================================
def build_bert_cnn(num_classes):
    # 🔥 CREATE BACKBONE DI DALAM FUNCTION
    bert = TFBertModel.from_pretrained(
        "indobenchmark/indobert-base-p1"
    )

    input_ids = layers.Input(
        shape=(MAX_LEN,),
        dtype=tf.int32,
        name="input_ids"
    )
    attention_mask = layers.Input(
        shape=(MAX_LEN,),
        dtype=tf.int32,
        name="attention_mask"
    )

    bert_output = bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        training=False
    )[0]

    x = layers.Conv1D(128, 3, activation="relu")(bert_output)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
