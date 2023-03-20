import tensorflow as tf
from logistic_regression.logistic_regression import LogisticRegression
from dataset.cnews_dataset import CnewsDataset
from neuralprophet import NeuralProphet

VOCAB_SIZE = 10799
sess = tf.Session()
model = LogisticRegression(10, VOCAB_SIZE)
merged = tf.summary.merge_all()
log_writer = tf.summary.FileWriter("../data/cnews_log/log")
sess.run(tf.global_variables_initializer())
for epoch in range(1):
    dataset_train = CnewsDataset.readTFRecord("../data/cnews/train.tf_record.train", 64, VOCAB_SIZE)
    dataset_val = CnewsDataset.readTFRecord("../data/cnews/train.tf_record.val", 64, VOCAB_SIZE)
    train_loss, train_acc = model.train(dataset_train, sess, True)
    val_loss, val_acc = model.evaluate(dataset_val, sess)
    tf.summary.scalar("train_loss", train_loss)
    tf.summary.scalar("val_loss", val_loss)
    print(f"Epoch:{epoch} train step: loss_{train_loss}, acc_{train_acc} Val step: loss_{val_loss}, acc_{val_acc}")
tf.train.Saver().save(sess, "../data/model_ckpt/cnews/cnews")

