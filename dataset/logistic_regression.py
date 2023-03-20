import tensorflow as tf
from dataset.cnews_dataset import CnewsDataset


class LogisticRegression:
    def __init__(self, num_class, seq_length):
        self.seq_length = seq_length
        self.num_class = num_class

        self.x_ph = tf.placeholder(tf.float32, [None, seq_length])
        self.label_ph = tf.placeholder(tf.int64, [None, 1])
        self.label = tf.one_hot(tf.squeeze(self.label_ph), depth=num_class, dtype=tf.float32)
        with tf.name_scope("weight"):
            self.weight = tf.Variable(tf.ones([seq_length, self.num_class]))
            # tf.summary.histogram("weight", self.weight)
        with tf.name_scope("bias"):
            self.bias = tf.Variable(tf.zeros([self.num_class]))
            # tf.summary.histogram("bias", self.bias)
        out = tf.matmul(self.x_ph, self.weight) + self.bias
        y = tf.nn.softmax(out, dim=-1)
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(y), reduction_indices=[1])) + tf.nn.l2_loss(self.weight) * 0.5
            # tf.summary.scalar("loss", self.loss)
        self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

        self.correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(self.label, 1))
        with tf.name_scope("acc"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
            # tf.summary.scalar("acc", self.accuracy)
        self.predict_cls = tf.argmax(y, 1)

    def train(self, dataset, sess, is_traing=True):
        total_sample_num = 0
        total_loss = 0.0
        total_correct_num = 0
        while True:
            try:
                train_data = sess.run(dataset)
                vec, label = train_data["vec"], train_data["label"]
                loss_train, acc_train = sess.run([self.loss, self.accuracy], feed_dict={self.x_ph: vec, self.label_ph: label})
                sess.run(self.train_op, feed_dict={self.x_ph: vec, self.label_ph: label})
                batch_size = vec.shape[0]
                total_sample_num += batch_size
                total_loss += (loss_train * batch_size)
                total_correct_num += (acc_train * batch_size)
                # self.log_rslt = sess.run(merged, feed_dict={self.x_ph: vec, self.label_ph: label})
            except tf.errors.OutOfRangeError:
                break
        return round(total_loss / total_sample_num, 5), round(total_correct_num / total_sample_num, 3)

    def evaluate(self, dataset, sess):
        loss, acc = self.train(dataset, sess, False)
        return loss, acc


class LogisticRegressionV2:
    def __init__(self, cate_feat_num, numeric_feat_num, vocab_size):
        self.input_cate_pd = tf.placeholder(dtype=tf.int64, shape=[None, cate_feat_num])
        self.input_num_ph = tf.placeholder(dtype=tf.float32, shape=[None, numeric_feat_num])
        self.label_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.label = tf.cast(self.label_ph, tf.float32)

        self.embedding_cate = tf.get_variable(name="cate_weight", shape=[vocab_size, 1], dtype=tf.float32,
                                              initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.embedding_nume = tf.get_variable(name="num_weight", shape=[numeric_feat_num, 1], dtype=tf.float32,
                                              initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.cat_out = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_cate, self.input_cate_pd), axis=0)

        self.num_out = tf.reduce_sum(tf.matmul(self.input_num_ph, self.embedding_nume), axis=0)
        self.bias = tf.Variable(0.)
        # print(self.cat_out.shape, self.num_out.shape)
        self.out = self.cat_out + self.num_out + self.bias
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.squeeze(self.label), logits=self.out)) + 0.1 * tf.nn.l2_loss(self.embedding_cate) + 0.1 * tf.nn.l2_loss(self.embedding_nume)

        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
        self.tvars = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.tvars)

if __name__ == '__main__':
    VOCAB_SIZE = 10799
    sess = tf.Session()
    model = LogisticRegression(10, VOCAB_SIZE)
    merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter("../data/cnews_log/log")
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        dataset_train = CnewsDataset.readTFRecord("../data/cnews/train.tf_record.train", 64, VOCAB_SIZE)
        dataset_val = CnewsDataset.readTFRecord("../data/cnews/train.tf_record.val", 64, VOCAB_SIZE)
        train_loss, train_acc = model.train(dataset_train, sess, True)
        val_loss, val_acc = model.evaluate(dataset_val, sess)
        tf.summary.scalar("train_loss", train_loss)
        tf.summary.scalar("val_loss", val_loss)

        print(f"Epoch:{epoch} train step: loss_{train_loss}, acc_{train_acc} Val step: loss_{val_loss}, acc_{val_acc}")
        log_writer.add_summary(model.log_rslt, epoch)






        # total_batch = 0
        # total_sample = 0
        # total_acc = 0
        # loss = 0.0
        # while True:
        #     try:
        #         train_data = sess.run(dataset_train)
        #         vec, label = train_data["vec"], train_data["label"]
        #         loss_train, acc_train, crect = sess.run([model.loss, model.accuracy, model.correct_predictions], feed_dict={model.x_ph: vec, model.label_ph: label})
        #         sess.run(model.train_op, feed_dict={model.x_ph: vec, model.label_ph: label})
        #         total_batch += 1
        #         print(crect, loss_train)
        #         total_sample += vec.shape[0]
        #         total_acc += (acc_train * vec.shape[0])
        #         loss += (loss_train * vec.shape[0])
        #     except tf.errors.OutOfRangeError:
        #         break
        # print(f"Epoch {epoch}", loss / total_sample, total_acc / total_sample)
        # val_data = sess.run(dataset_val)
        # vec_val, label_val = val_data["vec"], val_data["label"]
        # loss_val, acc_val = sess.run([model.loss, model.accuracy],
        #                              feed_dict={model.x_ph: vec_val, model.label_ph: label_val})
        # print(f"Epoch {epoch} val step", loss_val, acc_val)
