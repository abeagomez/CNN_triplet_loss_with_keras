import tensorflow as tf

tf.InteractiveSession()

xi = tf.constant([[1,1,1], [0,1,1], [0,0,1]])
yi = tf.constant([[0,0,0], [0,1,0], [0,0,1]])
yk = tf.constant([[1,0,1], [1,1,1], [1,0,1]])
yl = tf.constant([[1,1,0], [0,1,1], [0,0,1]])

norm_xi_yk = tf.reduce_sum(tf.square(tf.subtract(xi, yk)), 1)
norm_yi_yl = tf.reduce_sum(tf.square(tf.subtract(yi, yl)), 1)
norm_xi_yi = tf.reduce_sum(tf.square(tf.subtract(xi, yi)), 1)

dif_one_and_norm_xi_yk = tf.subtract(tf.ones(
                            tf.shape(norm_xi_yk), dtype=tf.int32), norm_xi_yk)

dif_one_and_norm_yi_yl = tf.subtract(tf.ones(
                            tf.shape(norm_yi_yl), dtype=tf.int32), norm_yi_yl)

term1 = dif_one_and_norm_xi_yk
term2 = dif_one_and_norm_yi_yl

max_zero_and_term1 = tf.maximum(
                            tf.zeros(tf.shape(term1), dtype=tf.int32), term1)
max_zero_and_term2 = tf.maximum(
                            tf.zeros(tf.shape(term2), dtype=tf.int32), term2)

term3 = max_zero_and_term1
term4 = max_zero_and_term2

max_term3_and_term4 = tf.maximum(term3, term4)

F_xi_yi = tf.add(max_term3_and_term4, norm_xi_yi)
max_zero_and_F_xi_yi = tf.maximum(
                        tf.zeros(tf.shape(F_xi_yi), dtype=tf.int32), F_xi_yi)
term5 = max_zero_and_F_xi_yi
loss = tf.divide(tf.reduce_sum(term5),tf.size(term5))

a = tf.constant(6.0)
b = tf.constant(2)
c = tf.divide(a,tf.to_float(b))
print(c.eval())

