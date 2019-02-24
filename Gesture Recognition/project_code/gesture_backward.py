#coding:utf-8
import tensorflow as tf
import gesture_forward
import os
import gesture_generateds

BATCH_SIZE = 60
LEARNING_RATE_BASE =  0.01
LEARNING_RATE_DECAY = 0.99   
REGULARIZER = 0.0001 
STEPS = 10000 
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH="./model/" 
MODEL_NAME="my_geature_model" 
train_num_examples = 1080#2

def backward():
    x = tf.placeholder(tf.float32,[
	BATCH_SIZE,
	gesture_forward.IMAGE_SIZE,
	gesture_forward.IMAGE_SIZE,
	gesture_forward.NUM_CHANNELS]) 
    y_ = tf.placeholder(tf.float32, [None, gesture_forward.OUTPUT_NODE])
    
    y = gesture_forward.forward(x,True, REGULARIZER) 
    global_step = tf.Variable(0, trainable=False) 

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) 
    loss = cem + tf.add_n(tf.get_collection('losses')) 

    learning_rate = tf.train.exponential_decay( 
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE, 
		LEARNING_RATE_DECAY,
        staircase=True) 
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): 
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver() 
    img_batch, label_batch = gesture_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)#3
    
    with tf.Session() as sess: 
        init_op = tf.global_variables_initializer() 
        sess.run(init_op) 

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
        if ckpt and ckpt.model_checkpoint_path:
        	saver.restore(sess, ckpt.model_checkpoint_path) 
        
        coord = tf.train.Coordinator()#4
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)#5
        
        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])#6
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys}) 
            if i % 100 == 0: 
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        
        coord.request_stop()#7
        coord.join(threads)#8

def main():
    #mnist = input_data.read_data_sets("./data/", one_hot=True) 
    backward()

if __name__ == '__main__':
    main()


