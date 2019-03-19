import os
import numpy as np
import tensorflow as tf
import model
import input_data

n_class = 2
image_width = 208
image_height = 208
batch_size = 16
capacity = 2000
### max_step > 15000 when training
max_step = 20000
learning_rate = 0.0001

def run_training():
    train_dir = 'C://Users/Sizhe/Desktop/CatsvsDogs/data/train/'
    logs_train_dir = 'C://Users/Sizhe/Desktop/CatsvsDogs/data/logs/train/'

    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                               train_label,
                                               image_width,
                                               image_height,
                                               batch_size,
                                               capacity)
    train_logits = model.inference(train_batch, batch_size, n_class)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    try:
        for step in np.arange(max_step):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

### step%50 when training            
            if step%50 == 0:
                print ('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.00))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step%2000 == 0 or step == max_step-1:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step= step)

    except tf.errors.OutOfRangeError:
        print ('Training finished -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
### Evaluate one image, comment the whole thing when training
'''
from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
    
### Randomly pick one image from training data and return ndarry    
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    
    image = Image.open(img_dir)
    plt.imShow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def get evaluate_one_image():
    
### Test one image against the saved models and parameters

    train_dir = 'C://Users/Sizhe/Desktop/CatsvsDogs/data/train/'   
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)
    
    with tf.Graph().as_default():
        batch_size = 1
        n_class = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        iamge = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, batch_size, n_class)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape = [208, 208, 3])
        
        logs_train_dir = 'C://Users/Sizhe/Desktop/CatsvsDogs/data/logs/train/'
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print ("Reading Checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print ('Loading success, global step is %s' %global_step)
            else:
                print ("Checkpoint file not found")
            
            prediction = sess.run(logit, feed_dict = {x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])

'''