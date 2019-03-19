import tensorflow as tf
import numpy as np
import os


img_width = 208
img_height = 208

train_dir = 'C://Users/Sizhe/Desktop/CatsvsDogs/data/train/'

def get_files(file_dir):

    cats = []
    cats_label = []
    dogs = []
    dogs_label = []
    for file in os.listdir(file_dir):
        name = file.split(sep = '.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            cats_label.append(0)
        else:
            dogs.append(file_dir + file)
            dogs_label.append(1)
    print ('There are %d cats \n There are %d dogs' %(len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((cats_label, dogs_label))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return (image_list, label_list)

def get_batch(image, label, image_width, image_height, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels = 3)
###    image = tf.image.resize_image_with_crop_or_pad(image, image_width, image_height)
    image = tf.image.resize_images(image, [image_width, image_height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

###    comment the next sentence if you want to see images, do not comment it when training
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return (image_batch, label_batch)


### Test
### Comment the lines below when training
'''
import matplotlib.pyplot as plt
batch_size = 2
capacity = 256
image_width = 208
image_height = 208

train_dir = 'C://Users/Sizhe/Desktop/CatsvsDogs/data/train/'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, image_width, image_height, batch_size, capacity)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(batch_size):
                print ("label: %d" %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)
'''















    
