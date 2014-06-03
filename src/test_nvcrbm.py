import cPickle
import numpy as np
import utils
from PIL import Image

def dump_filter_image(filters, filename="filters.png"):
    img_array = utils.tile_raster_images(filters, (8,8), (4,8), (1,1))
    Image.fromarray(img_array).save(filename)

def test_load():
    kyoto_data = cPickle.load(open("../data/kyoto_train.pkl", "r"))

    test_data = np.arange(1, 65, dtype="float32")
    test_data = np.tile(test_data, 10*3*64).reshape(10, 3*64*64)
    test_data = test_data / 64.0;
    #test_data = np.array(10*3*64*64*[1], dtype="float32").reshape(10, 3*64*64)
    #test_filter = np.array(64*[1]+ 64*[2]+ 64*[3], dtype="float32")
    #test_filter = np.tile(test_filter, 32).reshape(32, 3*64)

    filter_num          = 32
    filter_size         = 8
    input_batch_num     = 10
    input_size          = 64
    channel_num         = 1
    pooling_rate        = 2
    left_upper_padding  = 4
    right_lower_padding = 3

    init_filters = np.array(np.random.normal(size=filter_num * channel_num *
        filter_size*filter_size), dtype="float32")
    init_filters = 0.01 * init_filters.reshape(filter_num, channel_num*filter_size*filter_size)

    init_hbias = np.array([-1.0] * filter_num, dtype="float32").reshape(filter_num, 1)

    init_vbias = np.array([0.0] * channel_num, dtype="float32").reshape(channel_num, 1)

    libnvcrbm = __import__("nvcrbm")
    cur_filters = libnvcrbm.init(filter_num, filter_size, 
            input_batch_num, input_size, channel_num,
            pooling_rate, left_upper_padding, right_lower_padding,
            init_filters, init_hbias, init_vbias)
    #init_filter = libnvcrbm.init(32, 8, 10, 64, 3, 2, 4, 3)

    batch_num = 500
    batch_size = 2
    for batch_idx in xrange(batch_num/batch_size):
        batch_data = kyoto_data[batch_idx*batch_size: 
                (batch_idx+1)*batch_size]
        batch_data = np.asarray(batch_data).reshape(batch_size, 
                channel_num * input_size * input_size)
        libnvcrbm.run_batch(batch_data)
        if batch_idx % 10 == 0:
            cur_filters = libnvcrbm.get_filters()
            dump_filter_image(cur_filters, "../data/kyoto/filters/batch_%d.png" % batch_idx)

if __name__ == "__main__":
    test_load()
