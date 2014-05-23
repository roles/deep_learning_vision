import nvcrbm
import cPickle
import numpy as np

def test_load():
    kyoto_data = cPickle.load(open("../data/kyoto_train.pkl", "r"))
    test_data = np.arange(1, 65, dtype="float32")
    test_data = np.tile(test_data, 10*3*64).reshape(10, 3*64*64)
    test_data = test_data / 64.0;
    #test_data = np.array(10*3*64*64*[1], dtype="float32").reshape(10, 3*64*64)
    test_filter = np.array(64*[1]+ 64*[2]+ 64*[3], dtype="float32")
    test_filter = np.tile(test_filter, 32).reshape(32, 3*64)
    nvcrbm.run(32, 8, 10, 64, 3, 2, 4, 3, test_data, test_filter);

if __name__ == "__main__":
    test_load()
