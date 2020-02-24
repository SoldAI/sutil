# NOTE:
# For now the tests only check functionality with costs = 1
# Use other costs at your own peril

import unittest

from sutil.base.Dataset import Dataset

class TestDataset(unittest.TestCase):
    
    def test_load_data(self):
        """
        Test for the dataset from file function
        """
        datafile = './sutil/datasets/ex1data1.txt'
        d = Dataset.fromDataFile(datafile, ',')
        d.plotDataRegression('example', False)
        print(d.shape)
        for i in range(len(d.X)):
            print(str(d.X[i]) + ' -->' + str(d.y[i]))
            
        datafile = './sutil/datasets/ex1data2.txt'
        d2 = Dataset.fromDataFile(datafile, ',')
        print(d2.shape)
        
        datafile = './sutil/datasets/ex2data1.txt'
        d3 = Dataset.fromDataFile(datafile, ',')
        print(d3.shape)
        d3.plotData('example3')
        
        datafile = './sutil/datasets/ex2data2.txt'
        d4 = Dataset.fromDataFile(datafile, ',')
        print(d4.shape)
        d4.plotData('example4')

    def test_normalize_features(self):
        """
        Test for normalized distance function
        """
        datafile = './sutil/datasets/ex1data1.txt'
        d = Dataset.fromDataFile(datafile, ',')
        print(d.shape)
        #d.plotDataRegression('example')
        #self.assertAlmostEqual(m.normalized_distance("abc", "ca"), 2/3)

    def test_biased_x(self):
        print("=" * 20)
        print("Testing biased x")
        datafile = './sutil/datasets/ex1data1.txt'
        d = Dataset.fromDataFile(datafile, ',')
        print(d.shape)
        print(d.X[0])
        print(d.normalizeFeatures())
        print(d.getBiasedX())
        #d.plotDataRegression('example')
        #self.assertEqual(m.similarity("abc", "abc"), 1)
        
    def test_split(self):
        print("=" * 20)
        print("Testing split")
        datafile = './sutil/datasets/ex1data1.txt'
        d = Dataset.fromDataFile(datafile, ',')
        d.plotDataRegression('example', True)
        print(d.shape)
        train, validation, test = d.split(0.8, 0.2)
        print(train.shape, validation.shape, test.shape)
        print(train.m, validation.m, test.m)
        print(train.m/d.m, validation.m/d.m, test.m/d.m)
        print(d.shape)
        train1, test1 = d.split(0.8, 0)
        print(train1.shape, test1.shape)
        print(train1.m, test1.m)
        print(train1.m/d.m, test1.m/d.m)
        print(d.shape)
        
    def test_save(self):
        print("=" * 20)
        print("Testing save")
        datafile = './sutil/datasets/ex1data1.txt'
        d = Dataset.fromDataFile(datafile, ',')
        d.plotDataRegression('example')
        d.save()
        d.save('test')
        print(d.shape)
    
    def test_load(self):
        print("=" * 20)
        print("Testing load")
        d = Dataset.load('test')
        print(d.shape)
        
if __name__ == '__main__':
    unittest.main()
