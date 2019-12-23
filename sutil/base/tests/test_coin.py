# -*- coding: utf-8 -*-
# NOTE:
# For now the tests only check functionality with costs = 1
# Use other costs at your own peril

import unittest

from sutil.base.Coin import Coin

class TestCoin(unittest.TestCase):
    
    def test_toss(self):
        """
        Test for distance function
        """
        c0 = Coin(1)
        c1 = Coin(0)
        c2 = Coin(3)
        c3 = Coin(-3)
        self.assertTrue(c0.toss())
        self.assertFalse(c1.toss())
        self.assertTrue(c2.toss())
        self.assertFalse(c3.toss())
        
       
if __name__ == '__main__':
    unittest.main()