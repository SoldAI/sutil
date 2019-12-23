import random

class Coin:

	def __init__(self, probability):
		self.probability = 0 if probability <=0 else min(1, probability)

	def toss(self):
		 return random.random() <= self.probability
