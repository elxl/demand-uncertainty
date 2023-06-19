from torch.utils.data import Dataset	

class NY_Data(Dataset):

	def __init__(self, x, y, adj, history, weather, los):
		self.x = x
		self.y = y
		self.adj = adj
		self.history = history
		self.weather = weather
		self.los = los
		

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):

		sample = tuple(self.x[idx],self.y[idx],self.adj,self.history[idx],self.weather[idx],self.los[idx])

		return sample