import numpy as np
import matplotlib.pyplot as plt
import skimage


# Anisotropic Total Variation ROF Denoising
# f = noise image
# mu = regularization parameter
# lam = regularization parameter for Bregman Iteration
# max_iter = Maximum number of iterations
class Denoise:
	def __init__(self, f, mu=100, lam=50, max_iter=50):

		self.f = f
		self.mu = mu
		self.lam = lam
		self.max_iter = max_iter

		self.width = self.f.shape[1]
		self.height = self.f.shape[0]

		self.tol = 1e-4

		pass


	def atv_rof_sb(self):

		# Declaring some work variables
		self.__dx = np.zeros(self.f.shape)
		self.__dy = np.zeros(self.f.shape)
		self.__bx = np.zeros(self.f.shape)
		self.__by = np.zeros(self.f.shape)
		
		# u is the denoised image
		u = np.zeros(self.f.shape)

		# Iteration
		k = 1
		error = 1

		while error > self.tol and k < self.max_iter:
			k += 1
			uprev = u
			u = self.gs(u)
			# TODO(mjrodriguez): Compute ux and uy
			# TODO(mjrodriguez): self.__dx = self.shrink()



		return u


	def gs(self, u):

		G = np.zeros(self.f.shape)
		G = u

		a = self.mu/(self.mu + 4*self.lam)
		b = self.lam/(self.mu + 4*self.lam)

		for i in range(1,G.shape[0]-1):
			for j in range(1,G.shape[1]-1):
				d_dx = self.__dx[i,j]-self.__dx[i-1,j]
				d_bx = self.__bx[i,j]-self.__bx[i-1,j]
				d_dy = self.__dy[i,j] - self.__dy[i,j-1]
				d_by = self.__by[i,j] - self.__by[i,j-1]
				G[i,j] = a*self.f[i,j] + b*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) + b*( d_dx - d_bx + d_dy - d_by )

		return G
	
	def shrink(self,x: np.array,y: np.array) -> np.array:
		Z = np.zeros(x.shape)
		return np.sign(x)*np.max(np.abs(x)-y,Z)

if __name__ == "__main__":

	img = skimage.io.imread("./pics/len_std.jpg", as_gray=True)

	nimg = skimage.util.random_noise(img,mode='gaussian', seed=None, clip=True)

