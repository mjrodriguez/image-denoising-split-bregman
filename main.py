from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# Anisotropic Total Variation ROF Denoising
# f = noise image
# mu = regularization parameter
# lam = regularization parameter for Bregman Iteration
# max_iter = Maximum number of iterations
class Denoise:
	def __init__(self, f, mu=100, lam=50, max_iter=15, BCs='mirror'):

		self.f = f
		self.mu = mu
		self.lam = lam
		self.max_iter = max_iter

		self.width = self.f.shape[1]
		self.height = self.f.shape[0]
		self.BCs = BCs
		self.tol = 1e-3*np.linalg.norm(f.flatten(),2)
		pass


	def atv_rof_sb(self) -> np.array:

		# Declaring some work variables
		self.__dx = np.zeros(self.f.shape)
		self.__dy = np.zeros(self.f.shape)
		self.__bx = np.zeros(self.f.shape)
		self.__by = np.zeros(self.f.shape)
		
		# u is the denoised image
		u = np.zeros(self.f.shape)

		# Iteration
		k = 1
		error = self.tol*1e3
		print("k = ", k, " Error = ", error)

		while error > self.tol and k <= self.max_iter:
			k += 1
			uprev = np.array(u)

			# Subproblem 1: Compute the optimal solution the u subproblem
			u = self.gs(u)

			# print("max u = ", np.amax(np.amax(u)))

			# Subproblem 2: Update dx and dy	
			# Computing the finite difference derivatives du/dx and du/dy
			uxtemp = np.c_[u[:,0], u[:,:self.width-1]]
			ux = u-uxtemp
			uytemp = np.vstack((u[0,:], u[:self.height-1,:]))
			uy = u-uytemp

			self.__dx = self.shrink(ux+self.__bx, 1/self.lam)
			self.__dy = self.shrink(uy+self.__by, 1/self.lam)
			self.__bx = self.__bx + (ux - self.__dx)
			self.__by = self.__by + (uy - self.__dy)

			error = np.sum(np.sum( (uprev-u)*(uprev-u) ))
			print("k = ", k, " Error = ", error, " Tol = ", self.tol)
		return u


	def gs(self, U: np.array) -> np.array:
		#TODO(mjrodriguez): Implement no flux BCs
		G = np.zeros(U.shape)

		if self.BCs == 'periodic':
			u = np.zeros([self.height+2, self.width+2])

			u[1:self.height+1,1:self.width+1] = U
			u[0, 1:self.width+1] = U[-1,:]
			u[-1,1:self.width+1] = U[0,:]
			u[1:self.height+1,0] = U[:,-1]
			u[1:self.height+1,-1] = U[:,0]

			a = self.mu/(self.mu + 4*self.lam)
			b = self.lam/(self.mu + 4*self.lam)

			for i in range(G.shape[0]):
				for j in range(G.shape[1]):
					# print(i,j)
					d_dx = self.__dx[i,j] - self.__dx[i-1,j]
					d_bx = self.__bx[i,j] - self.__bx[i-1,j]
					d_dy = self.__dy[i,j] - self.__dy[i,j-1]
					d_by = self.__by[i,j] - self.__by[i,j-1]
					G[i,j] = a*self.f[i,j] + b*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) + b*( d_dx - d_bx + d_dy - d_by )

		elif self.BCs == 'mirror':
			u = np.zeros([self.height+2, self.width+2])

			u[1:self.height+1,1:self.width+1] = U
			u[0, 1:self.width+1] = U[0,:]
			u[-1,1:self.width+1] = U[-1,:]
			u[1:self.height+1,0] = U[:,0]
			u[1:self.height+1,-1] = U[:,-1]

			a = self.mu/(self.mu + 4*self.lam)
			b = self.lam/(self.mu + 4*self.lam)

			for i in range(G.shape[0]):
				for j in range(G.shape[1]):
					# print(i,j)
					d_dx = self.__dx[i,j] - self.__dx[i-1,j]
					d_bx = self.__bx[i,j] - self.__bx[i-1,j]
					d_dy = self.__dy[i,j] - self.__dy[i,j-1]
					d_by = self.__by[i,j] - self.__by[i,j-1]
					G[i,j] = a*self.f[i,j] + b*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) + b*( d_dx - d_bx + d_dy - d_by )
			
		elif self.BCs == 'none':
			u = np.array(U)

			a = self.mu/(self.mu + 4*self.lam)
			b = self.lam/(self.mu + 4*self.lam)

			for i in range(1,G.shape[0]-1):
				for j in range(1,G.shape[1]-1):
					# print(i,j)
					d_dx = self.__dx[i,j] - self.__dx[i-1,j]
					d_bx = self.__bx[i,j] - self.__bx[i-1,j]
					d_dy = self.__dy[i,j] - self.__dy[i,j-1]
					d_by = self.__by[i,j] - self.__by[i,j-1]
					G[i,j] = a*self.f[i,j] + b*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]) + b*( d_dx - d_bx + d_dy - d_by )

		return G

	def shrink(self, x: np.array,y: np.array) -> np.array:
		Z = np.zeros(x.shape)
		return np.sign(x)*np.maximum(np.abs(x)-y,Z)

if __name__ == "__main__":

	# Pulling lena matrix from Bregman Cookbok Example code by Jerome Gilles
	mat = sio.loadmat('./pics/lena.mat') # imports a python dict
	img = mat['lena']
	f = img + 0.1*np.random.rand(img.shape[0], img.shape[1])

	# Default parameters
	# mu = 100
	# lambda = 50
	# iter = 15
	# BCs = 'mirror'
	
	dn = Denoise(f, max_iter=20)

	# Other parameters can be assigned this way:
	# dn = Denoise(f,mu=100,lam=15, max_iter=30)

	# Denoise Image
	clean_img = dn.atv_rof_sb()
	diff = img-clean_img
	# print(np.amin(np.amin(diff)), np.amax(np.amax(diff)))

	fig = plt.figure()
	fig.add_subplot(1,4,1)
	plt.title('Original Image')
	plt.axis('off')
	plt.imshow(img, cmap='gray')

	fig.add_subplot(1,4,2)
	plt.title('Noisy Image')
	plt.axis('off')
	plt.imshow(f,cmap='gray')

	fig.add_subplot(1,4,3)
	plt.title('Denoised Image')
	plt.axis('off')
	plt.imshow(clean_img, cmap='gray')

	fig.add_subplot(1,4,4)
	plt.title('Difference')
	plt.axis('off')
	plt.imshow(img-clean_img, cmap='gray')
	
	plt.show()




