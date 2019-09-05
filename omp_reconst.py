import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

def create_dct_bases_matrix(size):
	const1=np.sqrt(1/size)
	dct_bases=np.zeros((size,size),dtype='float')
	x,y=dct_bases.shape
	for u in range(0,x):
		for n in range(0,y):
			if u == 0:
				dct_bases[u,n]=const1
			else:
				const2=np.sqrt(2/size)
				angle=(np.pi*(2*n+1)*u)/(2*size)
				cos=np.cos(angle)
				dct_bases[u,n]=const2*cos
	return dct_bases

def read_video(video_link,frames_to_read):
	video = cv2.VideoCapture(video_link)
	video_frames=[]
	frames_read=0
	while video.isOpened():
		if frames_read == frames_to_read:
			break;
		ret,frame = video.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		gray = gray.astype('double')
		video_frames.append(gray)
		frames_read+=1
	return video_frames


def apply_omp(y,rand_code,frames_arr1):
	bases=np.kron(create_dct_bases_matrix(8).T,create_dct_bases_matrix(8).T)
	op_img_frames=np.zeros_like(frames_arr1)
	num_count=np.zeros_like(frames_arr1)
	for i in range(y.shape[0]-8+1):
		print(i)
		for j in range(y.shape[1]-8+1):
			code_patch = rand_code[:,i:i+8,j:j+8]
			phi = np.zeros([64,64*T])
			for k in range(T):
				C = code_patch[k,:,:]
				phi[:,(k)*64:(k+1)*64]=np.dot(np.diag(C.flatten()),bases)
			Y = y[i:i+8,j:j+8].flatten().reshape(64,1)
			theta,l = omp(Y,phi,2)
			for k in range(T):
				op_img_frames[k,i:i+8,j:j+8]+=np.dot(bases,theta[k*64:(k+1)*64]).reshape(8,8)#,out=op_img_frames[k,i:i+8,j:j+8],casting="unsafe")
				#print(op_img_frames[k,i:i+8,j:j+8])
				num_count[k,i:i+8,j:j+8]+=1
	return op_img_frames,num_count

def omp(y,d,error_code):
	resd = y
	soln = np.zeros([d.shape[1],y.shape[1]])
	supp_set = [] #np.zeros(d.shape[0],dtype='uint8')
	norm_dict = d/np.max(d,axis=0)
	error = np.power(error_code,2)*d.shape[0]
	resd_norm = np.sum(np.power(resd,2))
	j = 0
	a = 0
	max_coeff = y.shape[0]
	while resd_norm > error and j < max_coeff-1:
		proj = np.dot(norm_dict.T,resd)
		ind = np.argmax(np.abs(proj))
		supp_set.append(ind)
		a= np.linalg.lstsq(d[:,supp_set[0:j+1]],y,rcond=None)[0]
		#a = np.dot(a,y)
		resd = y - np.dot(d[:,supp_set[0:j+1]],a)
		resd_norm = np.sum(np.power(resd,2))
		j+=1
	soln[supp_set[0:j],0]=a.reshape(a.shape[0],)
	return soln,j


if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print("please provide T value")
		sys.exit(0)
	T = int(sys.argv[1])
	frames=read_video('cars.avi',T)
	frames_arr = np.asarray(frames)
	fr_size = frames_arr.shape
	frames_arr1 = frames_arr[:,174:fr_size[1],130:fr_size[2]]
	fr_size1 = frames_arr1.shape
	rand_code = np.random.randint(0,2,size=fr_size1)
	coded_snaps = np.multiply(rand_code,frames_arr1)
	sum_codded_snaps = np.sum(coded_snaps,0)
	op_img,counts=apply_omp(sum_codded_snaps,rand_code,frames_arr1)
	o_img=np.divide(op_img,counts)
	fig=plt.figure(figsize=(30,30))
	for i in range(T):
		fig.add_subplot(1,T,i+1)
		plt.imshow(o_img[i],cmap='gray')
		plt.colorbar()

	plt.show()