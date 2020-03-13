from mxnet import np, npx

npx.set_np()

# Anything beyond a 2 dimensional tensor will still use
# the Frobenius norm to compute the norm.
A = np.array([[[[1, 2, 3], [1, 2, 4]]]])
B = np.array([[1, 2, 3], [1, 2, 4]])


print(np.linalg.norm(B))
print(np.linalg.norm(A))
