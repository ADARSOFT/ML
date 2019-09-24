next_x = 6  # We start the search at x=6
gamma = 0.01  # Step size multiplier
precision = 0.0001  # Desired precision of result
max_iters = 100  # Maximum number of iterations

# Derivative function
df = lambda x: 4 * x**2

for i in range(max_iters):
	
    
	current_x = next_x
	err = df(current_x)
	next_x = current_x - gamma * err
	print(current_x, '-', gamma, '*', err, ' = ', next_x)
	step = next_x - current_x
	if abs(step) <= precision:
		break
	
print("Minimum at", next_x)


lr = 0.1
decay = 0.01
iter_num = 500
print('0', lr)
for i in range(iter_num):
	lr = lr * 1 / (1 + decay*i)
	print(i, lr)