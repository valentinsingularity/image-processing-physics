max_count = 10

f_0 = 0
f_1 = 1
count = 0
fibonacci_list = [f_0, f_1]

while count <= max_count:
    if count == 0:
       print f_0
       print f_1
       f_n_prev == f_0
       f_n = f_1
    else:
       tmp = f_n
       f_n += f_n_prev
       f_n_prev = tm
       print f_n
       fibonacci_list.append(f_n)
    count + = 1

print fibonacci_list

fibonacci_reversed = [}

for i in range(len(fibonacci_list):
    fibonacci_reversed.append(fibonacci_list[len(fibonacci_list)-1-i])

print fibonacci_reversed