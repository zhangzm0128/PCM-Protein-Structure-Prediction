import numpy as np
def tournament(K, N, fitness):
    '''
        tournament selection
        :param K: number of solutions to be compared
        :param N: number of solutions to be selected
        :param fit: fitness vectors
        :return: index of selected solutions
    '''
    n = len(fitness)
    if N > 0 and N < 1:
        N = int(N * n)
    if N is None or N > n or N < 0:
        N = n
    mating_pool = []
    for i in range(N):
        a = np.random.randint(n)
        for j in range(K):
            b = np.random.randint(n)
            for r in range(fitness[0, :].size):
                if fitness[(b, r)] < fitness[(a, r)]:
                    a = b
        mating_pool.append(a)
    return np.array(mating_pool)

def random_select(num_proteins, num_selection):
    mating_pool = np.random.randint(0, num_proteins, size=[1, num_selection])[0]
    return mating_pool