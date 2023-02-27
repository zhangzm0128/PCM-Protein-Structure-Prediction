import numpy as np
from scipy.spatial.distance import cdist
import time

from search.MOEA import MOEA
from search.SelectMethod import tournament

class PCM(MOEA):
    def __init__(self, proteins, config, energy, coder, logging, current_gen=1):
        super(PCM, self).__init__(config, energy, coder, logging, current_gen)
        self.pop_size = self.config['pop_size']
        self.proteins = proteins
        self.obj_num = self.proteins[0].obj_num
        self.max_angles, self.min_angles = self.proteins[0].get_angles_field()
        self.init_param()
        self.evolution_parameters_init()
        
    def init_param(self):
        self.t = self.config['t']
        self.archive_thresh = self.config['archive_threshold']
        self.mesh_div = self.config['mesh_div']
        
    def generate_offspring(self, obj_view, z_min):
        # calculate ASF
        weight = obj_view / np.tile(np.sum(obj_view, axis=1).reshape(-1, 1), [1, self.obj_num])
        weight = np.where(weight < 1e-6, 1e-6, weight)
        obj_view = obj_view - np.tile(z_min, [self.pop_size, 1])
        asf = np.max(obj_view / weight, axis=1)
        
        # obtain the rank value of each solution's ASF value
        rank = np.argsort(asf)
        asf_rank = np.argsort(rank)
        
        # calculate the min angle of each solution to others
        angle = np.arccos(1 - cdist(obj_view, obj_view, metric='cosine'))
        np.fill_diagonal(angle, np.inf)
        angle_min = np.min(angle, axis=1)
        
        # binary tournament selection
        mating_pool = np.zeros(self.pop_size, dtype=int)
        for x in range(self.pop_size):
            select_index = np.random.permutation(self.pop_size)[:2]
            if asf[select_index[0]] < asf[select_index[1]] and angle_min[select_index[0]] > angle_min[select_index[1]]:
                select_index = select_index[0]
            else:
                select_index = select_index[1]
            
            if np.random.random() < 1.0002 - asf_rank[select_index] / self.pop_size:
                mating_pool[x] = select_index
            else:
                mating_pool[x] = np.random.randint(0, high=self.pop_size)
        
        return mating_pool

    def NDSort(self, PopObj, nSort):
        # get all proteins' energy values
        PopObj, a, Loc = np.unique(PopObj, return_index=True, return_inverse=True, axis=0)

        rank = np.lexsort(PopObj[:, ::-1].T)
        PopObj = PopObj[rank].copy()
        table, _ = np.histogram(Loc, max(Loc) + 1)

        N, M = PopObj.shape
        # FrontNo = np.full((1, N), np.inf)
        FrontNo = np.ones(N) * np.inf
        FrontNo = np.array(FrontNo)
        MaxFNo = 0

        index_table = np.where(FrontNo < np.inf)
        # print(sum(table[index_table]))

        while sum(table[index_table]) < min(nSort, len(Loc)):
            # print(sum(table[index_table]))
            MaxFNo = MaxFNo + 1
            for i in range(N):
                if FrontNo[i] == np.inf:
                    Dominated = False
                    for j in range(i - 1, 0, -1):
                        if FrontNo[j] == MaxFNo:
                            m = 1
                            while m <= M - 1:
                                if PopObj[i, m] >= PopObj[j, m]:
                                    m = m + 1
                                else:
                                    break
                            Dominated = m >= M
                            if Dominated | M == 1:
                                break
                    if not Dominated:
                        FrontNo[i] = MaxFNo
            index_table = np.where(FrontNo < np.inf)

        FrontNo[rank] = FrontNo
        FrontNo = FrontNo[Loc]
        
        # pareto_dict = {}
        # for x in range(1, MaxFNo + 1):
        #     pareto_dict[x] = np.where(FrontNo == x)[0]
        
        
        return FrontNo, MaxFNo  # , pareto_dict

        

        
    def get_pareto_rank_array(self, pareto_front_dict):
        pareto_size = np.sum([len(x) for x in pareto_front_dict.values()])
        pareto_rank_array = np.zeros(pareto_size, dtype=int)
        
        for rank in pareto_front_dict:
            for x in pareto_front_dict[rank]:
                pareto_rank_array[x] = rank
        max_rank = np.max(pareto_rank_array)
        return pareto_rank_array, max_rank
        
    def select_protein(self, combine_obj_view, z_min):
        combine_size = combine_obj_view.shape[0]
        
        # calculate the distance between each solution to the ideal point
        combine_obj_view = combine_obj_view - np.tile(z_min, [combine_size, 1])
        con = np.sqrt(np.sum(combine_obj_view ** 2, axis=1))
        
        # calculate the angle between each two solutions
        angle = np.arccos(1 - cdist(combine_obj_view, combine_obj_view, metric='cosine'))
        np.fill_diagonal(angle, np.inf)
        
        # eliminate solutions one by one 
        remain = np.arange(combine_size, dtype=int)
        while len(remain) > self.pop_size:
            # identify the two solutions A and B with the min angle
            rank_1 = np.argsort(angle[remain.reshape(-1, 1), remain], axis=1)
            sort_a = np.sort(angle[remain.reshape(-1, 1), remain], axis=1)
            rank_2 = sort_a[:,0].argsort()
            solution_a = rank_2[0]
            solution_b = rank_1[solution_a, 0]
            
            # eliminate one of A and B
            if con[remain[solution_a]] - con[remain[solution_b]] > self.t:
                remain = np.delete(remain, solution_a)
            elif con[remain[solution_b]] - con[remain[solution_a]] > self.t:
                remain = np.delete(remain, solution_b)
            else:
                remain = np.delete(remain, solution_a)
        
        return remain
            
    def update_archive(self):
        temp_archive = self.proteins + self.archive
        temp_archive_angle_view = np.array([x.angle_view().copy() for x in temp_archive])
        _, save_index = np.unique(temp_archive_angle_view, axis=0, return_index=True)
        temp_archive = [temp_archive[x].copy() for x in save_index]
        
        temp_archive_obj = [x.obj.copy() for x in temp_archive]
        pareto_rank, max_rank = self.NDSort(temp_archive_obj, len(temp_archive_obj))
        select_pareto_front = np.where(pareto_rank == 1)[0]

        # pareto_front_dict = self.nonDominatedSort(temp_archive) 
        # select_pareto_front = pareto_front_dict[1]  # + pareto_front_dict[2]
        
        temp_archive = [temp_archive[x] for x in select_pareto_front]
        temp_archive_obj_view = np.array([x.obj for x in temp_archive])

        if len(temp_archive) > self.archive_thresh:
            self.delete_archive(temp_archive, temp_archive_obj_view)
        else:
            self.archive = temp_archive

        self.archive_obj_view = np.array([x.obj for x in self.archive])

    def delete_archive(self, temp_archive, temp_archive_obj_view):
        num_archive = len(temp_archive)

        # calculate the grid location of each solution
        obj_max = np.max(temp_archive_obj_view, axis=0)
        obj_min = np.min(temp_archive_obj_view, axis=0)
        div = (obj_max - obj_min) / self.mesh_div
        div = np.tile(div, (num_archive, 1))
        obj_min = np.tile(obj_min, (num_archive, 1))

        grid_location = np.floor((temp_archive_obj_view - obj_min) / div)
        grid_location[grid_location >= self.mesh_div] = self.mesh_div - 1
        grid_location[np.isnan(grid_location)] = 0

        # detect the grid of each solution belongs to
        _, _, site = np.unique(grid_location, return_index=True, return_inverse=True, axis=0)

        # calculate the crowd degree of each grid
        crowd_degree = np.histogram(site, np.max(site) + 1)[0]

        del_index = np.zeros(num_archive, dtype=bool)

        while np.sum(del_index) < num_archive - self.archive_thresh:
            max_grid = np.where(crowd_degree == max(crowd_degree))[0]
            temp = np.random.randint(0, len(max_grid))
            grid = max_grid[temp]

            in_grid = np.where(site == grid)[0]

            temp = np.random.randint(0, len(in_grid))
            p = in_grid[temp]
            del_index[p] = True
            site[p] = -100
            crowd_degree[grid] = crowd_degree[grid] - 1

        del_index = np.where(del_index == 1)[0]

        # for x in range(len(temp_archive)):
        #    if x not in del_index:
        #        self.archive = temp_archive[x]

        self.archive = [temp_archive[x].copy() for x in range(len(temp_archive)) if x not in del_index]    

    
    
    def run(self):
        
        pareto_front_dict = self.nonDominatedSort(self.proteins)
        self.archive = [self.proteins[x].copy() for x in pareto_front_dict[1]]
        proteins_obj_view = np.array([x.obj.copy() for x in self.proteins])
        z_min = np.min(proteins_obj_view, axis=0)
        
        for x in range(self.current_gen, self.max_gen):
            start_time = time.time()
            mating_pool = self.generate_offspring(proteins_obj_view, z_min)
            offspring = [self.proteins[x].copy() for x in mating_pool]

            # self.pro_m = np.exp(-self.current_gen / (4 * self.max_gen))
            offspring_angle_view = self.crossover_binary(offspring)
            offspring_angle_view = self.mutation_polynomial(offspring_angle_view)
            
            for protein, angle in zip(offspring, offspring_angle_view):
                protein.update_angle_from_view(angle)
            self.energy.calculate_energy(offspring)
            
            combine_proteins = [x.copy() for x in self.proteins + offspring]
            combine_proteins_obj_view = np.array([x.obj.copy() for x in combine_proteins])
            
            offspring_obj_view = [x.obj.copy() for x in offspring]
            z_min = np.min(np.vstack([z_min, offspring_obj_view]), axis=0)                      
            
            next_index = self.select_protein(combine_proteins_obj_view, z_min)
            self.proteins = [combine_proteins[x].copy() for x in next_index]
            
            self.update_archive()
            
            self.current_gen = x
            print(x, time.time() - start_time, z_min, len(self.archive))
            self.logging.write(self.proteins, self.coder, self.config['save_all'], x)
            # self.logging.write_pdb(x)
        
        self.logging.write_archive(self.archive, self.coder)
        self.energy.stop()
      
