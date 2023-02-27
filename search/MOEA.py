import os
import numpy as np
import random
import sklearn
from scipy import misc
import time
from scipy.spatial.distance import cdist
from .SelectMethod import *


class MOEA(object):
    # This class is using for many objects search algorithm in Protein<class>.
    # The evaluation function is using Energy<class>.
    # The definition of Protein can be found in ProteinPredict_MaOSearch/utils/ProteinUtils.py
    # The definition of Energy can be found in ProteinPredict_MaOSearch/energy/energy.py
    def __init__(self, config, energy, coder, logging, current_gen=1):
        self.config = config
        self.energy = energy
        self.coder = coder
        self.logging = logging
        self.current_gen = current_gen

        if 'select_method' in config:
            self.select_mode = config['select_method']
        else:
            self.select_mode = None
        self.select_method_init()

        self.name = config['name']
        self.max_gen = config['max_gen']
        self.pop_size = config['pop_size']

        self.start_time = time.localtime()

        self.obj_num = None
        self.max_angles, self.min_angles = None, None


    def evolution_parameters_init(self):
        self.pro_c = self.config['pro_c']
        self.dis_c = self.config['dis_c']

        if 'pro_m' in self.config:
            self.pro_m = self.config['pro_m']
        else:
            self.pro_m = 1 / len(self.max_angles)
        self.dis_m = self.config['dis_m']

    def format_output_file(self):
        pass

    def select_method_init(self):
        if self.select_mode in ['tour', 'tournament']:
            self.select_method = tournament
        elif self.select_mode in ['random', 'urs']:
            self.select_method = random_select
        else:
            raise RuntimeError('Invalid selection method: %s.'%(self.select_mode))

    def nonDominatedSort(self, proteins):
        # get all proteins' energy values
        energys = []
        for x in proteins:
            energys.append(x.obj.copy())
        energys = np.array(energys)
        index_energys = np.arange(len(energys))

        # calculate different level pareto front
        rank = 1
        front_dict = {}
        while energys.shape[0] != 0:
            current_front = np.ones(energys.shape[0], dtype=bool)
            for i, energy in enumerate(energys):
                current_front[i] = np.all(np.any(energys >= energy, axis=1))

            for x in index_energys[current_front]:
                proteins[x].paretoRank = rank

            front_dict[rank] = index_energys[current_front]

            energys = energys[~current_front]
            index_energys = index_energys[~current_front]
            rank = rank + 1

        return front_dict

    @staticmethod
    def n_choose_k(n, k):
        # the type of n must in [int, float, np.ndarray]
        if isinstance(n, int) or isinstance(n, float):
            return int(misc.comb(n, k))
        else:
            v = n
            v_size = v.size
            kk = min(k, v_size - k)
            m = np.prod(np.arange(v_size + 1 - kk, v_size + 1)) / np.prod(np.arange(1, kk + 1))
            x = np.zeros((int(m), k), dtype=int)
            irow_end = v_size + 1 - k
            x[0: irow_end][:, k - 1] = np.arange(k, v_size + 1)
            for cbn_begin in range(k - 1, 0, -1):
                nrow = irow_end
                irow_end_last = irow_end
                x[0: irow_end][:, cbn_begin - 1] = cbn_begin
                for cbn_first in range(cbn_begin + 1, cbn_begin + v_size - k + 1):
                    nrow = int(nrow * (v_size + 1 + cbn_begin - cbn_first - k) / (v_size + 1 - cbn_first))
                    irow_begin = irow_end + 1
                    irow_end = irow_begin + nrow - 1

                    x[irow_begin - 1: irow_end][:, cbn_begin - 1] = cbn_first
                    x[irow_begin - 1: irow_end][:, cbn_begin: k] = x[irow_end_last-nrow: irow_end_last][:, cbn_begin: k]
            x = v[x - 1]
            return x.astype(float)

    @classmethod
    def generate_uniform_point(cls, n, m):
        # input parameters:
        #     n is the size of population
        #     m is the number of objective
        # return:
        #     w is reference points set
        #     num_points is the number of the reference points
        h1 = 1
        while cls.n_choose_k(h1 + m, m - 1) <= n:
            h1 = h1 + 1
        w = cls.n_choose_k(np.arange(1, h1 + m), m - 1) - \
              np.tile(np.arange(m-2 + 1), (cls.n_choose_k(h1 + m-1, m-1), 1)) - 1
        w = (np.column_stack((w, np.zeros(w.shape[0]) + h1)) - np.column_stack((np.zeros(w.shape[0]), w))) / h1
        if h1 < m:
            h2 = 0
            while cls.n_choose_k(h1 + m-1, m-1) + cls.n_choose_k(h2 + m, m-1) <= m:
                h2 = h2 + 1
            if h2 > 0:
                w2 = cls.n_choose_k(np.arange(1, h2 + m), m-1) - \
                     np.tile(np.arange(m-2 + 1), (cls.n_choose_k(h2 + m-1, m-1), 1)) - 1
                w2 = (np.column_stack((w2, np.zeros(w2.shape[0])+h2))-np.column_stack((np.zeros(w2.shape[0]), w2)))/h2
                w = np.row_stack((w, w2 / 2 + 1 / (2 * m)))

        uniform_point = np.maximum(w, 1e-6)
        num_points = uniform_point.shape[0]

        return uniform_point, num_points

    def ref_guided_select(self, proteins_with_ref, ref_point, alpha):
        proteins_with_ref_obj_view = np.array([x.obj for x in proteins_with_ref])
        transl_obj_view = proteins_with_ref_obj_view - np.min(proteins_with_ref_obj_view, axis=0)

        penalty_factor = self.obj_num * (self.current_gen / self.max_gen) ** alpha

        cosine_gamma = cdist(ref_point, ref_point,'cosine')
        cosine_gamma[np.eye(len(cosine_gamma), dtype=bool)] = 0
        gamma = np.min(np.arccos(cosine_gamma), axis=1)

        theta = np.arccos(cdist(transl_obj_view, ref_point, 'cosine'))
        assigned_vectors = np.unique(np.argmin(theta, 1))

        apd = np.zeros((len(assigned_vectors), len(proteins_with_ref)))
        for j in range(len(proteins_with_ref)):
            for i in range(len(assigned_vectors)):
                p_theta =  penalty_factor * theta[i, j] / gamma[j]
                apd[i, j] = (1 + p_theta) * np.sqrt(np.sum(np.power(transl_obj_view[i], 2), axis=1))

        select_index = np.min(apd, axis=0)
        selection = [proteins_with_ref[x].copy() for x in select_index]

        return selection

    def apd_select(self, proteins, ref_point, alpha, gamma = None):
        penalty_factor = self.obj_num * (self.current_gen / self.max_gen) ** alpha

        proteins_obj_view = np.array([x.obj for x in proteins])
        transl_obj_view = proteins_obj_view - np.min(proteins_obj_view, axis=0)

        proteins_obj_norm = np.linalg.norm(transl_obj_view, axis=1)
        proteins_obj_norm = np.repeat(proteins_obj_norm, len(transl_obj_view[0, :])).reshape(
            len(proteins_obj_view), len(proteins_obj_view[0, :]))

        proteins_obj_norm[proteins_obj_norm == 0] = np.finfo(float).eps
        normalized_fitness = np.divide(proteins_obj_view, proteins_obj_norm)

        cosine_theta = np.dot(normalized_fitness, np.transpose(ref_point))
        cosine_theta[np.where(cosine_theta > 1)] = 1
        cosine_theta[np.where(cosine_theta < -1)] = -1
        theta = np.arccos(cosine_theta)

        if gamma is None:
            cosine_gamma = cdist(ref_point, ref_point, 'cosine')
            cosine_gamma[np.eye(len(cosine_gamma), dtype=bool)] = 0
            gamma = np.min(np.arccos(cosine_gamma), axis=1)

        assigned_vectors = np.argmax(cosine_theta, axis=1)
        selection = np.array([], dtype=int)

        for x in range(len(ref_point)):
            sub_population_index = np.atleast_1d(np.squeeze(np.where(assigned_vectors == x)))
            sub_population_fitness = transl_obj_view[sub_population_index]
            if len(sub_population_fitness > 0):
                angles = theta[sub_population_index, x]
                angles = np.divide(angles, gamma[x])
                sub_pop_fitness_magnitude = np.sqrt(
                    np.sum(np.power(sub_population_fitness, 2), axis=1)
                )
                apd = np.multiply(
                    np.transpose(sub_pop_fitness_magnitude),
                    (1 + np.dot(penalty_factor, angles)),
                )
                minidx = np.where(apd == np.nanmin(apd))
                if np.isnan(apd).all():
                    continue
                selx = sub_population_index[minidx]
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))


        return np.sort(selection.reshape(-1)), gamma

    def generate_mating_index(self, mating_num):
        shuffled_ids = list(range(mating_num))
        random.shuffle(shuffled_ids)
        # Create random pairs from the population for mating
        mating_index = [
            shuffled_ids[x * 2: (x + 1) * 2] for x in range(int(len(shuffled_ids) / 2))
        ]
        return mating_index

    '''
    def crossover_binary(self, mating_pool):
        mating_pool_view = np.array([x.angle_view().copy() for x in mating_pool])
        mating_num, num_angle = mating_pool_view.shape

        mating_index = self.generate_mating_index(mating_num)
        offspring_view = np.zeros((0, num_angle))

        for x in mating_index:
            beta = np.zeros(num_angle)
            miu = np.random.rand(num_angle)
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (self.dis_c + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (self.dis_c + 1))
            beta = beta * ((-1) ** np.random.randint(0, high=2, size=num_angle))
            beta[np.random.rand(num_angle) > self.pro_c] = 1

            avg = (mating_pool_view[x[0]] + mating_pool_view[x[1]]) / 2
            diff = (mating_pool_view[x[0]] - mating_pool_view[x[1]]) / 2

            offspring_view = np.vstack((offspring_view, avg + beta * diff))
            offspring_view = np.vstack((offspring_view, avg - beta * diff))

        return offspring_view
    '''    
        
    def crossover_binary(self, mating_pool):
        parent_index = np.array(list(range(len(mating_pool))) + list(range(int(np.ceil(len(mating_pool) / 2)) * 2 - len(mating_pool))))
        mating_pool_view = np.array([mating_pool[x].angle_view().copy() for x in parent_index])
        
        mating_pool_size, num_angle = mating_pool_view.shape
        
        parent_1_angle_view = mating_pool_view[:int(mating_pool_size/2), :]
        parent_2_angle_view = mating_pool_view[int(mating_pool_size/2):, :]
        
        beta = np.zeros((int(mating_pool_size / 2), num_angle))
        miu = np.random.random((int(mating_pool_size / 2), num_angle))
        
        beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (self.dis_c + 1));
        beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (self.dis_c + 1))
        beta = beta * ((-1) ** np.random.randint(0, high=2, size=beta.shape))
        beta[np.random.random(beta.shape) > 0.5] = 1
        beta[np.tile(np.random.random((int(mating_pool_size/2), 1)) > self.pro_c, [1, num_angle])] = 1
        
        avg = (parent_1_angle_view + parent_2_angle_view) / 2
        diff = (parent_1_angle_view - parent_2_angle_view) / 2
        
        offspring_view = np.zeros_like(mating_pool_view)
        offspring_view[:int(mating_pool_size/2), :] = avg + beta * diff
        offspring_view[int(mating_pool_size/2):, :] = avg - beta * diff
        
        return offspring_view
        
    def differental_evolution(self, mating_pool):
        parent_index = np.array(list(range(len(mating_pool))) + list(range(int(np.ceil(len(mating_pool) / 3)) * 3 - len(mating_pool))))
        mating_pool_view = np.array([mating_pool[x].angle_view().copy() for x in parent_index])
        
        mating_pool_size, num_angle = mating_pool_view.shape
        
        parent_1_angle_view = mating_pool_view[:int(mating_pool_size / 3), :]
        parent_2_angle_view = mating_pool_view[int(mating_pool_size / 3): int(mating_pool_size / 3 * 2), :]
        parent_3_angle_view = mating_pool_view[int(mating_pool_size / 3 * 2):, :]
        
        offspring_angle_view = parent_1_angle_view.copy()
        site = np.random.random(offspring_angle_view.shape) < self.cr
        offspring_angle_view[site] = offspring_angle_view[site] + self.F * (parent_2_angle_view[site] - parent_3_angle_view[site])
        
        return offspring_angle_view

    def mutation_polynomial(self, offspring_view):
        offspring = offspring_view.copy()

        min_val = np.ones_like(offspring) * self.min_angles
        max_val = np.ones_like(offspring) * self.max_angles
        k = np.random.random(offspring.shape)
        miu = np.random.random(offspring.shape)
        temp = np.logical_and((k <= self.pro_m), (miu < 0.5))

        offspring_scaled = (offspring - min_val) / (max_val - min_val)
        offspring_scaled[np.isnan(offspring_scaled)] = 0
        offspring_scaled[np.isinf(offspring_scaled)] = 0

        offspring[temp] = offspring[temp] + \
                               ((max_val[temp] - min_val[temp]) * ((2 * miu[temp] + (1 - 2 * miu[temp]) *
                                                        (1 - offspring_scaled[temp])
                                                        ** (self.dis_m + 1)) ** (1 / (self.dis_m + 1)) - 1))
        temp = np.logical_and((k <= self.pro_m), (miu >= 0.5))
        offspring[temp] = offspring[temp] + \
                               ((max_val[temp] - min_val[temp]) * (1 - (2 * (1 - miu[temp]) +
                                                        2 * (miu[temp] - 0.5) * offspring_scaled[temp] **
                                                                        (self.dis_m + 1)) ** (1 / (self.dis_m + 1))))

        offspring = np.clip(offspring, min_val, max_val)

        return offspring

    def clip(self, angle_view, min_val, max_val):
        top_index = angle_view > max_val
        low_index = angle_view < min_val

        angle_view[top_index] = 2 * max_val[top_index] - angle_view[top_index]
        angle_view[low_index] = 2 * min_val[low_index] - angle_view[low_index]

        return angle_view

    def crossover_mutation_by_chain(self, mating_pool, chain_type, choose_type):
        mating_pool_view = np.array([x.angle_view().copy() for x in mating_pool])
        mating_num, num_angle = mating_pool_view.shape

        mating_index = self.generate_mating_index(mating_num)
        offspring_view = np.zeros((0, num_angle))

        main_chain = np.where(chain_type == 1)[0]
        side_chain = np.where(chain_type == 0)[0]

        if choose_type == True:
            choose_angle = np.r_[main_chain, np.random.choice(side_chain, int(len(side_chain) / 10))]
        else:
            choose_angle = np.r_[np.random.choice(main_chain, int(len(main_chain) / 10)), side_chain]
        choose_num = len(choose_angle)

        for x in mating_index:
            beta = np.zeros(choose_num)
            miu = np.random.rand(choose_num)
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (self.dis_c + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (self.dis_c + 1))
            beta = beta * ((-1) ** np.random.randint(0, high=2, size=choose_num))
            beta[np.random.rand(choose_num) > self.pro_c] = 1

            avg = (mating_pool_view[x[0]] + mating_pool_view[x[1]]) / 2
            diff = (mating_pool_view[x[0]] - mating_pool_view[x[1]]) / 2


            calc_beta = np.ones(num_angle)
            calc_beta[choose_angle] = beta
            offspring = avg + calc_beta * diff
            offspring_view = np.vstack((offspring_view, offspring))

            offspring = avg - calc_beta * diff
            offspring_view = np.vstack((offspring_view, offspring))



        min_val = np.ones_like(offspring_view) * self.min_angles
        max_val = np.ones_like(offspring_view) * self.max_angles
        k = np.random.random(offspring_view.shape)
        miu = np.random.random(offspring_view.shape)
        temp = np.logical_and((k <= self.pro_m), (miu < 0.5))
        for x in choose_angle:
            temp[:, x] = False


        offspring_scaled = (offspring_view - min_val) / (max_val - min_val)
        offspring_scaled[np.isnan(offspring_scaled)] = 0
        offspring_scaled[np.isinf(offspring_scaled)] = 0

        offspring_view[temp] = offspring_view[temp] + \
                          ((max_val[temp] - min_val[temp]) * ((2 * miu[temp] + (1 - 2 * miu[temp]) *
                                                               (1 - offspring_scaled[temp])
                                                               ** (self.dis_m + 1)) ** (1 / (self.dis_m + 1)) - 1))
        temp = np.logical_and((k <= self.pro_m), (miu >= 0.5))
        for x in choose_angle:
            temp[:, x] = False

        offspring_view[temp] = offspring_view[temp] + \
                          ((max_val[temp] - min_val[temp]) * (1 - (2 * (1 - miu[temp]) +
                                                                   2 * (miu[temp] - 0.5) * offspring_scaled[temp] **
                                                                   (self.dis_m + 1)) ** (1 / (self.dis_m + 1))))

        offspring_view = np.clip(offspring_view, min_val, max_val)

        return offspring_view

