import os
import random

import numpy as np


class Protein(object):
    def __init__(self, obj_num, status, coder):
        self.obj_num = obj_num
        self.status = status
        self.coder = coder

        self.res = []
        self.obj = np.zeros(self.obj_num)
        self.to_evaluate = 0
        self.paretoRank = 1
        self.cd = 0

        self.pose = None
        self.protein_path = None

    def add_res(self, res_created):
        self.res.append(res_created)

    def angle_view(self):
        # return each residues' angles
        # and reshape these angles to 1-dim
        angles = []
        for x in self.res:
            angles = angles + x.angle.tolist()
        angles = np.array(angles)
        return angles

    def update_angle_from_view(self, angle):
        # the type of angle is numpy
        angle_index = 0
        for x in self.res:
            num_angle = x.num_angle
            x.angle = angle[angle_index: angle_index + num_angle].copy()
            angle_index = angle_index + num_angle

    def get_angles_field(self):
        self.max_angles = []  # where self.angles_max and self.angles_min are list for all angles limit
        self.min_angles = []  # the shape of them are (num_residues, the_most_residue_angles)
        for residue in self.res:
            res_max, res_min = self.coder.get_angle_field(residue)
            self.max_angles = self.max_angles + res_max
            self.min_angles = self.min_angles + res_min
        self.max_angles = np.array(self.max_angles)
        self.min_angles = np.array(self.min_angles)
        return self.max_angles, self.min_angles

    def get_chain_type(self):
        # chain_type has two elements, 0 and 1.
        # 0 represents this angle is side-chain angle
        # 1 represents this angle is main-chain angle
        chain_type = []
        for residue in self.res:
            chain_type = chain_type + [1, 1]  # phi and psi angle
            chain_type = chain_type + [0] * (residue.num_angle - 2)  # side-chain angle
        chain_type = np.array(chain_type, dtype=int)

        return chain_type

    def set_rosetta_pose(self, pose):
        self.pose = pose
    def return_rosetta_pose(self):
        return self.pose

    def copy(self):
        copy_protein = Protein(self.obj_num, self.status, self.coder)

        copy_protein.obj = self.obj.copy()
        copy_protein.to_evaluate = self.to_evaluate
        copy_protein.paretoRank = self.paretoRank
        copy_protein.cd = self.cd

        copy_res = []
        for x in self.res:
            copy_res.append(x.copy())
        copy_protein.res = copy_res
        if self.pose is not None:
            copy_protein.pose = self.pose.clone()
        if self.protein_path is not None:
            copy_protein.protein_path = self.protein_path

        return copy_protein


class Residue(object):
    def __init__(self, name, ss_type, angle, sidechain_num, predicted):
        self.name = name
        self.ss_type = ss_type
        self.angle = angle
        self.num_angle = len(self.angle)
        self.sidechain_num = sidechain_num
        self.predicted = predicted

    def get_angle(self, type):
        # The function is get the angle of residue,
        # and the type of angle should be in ["phi", "psi", "omega", "sidechain"]
        # When type is "phi", "psi" or "omega", the dtype of return value is a float.
        # When type is "sidechain", the dtype of return value is a including all side chain angles list.
        if type == 'phi':
            return self.angle[0]
        if type == 'psi':
            return self.angle[1]
        if type == 'omega':
            # return self.angle[2]
            return 180.0
        if type == 'sidechain':
            return self.angle[2:]
        else:
            RuntimeError('The type of residue angle should be in ["phi", "psi", "omega", "sidechain"]. '
                         'Invalid angle type: %s' % type)

    def copy(self):
        copy_residue = Residue(self.name, self.ss_type, self.angle, self.sidechain_num, self.predicted)
        copy_residue.angle = self.angle.copy()
        return copy_residue


class Coding(object):
    def __init__(self, config, protein_status):
        self.config = config
        self.protein_status = protein_status

        self.generate_angle_dict()  # generate the dictionary of angle include the name and angle number of amino acid
        self.generate_phi_psi_dict()  # generating the constraint of the psi and phi
        self.generate_sidechain_dict()  # generation the constraint of side-chain
        self.generate_decode_phi_psi_dict()  #

    # Generating the protein by the amino acid list
    def decoder_from_seq(self, data_path, protein):
        # data_path is the secondary structure file
        for x in open(data_path, 'r'):
            line = x.replace('\n', '')  # the file is in Linux, Unix and macOS format
            # line = x.replace('\r\n', '') # which is in windows format
            residue_name, second_struct = line.split(' ')
            sidechain_num = self.num_angle_dict[residue_name]
            angle = self.get_angle(residue_name, second_struct, sidechain_num)
            residue_add = Residue(residue_name, second_struct, angle, sidechain_num, None)
            protein.add_res(residue_add)
        return protein

    def decoder_from_logger(self, data_path, protein):
        # the format of protein solution is
        # residue second_structure phi psi omega sidechain1 sidechain2 ... sidechainN
        solution_file = open(os.path.join(data_path, 'solution.dat'), 'r')
        for x in solution_file:
            line = x.replace('\n', '')
            residue_name, second_struct = line.split(' ')[:2]
            sidechain_num = self.num_angle_dict[residue_name]
            angle = []
            for y in line.split(' ')[3:]:
                angle.append(float(y))
            angle = np.array(angle)
            residue_add = Residue(residue_name, second_struct, angle, sidechain_num, None)
            protein.add_res(residue_add)

        energy_file = open(os.path.join(data_path, 'energy.csv'), 'r')
        header = energy_file.readline().split(',')
        for x in energy_file:
            line = x.replace('\n', '')
            objs_value = line.split(',')
            for obj_index, value in enumerate(objs_value):
                protein.obj[obj_index] = float(value)
        return protein

    def encoder_to_logger(self, data_path, protein, save_all, current_gen):
        # the format of protein solution is
        # residue second_structure phi psi omega sidechain1 sidechain2 ... sidechainN

        # write protein solution file
        if save_all:
            solution_file = open(os.path.join(data_path, 'solution_{}.dat'.format(current_gen)), 'w')
        else:
            solution_file = open(os.path.join(data_path, 'solution.dat'), 'w')
        for residue in protein.res:
            write_line = '{}' + (len(residue.angle) + 1) * ' {}' + '\n'
            write_angles = np.insert(residue.angle, 2, 180.0)
            write_line = write_line.format(residue.name, *write_angles)
            solution_file.write(write_line)
        solution_file.close()

        # write energy file
        with open(os.path.join(data_path, 'energy.csv'), 'a') as energy_file:
            # energy_file = open(os.path.join(data_path, 'energy.csv'), 'a')
            write_line = protein.obj_num * '{},'
            write_line = write_line[:-1] + '\n'
            write_line = write_line.format(*protein.obj)
            energy_file.write(write_line)
            energy_file.close()

    def encoder_to_loggerforarchive(self, data_path, protein,protein_index):
        # the format of protein solution is
        # residue second_structure phi psi omega sidechain1 sidechain2 ... sidechainN
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        # write protein solution file
        solution_file = open(os.path.join(data_path, 'solution_{}.dat'.format(protein_index)), 'w')
        for residue in protein.res:
            write_line = '{} {}' + (len(residue.angle) + 1) * ' {}' + '\n'
            write_angles = np.insert(residue.angle, 2, 180.0)
            write_line = write_line.format(residue.name, residue.ss_type, *write_angles)
            solution_file.write(write_line)
        solution_file.close()

        # write energy file
        with open(os.path.join(data_path, 'energy.csv'), 'a') as energy_file:
            # energy_file = open(os.path.join(data_path, 'energy.csv'), 'a')
            write_line = protein.obj_num * '{},'
            write_line = write_line[:-1] + '\n'
            write_line = write_line.format(*protein.obj)
            energy_file.write(write_line)
            energy_file.close()

    def get_angle(self, residue_name, second_struct, sidechain_num):
        angle = []
        # generating the main-chain angle eg., phi psi
        [phi_low, phi_top] = self.decode_phi_dict[second_struct]  # get the angle constraint of phi
        angle.append(round(random.uniform(phi_low, phi_top), 3))  # randomly generating from interval [phi_low,phi_top]

        [psi_low, psi_top] = self.decode_psi_dict[second_struct]  # get the angle constraint of psi
        angle.append(round(random.uniform(psi_low, psi_top), 3))  # randomly generating from interval [psi_low,psi_top]

        # angle.append(180.0)
        # generating the side-chain angle
        for i in range(1, sidechain_num + 1):
            [sidechain_low, sidechain_top] = self.sidechain_dict[residue_name][i]
            angle.append(round(random.uniform(sidechain_low, sidechain_top), 3))
        return np.array(angle)

    def get_angle_field(self, residue):
        angle_max = []
        angle_min = []
        [phi_low, phi_top] = self.decode_phi_dict[residue.ss_type]
        angle_max.append(phi_top)
        angle_min.append(phi_low)

        [psi_low, psi_top] = self.decode_psi_dict[residue.ss_type]
        angle_max.append(psi_top)
        angle_min.append(psi_low)

        # angle_max.append(180.0)
        # angle_min.append(180.0)

        for i in range(1, residue.sidechain_num + 1):
            [sidechain_low, sidechain_top] = self.sidechain_dict[residue.name][i]
            angle_max.append(sidechain_top)
            angle_min.append(sidechain_low)

        return angle_max, angle_min

    def generate_angle_dict(self):
        # the relation between angle number and amino acid
        self.num_angle_dict = {}
        num_angle_config = self.config['num_angle']
        for x in num_angle_config:
            residue_name = num_angle_config[x].replace('\n', '').split(',')
            for y in residue_name:
                self.num_angle_dict[y] = int(x)

    def generate_phi_psi_dict(self):
        # the constraint of the main-chain angle

        # What phi&psi constraint dict format is
        # phi&psi_constraint = {
        #     'second_type': [low, up]
        # }
        # the type of low and up is float/
        self.phi_native_dict = {}
        self.phi_nonnative_dict = {}
        self.psi_native_dict = {}
        self.psi_nonnative_dict = {}

        def read_phi_psi(angle_type, statue):
            angle_constraint_dict = {}
            for second_type, field in self.config[angle_type][statue].items():
                low, up = field.split(',')
                angle_constraint_dict[second_type] = [float(low), float(up)]
            return angle_constraint_dict

        self.phi_native_dict = read_phi_psi('phi_constraint', 'native')
        self.phi_nonnative_dict = read_phi_psi('phi_constraint', 'non-native')
        self.psi_native_dict = read_phi_psi('psi_constraint', 'native')
        self.psi_nonnative_dict = read_phi_psi('psi_constraint', 'non-native')

    def generate_decode_phi_psi_dict(self):
        if self.protein_status == "native":
            self.decode_phi_dict = self.phi_native_dict
            self.decode_psi_dict = self.psi_native_dict
        elif self.protein_status == "nonnative":
            self.decode_phi_dict = self.phi_nonnative_dict
            self.decode_psi_dict = self.psi_nonnative_dict
        else:
            RuntimeError('The type of protein status must be in ["native", "nonnative"] '
                         'The invalid protein status: %s' % self.protein_status)

    def generate_sidechain_dict(self):
        # What sidechain constraint dict format is
        # sidechain_constraint = {
        #     'residue_name': {
        #         number_of_side_chain: [low, up]
        #     }
        # }
        # the type of low and up is float
        # the type of number_of_side_chain is int.
        self.sidechain_dict = {}
        sidechain_constraint_dict = self.config['sidechain_constraint']
        for res_name, angle_con in sidechain_constraint_dict.items():
            self.sidechain_dict[res_name] = {}
            for num in angle_con:
                low, up = angle_con[num].split(',')
                self.sidechain_dict[res_name][int(num)] = [float(low), float(up)]

    def get_num_angle(self, residue_name):
        return self.num_angle_dict[residue_name]
